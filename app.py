import argparse
import logging
import tempfile
from typing import Union

import ffmpeg
import whisper
from clams import ClamsApp, Restifier
from lapps.discriminators import Uri
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

import metadata as app_metadata


class WhisperWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        self.whisper_models = {}

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        if not isinstance(mmif, Mmif):
            mmif: Mmif = Mmif(mmif)

        # try to get AudioDocuments
        docs = mmif.get_documents_by_type(DocumentTypes.AudioDocument)
        # and if none found, try VideoDocuments
        if not docs:
            docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        conf = self.get_configuration(**parameters)
        whisper_model = self.whisper_models.get(conf['modelSize'], None)
        self.logger.debug(f'whisper model: {conf["modelSize"]}')
        if whisper_model is None:
            self.logger.debug(f'model not cached, downloading now')
            whisper_model = whisper.load_model(conf['modelSize'])
            self.whisper_models[conf['modelSize']] = whisper_model
        for doc in docs:
            audio_tmpdir = tempfile.TemporaryDirectory()
            resampled_audio_fname = f"{audio_tmpdir.name}/{doc.id}_16kHz.wav"
            self.logger.debug(f'starting processing of {doc.location_path()}')
            ffmpeg.input(doc.location_path(nonexist_ok=False)).output(
                resampled_audio_fname, ac=1, ar=16000
            ).run()
            transcript = whisper_model.transcribe(audio=resampled_audio_fname, word_timestamps=True)
            view: View = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(DocumentTypes.TextDocument)
            view.new_contain(Uri.TOKEN)
            view.new_contain(AnnotationTypes.TimeFrame, timeUnit=app_metadata.timeunit, document=doc.id)
            view.new_contain(AnnotationTypes.Alignment)
            self._whisper_to_textdocument(
                transcript, view, mmif.get_document_by_id(doc.id)
            )
        return mmif

    @staticmethod
    def _whisper_to_textdocument(transcript, view, source_audio_doc):
        raw_text = transcript["text"]
        # make annotations
        textdoc = view.new_textdocument(raw_text)
        view.new_annotation(AnnotationTypes.Alignment, source=source_audio_doc.id, target=textdoc.id)
        char_offset = 0
        for segment in transcript["segments"]:
            for word in segment["words"]:
                raw_token = word["word"]
                tok_start = char_offset
                tok_end = tok_start + len(raw_token)
                char_offset += len(raw_token) + len(' ')
                token = view.new_annotation(Uri.TOKEN, word=raw_token, start=tok_start, end=tok_end, document=f"{view.id}:{textdoc.id}")
                tf_start = word["start"]
                tf_end = word["end"]
                tf = view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=tf_start, end=tf_end)
                view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen" )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    # create the app instance
    app = WhisperWrapper()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
