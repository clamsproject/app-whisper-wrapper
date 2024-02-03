import argparse
import logging
from typing import Union

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

    def _annotate(self, mmif: Union[str, dict, Mmif], whisper_language: str = "None", **parameters) -> Mmif:
        if not isinstance(mmif, Mmif):
            mmif: Mmif = Mmif(mmif)

        # try to get AudioDocuments
        docs = mmif.get_documents_by_type(DocumentTypes.AudioDocument)
        # and if none found, try VideoDocuments
        if not docs:
            docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        conf = self.get_configuration(**parameters)
        whisper_model = self.whisper_models.get(conf['modelSize'], None)
        whisper_language = conf['modelLang']

        self.logger.debug(f'whisper model: {conf["modelSize"]}')
        if whisper_model is None:
            self.logger.debug(f'model not cached, downloading now')
            whisper_model = whisper.load_model(conf['modelSize'])
            self.whisper_models[conf['modelSize']] = whisper_model
        for doc in docs:
            # transcript = whisper_model.transcribe(audio=doc.location_path(nonexist_ok=False), word_timestamps=True)
            if whisper_language == "None":
                transcript = whisper_model.transcribe(audio=doc.location_path(nonexist_ok=False), word_timestamps=True)
            else:
                transcript = whisper_model.transcribe(language=whisper_language, audio=doc.location_path(nonexist_ok=False), word_timestamps=True)

            view: View = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(DocumentTypes.TextDocument)
            view.new_contain(Uri.TOKEN)
            view.new_contain(AnnotationTypes.TimeFrame, timeUnit=app_metadata.timeunit, document=doc.id)
            view.new_contain(AnnotationTypes.Alignment)
            self._whisper_to_textdocument(
                transcript, view, mmif.get_document_by_id(doc.id), whisper_language
            )
        return mmif

    @staticmethod
    def _whisper_to_textdocument(transcript, view, source_audio_doc, whisper_language):
        raw_text = transcript["text"]
        # make annotations
        if whisper_language != "None":
            textdoc = view.new_textdocument(raw_text, lang=whisper_language)
        else:
            textdoc = view.new_textdocument(raw_text)

        # textdoc = view.new_textdocument(raw_text, lang=whisper_language)
        view.new_annotation(AnnotationTypes.Alignment, source=source_audio_doc.id, target=textdoc.id)
        char_offset = 0
        for segment in transcript["segments"]:
            token_ids = []
            for word in segment["words"]:
                raw_token = word["word"].strip()
                tok_start = raw_text.index(raw_token, char_offset)
                tok_end = tok_start + len(raw_token)
                char_offset = tok_end
                token = view.new_annotation(Uri.TOKEN, word=raw_token, start=tok_start, end=tok_end, document=f"{view.id}:{textdoc.id}")
                token_ids.append(token.id)
                tf_start = word["start"]
                tf_end = word["end"]
                tf = view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=tf_start, end=tf_end)
                view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)
            view.new_annotation(Uri.SENTENCE, targets=token_ids, text=segment['text'])


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
