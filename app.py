import argparse
import tempfile
from typing import Dict, Union, List

import ffmpeg
import whisper
from clams import ClamsApp, Restifier
from lapps.discriminators import Uri
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

import metadata as app_metadata


class Whisper(ClamsApp):

    def __init__(self, model_size="medium"):
        self.whisper_model = whisper.load_model(model_size)
        super().__init__()

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        if not isinstance(mmif, Mmif):
            mmif: Mmif = Mmif(mmif)

        # get AudioDocuments with locations
        docs = [
            document
            for document in mmif.documents
            if document.at_type == DocumentTypes.AudioDocument
            and len(document.location) > 0
        ]
        conf = self.get_configuration(**parameters)
        files = {doc.id: doc.location_path() for doc in docs}

        transcripts = self._run_whisper(files)

        for file, transcript in zip(files, transcripts):
            # convert transcript to MMIF view
            view: View = mmif.new_view()
            self.sign_view(view, conf)
            view.new_contain(DocumentTypes.TextDocument)
            view.new_contain(Uri.TOKEN)
            view.new_contain(AnnotationTypes.TimeFrame, timeUnit=app_metadata.timeunit, document=file)
            view.new_contain(AnnotationTypes.Alignment)
            self._whisper_to_textdocument(
                transcript, view, mmif.get_document_by_id(file)
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
                raw_token = word
                tok_start = char_offset
                tok_end = tok_start + len(raw_token)
                char_offset += len(raw_token) + len(' ')
                token = view.new_annotation(Uri.TOKEN, word=raw_token, start=tok_start, end=tok_end, document=f"{view.id}:{textdoc.id}")
                tf_start = word["start"]
                tf_end = word["end"]
                tf = view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=tf_start, end=tf_end)
                view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)

    def _run_whisper(self, files: Dict[str, str]) -> List[dict]:
        """
        Run Whisper on each audio file.

        :param files: dict of {AudioDocument.id : physical file location}
        :return: A list of Whisper transcriptions in dict format
        """
        transcripts = []
        # make a temporary dir for whisper-ready audio files
        audio_tmpdir = tempfile.TemporaryDirectory()

        for audio_docid, audio_fname in files.items():
            resampled_audio_fname = f"{audio_tmpdir.name}/{audio_docid}_16kHz.wav"
            ffmpeg.input(audio_fname).output(
                resampled_audio_fname, ac=1, ar=16000
            ).run()
            transcripts.append(self.whisper_model.transcribe(audio=resampled_audio_fname, word_timestamps=True))
        return transcripts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parser.add_argument(
        "--model_size",
        action="store",
        default="medium",
        help="specify Whisper model size (small/medium/large)",
    )
    parsed_args = parser.parse_args()

    whisper_flask = Restifier(
        Whisper(parsed_args.model_size), port=int(parsed_args.port)
    )
    if parsed_args.production:
        whisper_flask.serve_production()
    else:
        whisper_flask.run()
