import argparse
import tempfile
import stable_whisper
from typing import Dict, Union, List

import ffmpeg
from clams import ClamsApp, Restifier, AppMetadata
from lapps.discriminators import Uri
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

__version__ = "0.1.1"


class Whisper(ClamsApp):

    timeunit = "seconds"
    token_boundary = " "
    timeunit_conv = {"milliseconds": 1000, "seconds": 1}

    def __init__(self, model_size="medium"):
        self.whisper_model = stable_whisper.load_model(model_size)
        super().__init__()

    def _appmetadata(self):
        metadata = AppMetadata(
            name="Whisper Wrapper",
            description="A CLAMS wrapper for Whisper-based ASR software originally developed by OpenAI,"
            " Wrapped software can be found at https://github.com/clamsproject/app-whisper-wrapper. ",
            app_version=__version__,
            analyzer_version="v4",
            analyzer_license="MIT",
            app_license="Apache 2.0",
            identifier="https://apps.clams.ai/aapb-pua-kaldi-wrapper/{__version__}",  # TODO: add
            url="https://github.com/clamsproject/app-whisper-wrapper",
        )
        metadata.add_input(DocumentTypes.AudioDocument)
        metadata.add_output(DocumentTypes.TextDocument)
        metadata.add_output(AnnotationTypes.TimeFrame, timeUnit=self.timeunit)
        metadata.add_output(AnnotationTypes.Alignment)
        metadata.add_output(Uri.TOKEN)
        return metadata

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
            view.new_contain(
                AnnotationTypes.TimeFrame, timeUnit=self.timeunit, document=file
            )
            view.new_contain(AnnotationTypes.Alignment)
            self._whisper_to_textdocument(
                transcript, view, mmif.get_document_by_id(file)
            )

        return mmif

    def _whisper_to_textdocument(self, transcript, view, source_audio_doc):
        raw_text = transcript["text"]
        # make annotations
        textdoc = self._create_td(view, raw_text)
        self._create_align(view, source_audio_doc, textdoc)
        char_offset = 0
        for index, segment in enumerate(transcript["segments"]):
            for t1, t2 in zip(
                segment["whole_word_timestamps"],
                segment["whole_word_timestamps"][1:] + [None],
            ):
                raw_token = t1["word"]
                tok_start = char_offset
                tok_end = tok_start + len(raw_token)
                char_offset += len(raw_token) + len(self.token_boundary)
                token = self._create_token(
                    view, raw_token, tok_start, tok_end, f"{view.id}:{textdoc.id}"
                )
                tf_start = t1["timestamp"]
                if t2:
                    tf_end = t2["timestamp"]
                else:
                    tf_end = segment["end"]
                tf = self._create_tf(view, tf_start, tf_end)
                self._create_align(
                    view, tf, token
                )  # counting one for TextDoc-AudioDoc alignment

    @staticmethod
    def _create_td(parent_view: View, doc: str) -> Document:
        td = parent_view.new_textdocument(doc)
        return td

    @staticmethod
    def _create_token(
        parent_view: View, word: str, start: int, end: int, source_doc_id: str
    ) -> Annotation:
        token = parent_view.new_annotation(
            Uri.TOKEN, word=word, start=start, end=end, document=source_doc_id
        )
        return token

    @staticmethod
    def _create_tf(parent_view: View, start: int, end: int) -> Annotation:
        # unlike _create_token, parent document is encoded in the contains metadata of TimeFrame
        tf = parent_view.new_annotation(
            AnnotationTypes.TimeFrame, frameType="speech", start=start, end=end
        )
        return tf

    @staticmethod
    def _create_align(
        parent_view: View, source: Annotation, target: Annotation
    ) -> Annotation:
        align = parent_view.new_annotation(
            AnnotationTypes.Alignment, source=source.id, target=target.id
        )
        return align

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
            transcripts.append(self.whisper_model.transcribe(resampled_audio_fname))
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
