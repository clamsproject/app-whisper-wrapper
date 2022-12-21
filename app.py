import argparse
import tempfile
import whisper
from typing import Dict, Union

import ffmpeg
from clams import ClamsApp, Restifier, AppMetadata
from lapps.discriminators import Uri
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

__version__ = '0.2.4' # TODO: Update
Model = 'large'
whisper_model = whisper.load_model(Model)

class Whisper(ClamsApp):

    timeunit = 'seconds'
    token_boundary = ' '
    timeunit_conv = {'milliseconds': 1000, 'seconds': 1}

    def _appmetadata(self):
        metadata = AppMetadata(
            name="Whisper Wrapper",
            description="A CLAMS wrapper for Whisper-based ASR software originally developed by OpenAI,"
                        " Wrapped software can be found at https://github.com/clamsproject/app-whisper-wrapper. ",
            app_version=__version__,
            analyzer_version="v4",
            analyzer_license="UNKNOWN",
            app_license="Apache 2.0",
            identifier=None,  # TODO: add
            url=None,  # TODO: add
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
        docs = [document for document in mmif.documents
                if document.at_type == DocumentTypes.AudioDocument and len(document.location) > 0]
        conf = self.get_configuration(**parameters)
        files = {doc.id: doc.location_path() for doc in docs}
        tf_src_view = {}

        transcripts = self._run_whisper(files)

        for file, transcript in zip(files, transcripts):
            # convert transcript to MMIF view
            view: View = mmif.new_view()
            self.sign_view(view, conf)
            view.new_contain(DocumentTypes.TextDocument)
            view.new_contain(Uri.TOKEN)
            view.new_contain(AnnotationTypes.TimeFrame,
                             timeUnit=self.timeunit,
                             document=file)
            view.new_contain(AnnotationTypes.Alignment)
            self._whisper_to_textdocument(transcript, view, mmif.get_document_by_id(file))

        return mmif

    def _whisper_to_textdocument(self, transcript, view, source_audio_doc):
        # join tokens
        raw_text = self.token_boundary.join([token for token in transcript['text']])
        # make annotations
        textdoc = self._create_td(view, raw_text)
        self._create_align(view, source_audio_doc, textdoc)
        char_offset = 0
        for index, segment in enumerate(transcript['segments']):
            raw_tokens = self.token_boundary.join([token for token in segment['tokens']])
            tok_start = char_offset
            tok_end = tok_start + len(raw_tokens)
            char_offset += len(raw_tokens) + len(self.token_boundary)
            # Whisper doesn't return single tokens?
            tokens = self._create_token(view, raw_tokens, tok_start, tok_end, f'{view.id}:{textdoc.id}')
            tf_start = segment['start']
            tf_end = segment['end']
            tf = self._create_tf(view, tf_start, tf_end)
            self._create_align(view, tf, tokens)  # counting one for TextDoc-AudioDoc alignment

    @staticmethod
    def _create_td(parent_view: View, doc: str) -> Document:
        td = parent_view.new_textdocument(doc)
        return td

    @staticmethod
    def _create_token(parent_view: View, word: str, start: int, end: int, source_doc_id: str) -> Annotation:
        token = parent_view.new_annotation(Uri.TOKEN,
                                           word=word,
                                           start=start,
                                           end=end,
                                           document=source_doc_id)
        return token

    @staticmethod
    def _create_tf(parent_view: View, start: int, end: int) -> Annotation:
        # unlike _create_token, parent document is encoded in the contains metadata of TimeFrame
        tf = parent_view.new_annotation(AnnotationTypes.TimeFrame,
                                        frameType='speech',
                                        start=start,
                                        end=end)
        return tf

    @staticmethod
    def _create_align(parent_view: View, source: Annotation, target: Annotation) -> Annotation:
        align = parent_view.new_annotation(AnnotationTypes.Alignment,
                                           source=source.id,
                                           target=target.id)
        return align

    @staticmethod
    def _run_whisper(files: Dict[str, str]) -> list[dict]:
        """
        Run Whisper on each audio file.

        :param files: dict of {AudioDocument.id : physical file location}
        :return: A list of Whisper transcriptions in dict format
        """
        transcripts = []
        # make a temporary dir for whisper-ready audio files
        audio_tmpdir = tempfile.TemporaryDirectory()

        for audio_docid, audio_fname in files.items():
            resampled_audio_fname = f'{audio_tmpdir.name}/{audio_docid}_16kHz.wav'
            ffmpeg.input(audio_fname).output(resampled_audio_fname, ac=1, ar=16000).run()
            transcripts.append(whisper.transcribe(whisper_model, resampled_audio_fname))
        return transcripts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        action='store',
        default='5000',
        help='set port to listen'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='run gunicorn server'
    )
    parsed_args = parser.parse_args()

    whisper_app = Whisper()