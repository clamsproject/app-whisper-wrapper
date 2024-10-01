import argparse
import logging
import re
import time
from typing import Union

import whisper
from clams import ClamsApp, Restifier
from lapps.discriminators import Uri
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

import metadata as app_metadata


class WhisperWrapper(ClamsApp):
    
    model_size_alias = {
        't': 'tiny', 
        'b': 'base', 
        's': 'small', 
        'm': 'medium', 
        'l': 'large', 
        'l2': 'large-v2', 
        'l3': 'large-v3',
        'tu': 'turbo',
    }

    def __init__(self):
        super().__init__()
        self.whisper_models = {}
        self.model_usage = {}

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
        lang = parameters['modelLang'].split('-')[0]
        size = parameters['modelSize']
        if size in self.model_size_alias:
            size = self.model_size_alias[size]
        if lang == 'en' and not size.startswith('large'):
            size += '.en'
        self.logger.debug(f'whisper model: {size} ({lang})')
        # taken from the default values for decoder arguments in whisper cli
        transcribe_args = {'best_of': 5, 
                           'beam_size': 5,
                           "patience": None,
                           "length_penalty": None,
                           "suppress_tokens": "-1",
                           }
        for param in self.metadata.parameters:
            if param.description.startswith(app_metadata.whisper_argument_delegation_prefix):
                pattern = re.compile(r'(?<!^)(?=[A-Z])')
                transcribe_args[pattern.sub('_', param.name).lower()] = parameters[param.name]
        # this is due to the limitation of the SDK that doesn't allow `None` for a default value for a parameter
        # (setting default to None is a reserved action to make the parameter optional)
        # So as a workaround, the default is set to an empty string, then to match the behavior of the whisper cli,
        # it's converted to None here.
        if transcribe_args['initial_prompt'] == '':
            transcribe_args['initial_prompt'] = None
        if size not in self.whisper_models:
            self.logger.debug(f'Loading model {size}')
            t = time.perf_counter()
            self.whisper_models[size] = whisper.load_model(size)
            self.logger.debug(f'Load time: {time.perf_counter() - t:.2f} seconds\n')
            self.model_usage[size] = False
        if not self.model_usage[size]:
            whisper_model = self.whisper_models.get(size)
            self.model_usage[size] = True
            cached = True
        else:
            self.logger.debug(f'Loading model {size} to avoid memory conflict')
            t = time.perf_counter()
            whisper_model = whisper.load_model(size)
            self.logger.debug(f'Load time: {time.perf_counter() - t:.2f} seconds\n')
            cached = False

        for doc in docs:
            transcribe_args['language'] = lang if len(lang) > 0 else None
            transcribe_args['word_timestamps'] = True
            self.logger.debug(f'whisper model args: {transcribe_args}')
            self.logger.debug('Transcribing audio')
            t = time.perf_counter()
            transcript = whisper_model.transcribe(audio=doc.location_path(nonexist_ok=False), 
                                                  **transcribe_args)
            self.logger.debug(f'Transcription time: {time.perf_counter() - t:.2f} seconds\n')
            # keep the original language parameter, that might have region code as well
            self.logger.debug(f'Preparing a new transcript view for {doc.id}')
            t = time.perf_counter()
            lang_to_record = parameters['modelLang'] if len(parameters['modelLang']) > 0 else transcript['language']
            view: View = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(DocumentTypes.TextDocument, _lang=lang_to_record)
            view.new_contain(Uri.TOKEN)
            view.new_contain(AnnotationTypes.TimeFrame, timeUnit=app_metadata.timeunit, document=doc.id)
            view.new_contain(AnnotationTypes.Alignment)
            self.logger.debug(f'View preparation time: {time.perf_counter() - t:.2f} seconds\n')
            self.logger.debug(f'Translating into MMIF')
            t = time.perf_counter()
            self._whisper_to_textdocument(transcript, view, mmif.get_document_by_id(doc.id), lang=lang_to_record)
            self.logger.debug(f'Translation time: {time.perf_counter() - t:.2f} seconds\n')
        
        if size in self.model_usage and cached == True:
                self.model_usage[size] = False
        return mmif

    @staticmethod
    def _whisper_to_textdocument(transcript, view, source_audio_doc, lang):
        raw_text = transcript["text"]
        # make annotations
        textdoc = view.new_textdocument(text=raw_text, lang=lang)
        view.new_annotation(AnnotationTypes.Alignment, source=source_audio_doc.id, target=textdoc.id)
        char_offset = 0
        for segment in transcript["segments"]:
            # skip empty segments
            if len(segment["words"]) == 0 or len(segment["text"]) == 0:
                continue
            token_ids = []
            for word in segment["words"]:
                raw_token = word["word"].strip()
                tok_start = raw_text.index(raw_token, char_offset)
                tok_end = tok_start + len(raw_token)
                char_offset = tok_end
                token = view.new_annotation(Uri.TOKEN, word=raw_token, start=tok_start, end=tok_end, document=f'{view.id}:{textdoc.id}')
                token_ids.append(token.id)
                tf_start = int(word["start"] * 1000)
                tf_end = int(word["end"] * 1000)
                tf = view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=tf_start, end=tf_end)
                view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)
            view.new_annotation(Uri.SENTENCE, targets=token_ids, text=segment['text'].strip())


def get_app():
    return WhisperWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    # create the app instance
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()