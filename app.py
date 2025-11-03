import argparse
import logging
import re
import time
from typing import Union

import whisper
from clams import ClamsApp, Restifier
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
        if lang and lang not in whisper.tokenizer.LANGUAGES:
            raise ValueError(f"unsupported language code: {lang}. Check whisper/tokenizer.py")

        size = parameters['modelSize']
        if size in self.model_size_alias:
            size = self.model_size_alias[size]

        # tiny, base, small, medium have English-only models. large and turbo do not.
        # tiny.en, base.en, small.en, medium.en:
        EN_MODELS = {'tiny', 'base', 'small', 'medium'}
        if lang == 'en' and size in EN_MODELS:
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
            view.new_contain(AnnotationTypes.Token)
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
        # Build text by concatenating words and create tokens simultaneously
        all_text_parts = []
        all_tokens_data = []
        char_offset = 0
        
        for segment in transcript["segments"]:
            segment_token_data = []
            for word in segment["words"]:
                raw_token = word["word"].strip()
                if not raw_token:  # skip empty tokens
                    continue
                    
                tok_start = char_offset
                tok_end = tok_start + len(raw_token)
                
                # Store token data for later annotation creation
                segment_token_data.append({
                    'word': raw_token,
                    'char_start': tok_start,
                    'char_end': tok_end,
                    'time_start': word["start"],
                    'time_end': word["end"]
                })
                
                all_text_parts.append(raw_token)
                char_offset = tok_end + 1  # +1 for space
            
            all_tokens_data.append(segment_token_data)
        
        # Build the full text document
        raw_text = " ".join(all_text_parts)
        
        # Create text document
        textdoc = view.new_textdocument(text=raw_text, lang=lang)
        view.new_annotation(AnnotationTypes.Alignment, source=source_audio_doc.id, target=textdoc.id)
        
        # Create all token and sentence annotations
        for segment_token_data in all_tokens_data:
            if not segment_token_data:  # skip empty segments
                continue
                
            token_ids = []
            sentence_words = []
            
            for token_data in segment_token_data:
                # Create token annotation
                token = view.new_annotation(
                    AnnotationTypes.Token, 
                    text=token_data['word'], 
                    start=token_data['char_start'], 
                    end=token_data['char_end'], 
                    document=f'{textdoc.id}'
                )
                token_ids.append(token.id)
                sentence_words.append(token_data['word'])
                
                # Create timeframe and alignment
                tf_start = int(token_data['time_start'] * 1000)
                tf_end = int(token_data['time_end'] * 1000)
                tf = view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=tf_start, end=tf_end)
                view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)
            
            # Create sentence annotation
            sentence_text = " ".join(sentence_words)
            view.new_annotation(AnnotationTypes.Sentence, targets=token_ids, text=sentence_text)


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