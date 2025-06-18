import argparse
import logging
import re
import time
from typing import Union

# == NEW IMPORTS ==
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# will try these later:
# from transformers import WhisperProcessor, WhisperForConditionalGeneration

from clams import ClamsApp, Restifier
from lapps.discriminators import Uri
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

import metadata as app_metadata


class WhisperWrapperHF(ClamsApp):
    
    model_size_alias = {
        't': 'tiny', 
        'b': 'base', 
        's': 'small', 
        'm': 'medium', 
        'l': 'large', 
        'l2': 'large-v2', 
        'l3': 'large-v3',
        'tu': 'large-v3-turbo',
    }

    def __init__(self):
        super().__init__()
        # Instead of storing raw Whisper models, we store references to Hugging Face pipelines or model objects
        self.hf_whisper_pipelines = {}
        self.model_usage = {}

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        if not isinstance(mmif, Mmif):
            mmif: Mmif = Mmif(mmif)

        # 1. Pick the audio doc(s)
        docs = mmif.get_documents_by_type(DocumentTypes.AudioDocument)
        if not docs:
            docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)

        # 2. Map model size to HF model name (example: openai/whisper-small)
        lang = parameters['modelLang'].split('-')[0]  # e.g. en-US -> en
        size = parameters['modelSize']
        if size in self.model_size_alias:
            size = self.model_size_alias[size]

        # If language == en and not large, use .en version (just like in OpenAI’s logic)
        # For instance, openai/whisper-small.en 
        # This is optional, depends on usage.
        hf_model_name = f"openai/whisper-{size}"
        if lang == "en" and not size.startswith("large"):
            hf_model_name += ".en"
        
        self.logger.debug(f"Hugging Face Whisper model: {hf_model_name} (lang={lang})")

        # 3. Build a `generate_kwargs` dict (some parameters from the old code)
        # Not all arguments (e.g. best_of, patience) map perfectly to HF
        # so use them as an approximation or omit them.
        transcribe_args = {
            "num_beams": parameters.get("beamSize", 5),     # was `beam_size` in the old code
            "length_penalty": parameters.get("lengthPenalty", 1.0),
            # etc. Add more as needed/available...
        }
        
        # Convert whisper-specific parameters (if relevant) into pipeline’s generate_kwargs or call-time arguments.
        # For example, “initial_prompt” requires more advanced usage with `prompt_ids` from the processor. 
        # https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor.get_prompt_ids
        if parameters.get("initialPrompt") == "":
            initial_prompt = None
        else:
            initial_prompt = parameters.get("initialPrompt")
        # might need to do something like: generate_kwargs["prompt_ids"] = ...
        
        # 4. Create or retrieve a pipeline. If not used, create a new pipeline
        # We could specify a device. If we want to run on CPU, ignore device=...
        # device_index = 0 if torch.cuda.is_available() else -1
        device_index = parameters.get("deviceIndex", -1)
        
        if hf_model_name not in self.hf_whisper_pipelines:
            self.logger.debug(f"Loading HF pipeline for model {hf_model_name}")
            t = time.perf_counter()
            # Use “automatic-speech-recognition” pipeline
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=hf_model_name,
                device=device_index,
                generate_kwargs=transcribe_args,  # forwarded to model.generate()
            )
            self.logger.debug(f"Load time: {time.perf_counter() - t:.2f} seconds\n")
            self.hf_whisper_pipelines[hf_model_name] = asr_pipeline
            self.model_usage[hf_model_name] = False

        # If the pipeline is currently “in use,” can either re-load it to avoid concurrency issues or just reuse. 
        if not self.model_usage[hf_model_name]:
            asr_pipeline = self.hf_whisper_pipelines[hf_model_name]
            self.model_usage[hf_model_name] = True
            cached = True
        else:
            self.logger.debug(f"Reloading pipeline for {hf_model_name} to avoid memory conflict.")
            t = time.perf_counter()
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=hf_model_name,
                device=device_index,
                generate_kwargs=transcribe_args
            )
            self.logger.debug(f"Load time: {time.perf_counter() - t:.2f} seconds\n")
            cached = False

        # 5. Transcribe each document
        for doc in docs:
            self.logger.debug(f"whisper model args: {transcribe_args}")
            # For Hugging Face, pass `return_timestamps="word"` to get word-level timestamps
            # (as opposed to chunk-level or no timestamps).
            self.logger.debug("Transcribing audio via HF pipeline")
            t = time.perf_counter()
            res = asr_pipeline(
                doc.location_path(nonexist_ok=False),
                return_timestamps="word"
            )
            self.logger.debug(f"Transcription time: {time.perf_counter() - t:.2f} seconds\n")

            # 6. Convert HF’s pipeline output to the format `_whisper_to_textdocument` expects
            # The HF pipeline output has shape like:
            # {
            #   'text': 'full text',
            #   'chunks': [ { 'timestamp':(start, end), 'text':'word' }, ... ]
            # }
            # We want something like:
            # {
            #   'text': 'full text',
            #   'segments': [
            #       {
            #         'start':  ...,
            #         'end':    ...,
            #         'text':   ...
            #         'words': [
            #             { 'word': '...', 'start': ..., 'end': ... }, ...
            #         ]
            #       }
            #   ]
            # }
            hug_transcript = {
                "text": res["text"],
                "segments": []
            }

            words = []
            if "chunks" in res:
                # Collect each chunk as a “word” for the original data structure
                for ch in res["chunks"]:
                    wstart, wend = ch["timestamp"]
                    words.append({
                        "word": ch["text"].strip(),
                        "start": wstart,
                        "end": wend,
                    })
            
            # Option 1: Put all words in a single “segment”
            # If want multiple segments, need to do more advanced chunking
            if len(words) > 0:
                segment = {
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "text": res["text"],
                    "words": words
                }
                hug_transcript["segments"].append(segment)

            # 7. Prepare a new view in the MMIF
            self.logger.debug(f"Preparing a new transcript view for {doc.id}")
            t = time.perf_counter()
            lang_to_record = parameters['modelLang'] if len(parameters['modelLang']) > 0 else lang
            view: View = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(DocumentTypes.TextDocument, _lang=lang_to_record)
            view.new_contain(Uri.TOKEN)
            view.new_contain(AnnotationTypes.TimeFrame, timeUnit=app_metadata.timeunit, document=doc.id)
            view.new_contain(AnnotationTypes.Alignment)
            self.logger.debug(f"View preparation time: {time.perf_counter() - t:.2f} seconds\n")

            # 8. Insert the transcript into the MMIF structure
            self.logger.debug("Translating HF result into MMIF format")
            t = time.perf_counter()
            self._whisper_to_textdocument(hug_transcript, view, mmif.get_document_by_id(doc.id), lang=lang_to_record)
            self.logger.debug(f"Translation time: {time.perf_counter() - t:.2f} seconds\n")
        
        # Mark model as no longer in use
        if hf_model_name in self.model_usage and cached is True:
            self.model_usage[hf_model_name] = False

        return mmif

    @staticmethod
    def _whisper_to_textdocument(transcript, view, source_audio_doc, lang):
        """
        Essentially the same as the original method, except now ‘transcript’ is
        shaped according to how we remapped the Hugging Face pipeline output.
        """
        raw_text = transcript["text"]
        # make annotations
        textdoc = view.new_textdocument(text=raw_text, lang=lang)
        view.new_annotation(AnnotationTypes.Alignment, source=source_audio_doc.id, target=textdoc.id)
        char_offset = 0

        for segment in transcript["segments"]:
            # skip empty segments
            if not segment["words"] or not segment["text"]:
                continue

            token_ids = []
            for word_dict in segment["words"]:
                raw_token = word_dict["word"]
                # find this token’s position in the entire text
                tok_start = raw_text.index(raw_token, char_offset)
                tok_end = tok_start + len(raw_token)
                char_offset = tok_end

                token = view.new_annotation(
                    Uri.TOKEN,
                    word=raw_token,
                    start=tok_start,
                    end=tok_end,
                    document=f'{view.id}:{textdoc.id}'
                )
                token_ids.append(token.id)

                tf_start = int(word_dict["start"] * 1000)
                tf_end = int(word_dict["end"] * 1000)
                tf = view.new_annotation(
                    AnnotationTypes.TimeFrame, 
                    frameType="speech", 
                    start=tf_start, 
                    end=tf_end
                )
                view.new_annotation(
                    AnnotationTypes.Alignment, 
                    source=tf.id, 
                    target=token.id
                )
            # Create a sentence annotation linking all tokens in this segment
            view.new_annotation(
                Uri.SENTENCE,
                targets=token_ids,
                text=segment['text'].strip()
            )


def get_app():
    return WhisperWrapperHF()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parser.add_argument("--deviceIndex", type=int, default=-1, help="set a specific device index if using GPU")
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
