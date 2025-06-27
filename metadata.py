"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

import re

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from lapps.discriminators import Uri
from mmif import DocumentTypes, AnnotationTypes

timeunit = "milliseconds"
default_model_size = "tiny"
default_model_lang = ''
whisper_version = [line.strip().rsplit('==')[-1]
                   for line in open('requirements.txt').readlines() if re.match(r'^openai-whisper==', line)][0]
whisper_lang_list = f"https://github.com/openai/whisper/blob/{whisper_version}/whisper/tokenizer.py"
whisper_argument_delegation_prefix = "(from whisper CLI) "

# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="Whisper Wrapper",
        description="A CLAMS wrapper for Whisper-based ASR software originally developed by OpenAI.",
        app_license="Apache 2.0",
        identifier="whisper-wrapper", 
        url="https://github.com/clamsproject/app-whisper-wrapper",
        analyzer_version=whisper_version,
        analyzer_license="MIT",
    )
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    metadata.add_output(DocumentTypes.TextDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit=timeunit)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(Uri.TOKEN)
    metadata.add_output(Uri.SENTENCE)
    
    metadata.add_parameter(
        name='modelSize', 
        description='The size of the model to use. When `modelLang=en` is given, for non-`large` and non -`turbo` models, '
                    'English-only models will be used instead of multilingual models for speed and accuracy. '
                    '(For `large` and `turbo` models, English-only models are not available.) (also can be given as alias: '
                    'tiny=t, base=b, small=s, medium=m, large=l, large-v2=l2, large-v3=l3, turbo=tu)',
        type='string',
        choices=['tiny', 't', 'base', 'b', 'small', 's', 'medium', 'm', 'large', 'l', 'large-v2', 'l2', 'large-v3', 'l3', 'turbo', 'tu'],
        default=default_model_size
    )

    metadata.add_parameter(
        name='modelLang', 
        description=f'Language of the model to use, accepts two- or three-letter ISO 639 language codes, '
                    f'however Whisper only supports a subset of languages. If the language is not supported, '
                    f'error will be raised.For the full list of supported languages, see {whisper_lang_list} . In '
                    f'addition to the langauge code, two-letter region codes can be added to the language code, '
                    f'e.g. "en-US" for US English. Note that the region code is only for compatibility and recording '
                    f'purpose, and Whisper neither detects regional dialects, nor use the given one for transcription. '
                    f'When the langauge code is not given, Whisper will run in langauge detection mode, and will use '
                    f'first few seconds of the audio to detect the language.',
        type='string',
        default=default_model_lang
    )
    # and some delegated parameters from the underlying whisper interface, copied from whisper's transcribe.py (cli())
    metadata.add_parameter(name="task", type='string', default="transcribe", choices=["transcribe", "translate"], 
                           description=whisper_argument_delegation_prefix + "whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    metadata.add_parameter(name="initialPrompt", type='string', default='',
                           description=whisper_argument_delegation_prefix + "optional text to provide as a prompt for the first window.")
    metadata.add_parameter(name="conditionOnPreviousText", type='boolean', default=True, 
                           description=whisper_argument_delegation_prefix + "if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    metadata.add_parameter(name="noSpeechThreshold", type='number', default=0.6, 
                           description=whisper_argument_delegation_prefix + "if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
