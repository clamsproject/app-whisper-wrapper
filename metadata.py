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
whisper_version = [line.strip().rsplit('==')[-1]
                   for line in open('requirements.txt').readlines() if re.match(r'^openai-whisper==', line)][0]
whisper_lang_list = f"https://github.com/openai/whisper/blob/{whisper_version}/whisper/tokenizer.py"
whisper_argument_delegation_prefix = "(from openai-whisper CLI) "

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

    # some delegated parameters from the underlying whisper interface, copied from whisper's transcribe.py (cli())
    metadata.add_parameter(
        name="model",
        type='string',
        default="turbo",
        description="(from openai-whisper CLI) name of the Whisper model to use"
    )
    metadata.add_parameter(
        name="language",
        type='string',
        default='',
        description="(from openai-whisper CLI) language spoken in the audio, specify None to perform language detection"
    )
    metadata.add_parameter(
        name="task",
        type='string',
        default="transcribe",
        choices=["transcribe", "translate"],
        description="(from openai-whisper CLI) whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')"
    )
    metadata.add_parameter(
        name="initialPrompt",
        type='string',
        default='',
        description="(from openai-whisper CLI) optional text to provide as a prompt for the first window."
    )
    metadata.add_parameter(
        name="conditionOnPreviousText",
        type='string',
        default=True,
        description="(from openai-whisper CLI) if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop"
    )
    metadata.add_parameter(
        name="noSpeechThreshold",
        type='number',
        default=0.6,
        description="(from openai-whisper CLI) if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence"
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
