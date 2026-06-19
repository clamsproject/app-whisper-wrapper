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

# Per-checkpoint version pins for reproducibility. Keys are the model names
# accepted by ``whisper.load_model()``; values are the SHA256 of the checkpoint
# file, taken from the download URLs in whisper's ``_MODELS`` dict
# (whisper/__init__.py). whisper verifies downloads against these hashes, so
# they uniquely identify the weights independently of the library version
# (recorded separately as ``analyzer_version``). Update when bumping
# ``openai-whisper`` if upstream re-publishes any checkpoint.
whisper_model_versions = {
    "tiny.en":        "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03",
    "tiny":           "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "base.en":        "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead",
    "base":           "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    "small.en":       "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872",
    "small":          "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "medium.en":      "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
    "medium":         "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    "large-v1":       "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
    "large-v2":       "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    "large-v3":       "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    "large-v3-turbo": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
}

# Short-name aliases accepted for the ``model`` parameter, each mapped to a
# canonical name in ``whisper_model_versions``. Kept separate so the canonical
# dict stays the single source of version pins; both dicts feed the ``model``
# parameter's ``choices``. ``large``/``turbo`` live here (not in the canonical
# dict) because they are whisper's own short aliases for ``large-v3`` /
# ``large-v3-turbo`` (identical checkpoints).
model_size_alias = {
    't': 'tiny',
    'b': 'base',
    's': 'small',
    'm': 'medium',
    'l': 'large-v3',
    'l2': 'large-v2',
    'l3': 'large-v3',
    'tu': 'large-v3-turbo',
    'large': 'large-v3',
    'turbo': 'large-v3-turbo',
}

# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="Whisper Wrapper",
        description="A CLAMS wrapper for Whisper-based ASR software originally developed by OpenAI.",
        app_license="Apache 2.0",
        identifier="whisper-wrapper", 
        url="https://github.com/clamsproject/app-whisper-wrapper",
        analyzer_versions=whisper_model_versions,
        analyzer_license="MIT",
        # Approximate VRAM (MB). min ~= "tiny"; typ ~= the default "turbo".
        # Larger checkpoints need more ("large" ~10GB); refine if needed.
        est_gpu_mem_min=1500,
        est_gpu_mem_typ=6000,
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
        choices=list(whisper_model_versions.keys()) + list(model_size_alias.keys()),
        description="(from openai-whisper CLI) name of the Whisper model to use. "
                    "Canonical names are the keys of this app's `analyzer_versions`; "
                    "short aliases (e.g. `tu`/`turbo` for `large-v3-turbo`) are also accepted."
    )
    metadata.add_parameter(
        name="language",
        type='string',
        default='',
        description="(from openai-whisper CLI) language spoken in the audio, specify None to perform language detection. "
                    "For the list of supported language codes, see "
                    "https://github.com/openai/whisper/blob/04f449b8a437f1bbd3dba5c9f826aca972e7709a/whisper/tokenizer.py"
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
