"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

import re

from clams.appmetadata import AppMetadata
from lapps.discriminators import Uri
from mmif import DocumentTypes, AnnotationTypes

timeunit = "seconds"


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="Whisper Wrapper",
        description="A CLAMS wrapper for Whisper-based ASR software originally developed by OpenAI.",
        app_license="Apache 2.0",
        identifier="whisper-wrapper", 
        url="https://github.com/clamsproject/app-whisper-wrapper",
        analyzer_version=[line.strip().rsplit('==')[-1] 
                          for line in open('requirements.txt').readlines() if re.match(r'^openai-whisper==', line)][0],
        analyzer_license="MIT",
    )
    metadata.add_input(DocumentTypes.AudioDocument)
    metadata.add_output(DocumentTypes.TextDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit=timeunit)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(Uri.TOKEN)
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    sys.stdout.write(appmetadata().jsonify(pretty=True))
