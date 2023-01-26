FROM brandeisllc/whisper

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg

COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

CMD ["python3", "app.py", "--production"]
