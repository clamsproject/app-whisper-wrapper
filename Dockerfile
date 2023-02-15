FROM clamsproject/clams-python-ffmpeg:latest

RUN apt-get install -y git

COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python -c "import whisper; whisper.load_model('tiny'); whisper.load_model('small'); whisper.load_model('medium'); whisper.load_model('large'); whisper.load_model('huge')"

CMD ["python3", "app.py", "--production"]
