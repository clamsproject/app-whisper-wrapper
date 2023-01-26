FROM clamsproject/clams-python-ffmpeg:latest

RUN apt-get install -y git

COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

CMD ["python3", "app.py", "--production"]
