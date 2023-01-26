FROM clamsproject/clams-python-ffmpeg:latest


COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/jianfch/stable-ts.git

CMD ["python3", "app.py", "--production"]
