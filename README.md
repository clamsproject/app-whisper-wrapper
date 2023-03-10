# app-whisper-wrapper

This repository wraps a Whisper ASR instance to a CLAMS app. It is intended to subsume the functionality of the CLAMS Kaldi wrapper (https://github.com/clamsproject/app-aapb-pua-kaldi-wrapper).

## Requirements 

- Docker to run the code as a server in a Docker container
- curl or some other utility to send of an HTTP request to the server
- Python 3 with the `clams-python` module installed to create the MMIF input

## Building and running the Docker image

From the project directory, run the following in your terminal to build the Docker image from the included Dockerfile:

```bash
docker build . -f Dockerfile -t whisper
```

Then to create a Docker container using that image, run:

```bash
docker run -v /path/to/data/directory:/data -p <port>:5000 whisper
```

where /path/to/data/directory is the location of your media files or MMIF objects and "<port>" is the port number you want your container to be listening to.
If your data directory consists of media files (either video or audio), you will first want to convert them to MMIF objects, which can be done in the command line via
```bash
clams source audio:<filename> > input.mmif
```
(Make sure you installed the same `clams-python` package version specified in the [`requirements.txt`](requirements.txt).)

You can then call the service with

```bash
curl -d @input.mmif -s localhost:<port> > output.mmif
```
