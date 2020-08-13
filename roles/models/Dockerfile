FROM python:3.7-slim-buster

RUN mkdir app
WORKDIR /home/app

RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt
RUN echo ". /opt/venv/bin/activate" >> ~/.bashrc

COPY *.py ./
COPY config ./config
COPY cta ./cta
COPY images/tfrecords ./images/tfrecords
COPY libml ./libml
COPY third_party ./third_party
COPY fully_supervised ./fully_supervised

RUN mkdir experiments
RUN mkdir logs

EXPOSE 8080