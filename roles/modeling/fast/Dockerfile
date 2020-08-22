FROM python:3.8-slim

WORKDIR /modeling

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY model.py ./model.py

CMD python model.py
