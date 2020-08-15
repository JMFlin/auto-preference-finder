FROM python:3.7-slim

#EXPOSE 8080 
EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py ./app.py

#CMD streamlit run --server.port 8080 --server.enableCORS false app.py
CMD streamlit run --server.port 8501 --server.enableCORS false app.py
