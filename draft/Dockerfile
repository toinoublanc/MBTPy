FROM debian:latest
RUN mkdir /app
WORKDIR /app
COPY requirements.text api/main.py models data/truc.csv ./
RUN apt-get update && apt-get install python3-pip -y && pip install -r requirements.txt
EXPOSE 8000
CMD uvicorn main.api --host 0.0.0.0