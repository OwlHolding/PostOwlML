FROM debian:bullseye
FROM python:3.9.16-bullseye

COPY . /

RUN pip3 install -r requirements.txt
RUN python3 -c "__import__('nltk').download('stopwords')"
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]