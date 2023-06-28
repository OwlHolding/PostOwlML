FROM debian:bullseye
FROM python:3.10-bullseye

COPY . /

RUN pip3 install -r requirements.txt

RUN python3 -c "__import__('sentence_transformers').SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')"
RUN python3 -c "__import__('sentence_transformers').SentenceTransformer('clip-ViT-B-32')"
RUN python3 -c "__import__('transformers').AutoTokenizer.from_pretrained('sentence-transformers/clip-ViT-B-32-multilingual-v1')"


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]