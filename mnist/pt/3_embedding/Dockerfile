FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["run.py"]