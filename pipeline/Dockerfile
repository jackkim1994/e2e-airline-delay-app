FROM --platform=linux/x86_64 python:3.11

WORKDIR /423-project

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "ml_pipeline.py"]

