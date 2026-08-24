FROM odsai/ecup26-matching-baseline:1.0

WORKDIR /solution
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=true \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-u", "run.py"]

