# Запуск, обучение и приёмка

## Где обучать

Для настоящего дообучения нужен GPU. Google Colab с T4 годится для короткого
эксперимента на `items_human` + `matches`; полный прогон лучше выполнять на
A100/H100 (RunPod, Vast.ai, сервер команды или облачная VM). Полные 4.1 ГБ
`items.parquet` и LLM-разметка нужны для второго этапа; первый этап полностью
работает на 214 МБ `items_human.parquet` и ручных метках.

## Google Colab

Включите `Runtime → Change runtime type → GPU`. В первой ячейке:

```python
!git clone <URL_вашего_репозитория> /content/ozon-matching
%cd /content/ozon-matching
!pip install -r requirements.txt
!mkdir -p /content/data /content/models
!wget -q --show-progress https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matches.parquet -O /content/data/matches.parquet
!wget -q --show-progress https://storage.yandexcloud.net/ozon-ecup-2026/Matching/items_human.parquet -O /content/data/items_human.parquet
!nvidia-smi
```

Сохраните чекпоинт на Google Drive, иначе он пропадёт после остановки Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
!python prepare_model.py --trust-remote-code --output /content/models/gte
!python train.py --items-path /content/data/items_human.parquet --matches-path /content/data/matches.parquet --model-path /content/models/gte --trust-remote-code --output-dir /content/drive/MyDrive/ozon-matching/gte-human --epochs 2 --batch-size 32 --max-length 256
```

На T4 начните с `--batch-size 16` при `CUDA out of memory`; на A100/H100 обычно
подойдут 64/128. Для второй стадии скачайте `items.parquet` и
`matches_llm.parquet`, скопируйте лучший human checkpoint в новую директорию и
выполните ту же команду с этими путями, одним коротким epoch. Выбор checkpoint
делайте только по валидации на ручной разметке.

## GPU-сервер

На Ubuntu нужны NVIDIA driver, Docker или Python 3.11. После копирования
репозитория и данных команды выше идентичны. На H100 включается bf16
автоматически; начните с `--batch-size 128`, при длинных атрибутах снизьте до 64.
При инференсе интернет не нужен: модель уже должна быть в
`models/product-matcher`.

## Инференс

```bash
python -u run.py \
  --items_path items_human.parquet \
  --matches_path matches.parquet \
  --output_path submit.csv \
  --model-path models/product-matcher \
  --trust-remote-code --batch-size 128 --max-length 256
```

На CPU это **запустится**: код выберет CPU и сохранит корректный CSV. Но 365
тыс. пар с cross-encoder будут считаться долго, поэтому CPU — только для
smoke-test на 100–1000 парах. На соревновательном H100 это штатный режим.
`train_lexical.py` обучает быстрый CPU-компаньон на числах/артикулах/атрибутах;
его стоит добавлять в ансамбль только после сравнения на holdout.

## Офлайн-приёмка сабмита

```bash
python - <<'PY'
import pandas as pd
m = pd.read_parquet('matches.parquet').head(1000)
m.drop(columns='target').to_parquet('check_matches.parquet', index=False)
PY
python -u run.py -i items_human.parquet -m check_matches.parquet -o check_submit.csv --model-path models/product-matcher --trust-remote-code
python - <<'PY'
import pandas as pd
source, out = pd.read_parquet('check_matches.parquet'), pd.read_csv('check_submit.csv')
assert list(out.columns) == ['id1', 'id2', 'predict']
assert len(out) == len(source) == 1000
assert out[['id1','id2']].equals(source[['id1','id2']])
assert out.predict.notna().all() and out.predict.between(0, 1).all()
print('PASS')
PY
```

Повторите проверку в контейнере без интернета. В финальном архиве нужны `run.py`,
`src/`, `models/product-matcher/`, все remote-code файлы GTE и `metadata.json`.

