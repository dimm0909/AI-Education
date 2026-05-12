# Исходный код BikeFlow

## Основные модули

- `train.py` — запуск обучения и сохранения артефактов.
- `data/`
  - `download_uci.py` — скачивание открытого датасета;
  - `download_pretrained_model.py` — скачивание предобученной модели с Яндекс.Диска;
  - `dataset.py` — загрузка, валидация, split по времени;
  - `make_sample.py` — генерация уменьшенной выборки.
- `features/transform.py` — генерация циклических признаков и препроцессор.
- `models/trainer.py` — baseline + improved модель, метрики, выбор финальной модели.
- `service/`
  - `app.py` — FastAPI приложение (`/health`, `/predict`);
  - `__main__.py` — запуск uvicorn.
- `utils/` — конфиг и логирование.

## Точки входа

```bash
python -m src.train
python -m src.data.download_pretrained_model
python -m src.service
```
