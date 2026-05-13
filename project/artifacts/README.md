# Артефакты BikeFlow

Файл модели можно получить двумя способами:

1. Скачать предобученную модель:

```bash
cd project
python -m src.data.download_pretrained_model
```

Ссылка источника: <https://disk.yandex.ru/d/fUbi6bAAbJJTOA>

2. Обучить модель локально:

```bash
cd project
python -m src.train --config configs/train.yaml
```

После этого создаются:

- `model.joblib` — сериализованная финальная модель + метаданные (создаётся локально после обучения);
- `metrics.json` — метрики baseline/improved и выбранная финальная модель.
- `figures/` — базовые графики обучения и сравнения моделей:
  - `metrics_regression.png`
  - `metrics_classification.png`
  - `best_model_actual_vs_pred.png`
  - `best_model_residuals.png`
  - `catboost_tuning_rmse.png`

В репозитории хранится только `metrics.json`, а `model.joblib` исключён из git, чтобы не хранить тяжёлые бинарные артефакты в истории.
