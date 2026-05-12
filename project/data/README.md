# Данные проекта BikeFlow

## Используемые файлы

- `raw/hour.csv` — полный открытый датасет UCI Bike Sharing (hourly).
- `raw/day.csv` — дневной вариант того же датасета (в проекте не используется для обучения API).
- `sample/hour_sample.csv` — уменьшенная подвыборка из `hour.csv` для быстрого запуска и тестов.

## Источник

UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>

## Как перескачать

```bash
cd project
python -m src.data.download_uci
```

## Как пересобрать sample

```bash
cd project
python -m src.data.make_sample --input data/raw/hour.csv --output data/sample/hour_sample.csv --rows 1200
```
