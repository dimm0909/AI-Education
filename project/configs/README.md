# Конфиги BikeFlow

- `train.yaml` — пути к данным, параметры split, гиперпараметры RandomForest/CatBoost/LightGBM/XGBoost, тюнинг CatBoost, пути артефактов и папка с графиками.
- `service.yaml` — host/port FastAPI и путь к модели.
- `.env.example` — пример переменных окружения.

Можно менять параметры через эти файлы без правки кода.
