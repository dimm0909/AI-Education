# Тесты BikeFlow

## Состав

- `test_dataset.py` — загрузка датасета и split.
- `test_training.py` — обучение моделей и сохранение артефактов.
- `test_service.py` — `/health` и `/predict` через FastAPI TestClient.

## Запуск

```bash
cd project
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -q
```
