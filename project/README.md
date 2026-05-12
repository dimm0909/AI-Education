# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

## 1. Паспорт проекта

- **Название проекта:** `BikeFlow: прогноз спроса на городские велосипеды`
- **Автор:** `<ФИО студента>`
- **Группа:** `<Группа>`
- **Контакт:** `<@telegram или e-mail>`

- **Краткое описание:**
  Проект решает задачу прогноза количества аренд велосипедов на следующий час по погодным и календарным признакам.
  Используется открытый датасет UCI Bike Sharing Dataset.
  Реализованы baseline и улучшенная модель, а также FastAPI-сервис с endpoint’ами `/health` и `/predict`.

---

## 2. Структура проекта

- `requirements.txt` — зависимости.
- `report.md` — отчёт по проекту.
- `self-checklist.md` — самопроверка по критериям курса.
- `notebooks/` — ноутбуки с EDA и сравнением моделей.
- `src/` — основной код:
  - `src/data/` — загрузка/валидация датасета, генерация sample;
  - `src/features/` — генерация признаков;
  - `src/models/` — обучение и оценка моделей;
  - `src/service/` — FastAPI приложение;
  - `src/train.py` — точка входа обучения.
- `data/` — данные:
  - `data/raw/hour.csv` — полный UCI-датасет;
  - `data/sample/hour_sample.csv` — уменьшенная выборка для быстрого запуска.
- `configs/` — конфиги обучения и сервиса.
- `tests/` — unit/sanity тесты.
- `artifacts/` — сохранённая модель и метрики.

---

## 3. Требования и установка

### 3.1. Требования

- Python `>=3.10`

### 3.2. Установка окружения

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. (Опционально) Скачать полный открытый датасет

Если `data/raw/hour.csv` отсутствует, скачайте датасет командой:

```bash
cd project
python -m src.data.download_uci
```

### 4.2. (Опционально) Пересобрать sample-данные

```bash
cd project
python -m src.data.make_sample --input data/raw/hour.csv --output data/sample/hour_sample.csv --rows 1200
```

### 4.3. Обучение модели

```bash
cd project
python -m src.train --config configs/train.yaml
```

После обучения артефакты появятся в:
- `artifacts/model.joblib`
- `artifacts/metrics.json`

### 4.4. Запуск сервиса (FastAPI)

```bash
cd project
python -m src.service
```

Сервис поднимается на `http://0.0.0.0:8000`.

Ключевые endpoint’ы:
- `GET /health`
- `POST /predict`

Проверка `health`:

```bash
curl http://127.0.0.1:8000/health
```

Пример запроса к `predict`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "season": 3,
    "yr": 1,
    "mnth": 7,
    "hr": 8,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.62,
    "atemp": 0.59,
    "hum": 0.43,
    "windspeed": 0.19
  }'
```

---

## 5. Данные

Источник: открытый датасет **UCI Bike Sharing Dataset**.

- Ссылка: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>
- Формат: табличные данные (hourly).
- Целевая переменная: `cnt` (количество аренд за час).

В репозитории хранятся:
- полный CSV (`data/raw/hour.csv`);
- sample CSV (`data/sample/hour_sample.csv`) для быстрых запусков и тестов.

---

## 6. Тесты

```bash
cd project
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -q
```

Тестами проверяется:
- корректная загрузка/разделение данных;
- обучение и сохранение артефактов;
- работоспособность `/health` и `/predict`.

---

## 7. Демонстрация на защите

На защите демонстрируется:

1. Запуск обучения `python -m src.train` и вывод выбранной модели + метрик.
2. Запуск API `python -m src.service`.
3. Запросы `GET /health` и `POST /predict` с 2 разными сценариями (например, рабочее утро и дождливый вечер).
4. Сравнение baseline и improved модели по `artifacts/metrics.json`.

---

## 8. Ограничения и дальнейшая работа

Текущие ограничения:
- модель прогнозирует только на 1 час вперёд;
- используется только один публичный датасет;
- отсутствует онлайн-мониторинг drift.

Дальнейшее развитие:
- добавить градиентный бустинг (CatBoost/LightGBM) и тюнинг;
- добавить логирование latency/ошибок в отдельное хранилище;
- добавить периодическое переобучение модели по расписанию.

---

## 9. Оценка проекта

Проект закрывает ключевые критерии шаблона:
- есть рабочий сервис `/health` + `/predict`;
- `/predict` использует сохранённую модель, а не заглушку;
- есть EDA/эксперименты и сравнение baseline vs improved;
- есть конфиги, тесты, инструкции запуска и отчёт.
