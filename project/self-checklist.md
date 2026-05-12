# Самопроверка проекта (self-checklist)

| #  | Критерий                                                                 | Да/Нет (студент) | Где смотреть / комментарий |
|----|---------------------------------------------------------------------------|------------------|-----------------------------|
| 1  | Сервис запускается по инструкциям из `project/README.md` и работает      | ✅               | `README.md`, `src/service/__main__.py`, `src/service/app.py` |
| 2  | Endpoint `/predict` использует **реальную модель**, а не заглушку        | ✅               | `src/service/app.py` (загрузка `artifacts/model.joblib`) |
| 3  | Есть EDA и хотя бы один эксперимент с метриками                          | ✅               | `notebooks/exp01_eda_baseline.ipynb`, `artifacts/metrics.json`, `report.md` |
| 4  | Есть baseline и улучшенная модель, есть **сравнение по метрикам**        | ✅               | `src/models/trainer.py`, `artifacts/metrics.json` |
| 5  | Код не свален в один ноутбук: есть внятная структура в `src/`            | ✅               | `src/data/`, `src/features/`, `src/models/`, `src/service/` |
| 6  | Есть Dockerfile **или** понятный сценарий развёртывания без Docker       | ✅               | `README.md` (установка, обучение, запуск API) |
| 7  | Есть `.env.example` и **нет** в репозитории реальных секретов/паролей    | ✅               | `configs/.env.example` |
| 8  | Реализованы логи/наблюдаемость (хотя бы консольные логи + `/health`)     | ✅               | `src/utils/logging_utils.py`, `src/service/app.py` (`/health`) |
| 9  | В `report.md` **обоснован выбор финальной модели** по результатам экспериментов | ✅          | `report.md`, разделы 4-5 |
| 10 | `project/README.md` и `report.md` позволяют понять сценарий демонстрации | ✅               | `README.md`, `report.md` |

---

## Подсчёт баллов

Итог по самооценке: **10/10**.
