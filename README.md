# Vehicle Insurance Premium Prediction

MLOps MVP для задачи регрессии страховой премии (`PREMIUM`) на датасете автострахования Ethiopian Insurance Corporation. Проект включает классический end-to-end pipeline для обучения моделей и отдельный MVP pipeline для пошагового обновления, инференса и генерации отчетов.

## Что есть в проекте

Проект поддерживает **два сценария работы**: полный обучающий pipeline через `main.py` и MVP pipeline через `run.py` с режимами `update`, `inference` и `summary`.

- `main.py` — классический end-to-end pipeline: загрузка данных, EDA, подготовка признаков, обучение нескольких моделей, валидация и сохранение артефактов.
- `run.py` — MVP pipeline с режимами `update`, `inference`, `summary` и версионированием моделей через реестр.
- `src/serve.py` — Flask API для инференса после обучения модели.
- `doc/` — документация для проверки задания 2: ожидаемые баллы и описание задачи.

## Документация

- [Ожидаемые баллы и статусы реализации](doc/grade.md)
- [Описание задачи, данных и ML pipeline](doc/task.md)

## Установка

```bash
pip install -r requirements.txt
```

Требования:

- Python 3.10+
- Данные в папке `data/`:
  - `motor_data11-14lats.csv`
  - `motor_data14-2018.csv`

## Использование

### Полный pipeline обучения

```bash
python main.py
```

Что делает pipeline:

1. Загружает и объединяет два CSV-файла.
2. Строит EDA-отчеты и графики в `reports/`.
3. Выполняет подготовку данных и feature engineering.
4. Обучает модели `LinearRegression`, `RandomForest`, `XGBoost`, `CatBoost`.
5. Сравнивает модели по метрикам и выбирает лучшую по `R2`.
6. Выполняет кросс-валидацию и latency-check.
7. Сохраняет артефакты в `models/`.

### MVP pipeline

```bash
# Запустить пайплайн на следующем батче данных
python run.py --mode update

# Применить production модель к новым данным
# Возвращает путь к CSV с добавленной колонкой предсказания
python run.py --mode inference --file "./path/to/new_data.csv"

# Сгенерировать сводный JSON-отчет по данным, качеству и моделям
python run.py --mode summary
```

## ML API

После обучения через `main.py` можно запустить Flask-сервис:

```bash
python src/serve.py
```

Доступные endpoints:

- `GET /health` — проверка статуса сервиса и наличия загруженной модели.
- `POST /predict` — предсказание для одного объекта.
- `POST /predict_batch` — предсказания для батча объектов.

Пример запроса:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 150000, 2015, 5, 0, 1.8, 3, 12, 2, 365, 3]}'
```

## Структура проекта

```text
main.py                      классический E2E pipeline
run.py                       точка входа для MVP pipeline
requirements.txt

src/
    data_extraction.py       загрузка и объединение данных
    eda.py                   анализ данных и генерация графиков
    data_preparation.py      подготовка признаков
    train.py                 обучение моделей
    evaluate.py              сравнение моделей по метрикам
    validate.py              кросс-валидация и latency-check
    mvp_pipeline.py          update / inference / summary + model registry
    serve.py                 Flask API для инференса

doc/
    grade.md                 ожидаемые баллы и статусы реализации
    task.md                  описание задачи, данных и ML pipeline

artifacts/
    state.json               состояние обработки батчей
    data_meta.jsonl          мета-информация по батчам
    data_quality.jsonl       логи контроля качества данных

models/
    best_model.pkl           лучшая модель из main.py
    scaler.pkl               scaler для API
    encoders.pkl             LabelEncoderы
    feature_cols.pkl         список признаков
    registry.json            реестр версий моделей MVP pipeline
    model_vXXX.joblib        версии моделей из run.py --mode update

reports/
    premium_distribution.png
    correlation_matrix.png
    categorical_features.png
    premium_by_insr_type.png
    descriptive_stats.csv
    model_comparison.csv
    model_comparison.png
    inference_YYYYMMDD_HHMMSS.csv
    summary_latest.json
```

## Метрики

При сравнении моделей используются следующие метрики:

- `MAE`
- `RMSE`
- `R2`
- `MAPE`

Лучшая модель в baseline pipeline выбирается по максимальному значению `R2`.

## Как работает update в MVP

Поток `python run.py --mode update` выглядит так:

1. Берет следующий необработанный батч из исходных CSV-файлов с `chunksize=20000`.
2. Сохраняет батч в `raw_store/` и мета-информацию в `artifacts/data_meta.jsonl`.
3. Загружает накопленные батчи из `raw_store/`.
4. Выполняет очистку данных по порогу заполненности `min_quality_non_null_ratio`.
5. Обучает набор моделей и выбирает лучшую.
6. Сохраняет новую версию модели в `models/model_vXXX.joblib`.
7. Обновляет `models/registry.json`.

## CI/CD workflow для задания 2

Для задания 2 проект доработан как продолжение MVP из задания 1. Выбран сценарий
**CRON-инкрементального обучения**, потому что в `run.py --mode update` уже есть
потоковая обработка батчей, состояние чтения данных, накопление `raw_store/` и
реестр версий моделей.

Workflow находится в `.github/workflows/mlops-ci.yml` и выполняет обязательные
части задания:

- запускается автоматически при `push` и `pull_request`;
- устанавливает окружение через `actions/setup-python` и `pip install -r requirements.txt`;
- запускает тесты ML-системы через `pytest`;
- выполняет обучение модели командой `python run.py --mode update`;
- сохраняет лог обучения `artifacts/training.log`;
- публикует артефакты GitHub Actions: лог, реестр моделей, сериализованную модель,
  состояние сборщика данных и summary-отчет.

Дополнительно workflow поддерживает запуск по расписанию:

```yaml
schedule:
  - cron: "0 6 * * 1"
```

При scheduled-запуске используется `actions/cache`: директории `artifacts/`,
`raw_store/`, `models/` и `reports/` восстанавливаются из предыдущего запуска и
сохраняются после нового обучения. Поэтому каждый CRON-запуск продолжает обработку
следующего батча, а не начинает pipeline с нуля.

### Управление обучением из YAML

В workflow параметры обучения задаются через переменные окружения:

```yaml
MLOPS_BATCH_SIZE: "150"
MLOPS_BATCHES_PER_UPDATE: "1"
MLOPS_MODEL_CANDIDATES: "LinearRegression,DecisionTree,RandomForest"
MLOPS_RANDOM_FOREST_ESTIMATORS: "30"
```

Эти же параметры можно передать локально:

```bash
python run.py --mode update \
  --batch-size 150 \
  --batches-per-update 1 \
  --model-candidates LinearRegression,DecisionTree,RandomForest \
  --random-forest-estimators 30
```

### Развертывание на GitHub Actions

1. Убедиться, что в репозитории есть файлы:
   `README.md`, `requirements.txt`, `.github/workflows/mlops-ci.yml`, `run.py`,
   `src/`, `scripts/` и `tests/`.
2. Загрузить проект в GitHub-репозиторий.
3. Вкладка **Actions** автоматически покажет workflow `MLOps CI/CD`.
4. Сделать `push` или открыть `pull request`: workflow установит зависимости,
   выполнит тесты, обучит модель и сохранит артефакты.
5. Для ручного запуска открыть **Actions -> MLOps CI/CD -> Run workflow** и при
   необходимости изменить `batch_size`, `batches_per_update` или список моделей.
6. После успешного запуска открыть run workflow и скачать архив
   `mlops-training-artifacts-*`.

Если исходные CSV-файлы не загружены в GitHub из-за размера или приватности данных,
workflow автоматически создает небольшой синтетический датасет командой
`python scripts/generate_sample_data.py`. Это позволяет проверяющему воспроизвести
установку окружения, тестирование, обучение и сохранение артефактов без локальных
файлов `data/*.csv`. При наличии реальных CSV в репозитории pipeline использует их.

### Состав артефактов CI/CD

- `artifacts/training.log` — текстовый лог обучения модели;
- `models/model_vXXX.joblib` — сериализованная модель, которую можно загрузить через `joblib`;
- `models/registry.json` — история версий моделей, метрик и гиперпараметров;
- `artifacts/data_collector_state.joblib` — сериализованное состояние сборщика батчей;
- `artifacts/state.json`, `artifacts/data_meta.jsonl`, `artifacts/data_quality.jsonl` —
  состояние потока данных и контроль качества;
- `reports/summary_latest.json` — dashboard-like summary истории обучения в JSON.
