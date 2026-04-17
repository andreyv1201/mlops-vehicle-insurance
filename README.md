# Vehicle Insurance Premium Prediction

MLOps MVP для задачи регрессии страховой премии (`PREMIUM`) на датасете автострахования Ethiopian Insurance Corporation. Проект включает классический end-to-end pipeline для обучения моделей и отдельный MVP pipeline для пошагового обновления, инференса и генерации отчетов.

## Что есть в проекте

Проект поддерживает **два сценария работы**: полный обучающий pipeline через `main.py` и MVP pipeline через `run.py` с режимами `update`, `inference` и `summary`.

- `main.py` — классический end-to-end pipeline: загрузка данных, EDA, подготовка признаков, обучение нескольких моделей, валидация и сохранение артефактов.
- `run.py` — MVP pipeline с режимами `update`, `inference`, `summary` и версионированием моделей через реестр.
- `src/serve.py` — Flask API для инференса после обучения модели.

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