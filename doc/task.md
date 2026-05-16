# Описание задачи и ML pipeline

## Постановка задачи

Проект решает задачу регрессии для ML-системы обработки потоковых данных.
Целевая переменная - `PREMIUM`, то есть страховая премия по договору
автострахования. По признакам автомобиля, типа страхования и параметрам договора
модель прогнозирует ожидаемый размер премии.

Проект является продолжением задания 1. В задании 2 добавлен CI/CD workflow для
автоматического запуска тестов, обучения, инкрементального обновления модели и
сохранения артефактов в GitHub Actions.

## Данные

В качестве исходных данных используются CSV-файлы из директории `data/`:

- `motor_data11-14lats.csv`;
- `motor_data14-2018.csv`.

Файлы относятся к данным автострахования Ethiopian Insurance Corporation. В
репозиторий реальные CSV не добавляются, потому что они могут быть большими и
потенциально приватными. Для воспроизводимой проверки на GitHub Actions добавлен
скрипт `scripts/generate_sample_data.py`, который создает небольшой синтетический
датасет с той же схемой колонок.

Ключевые поля:

- `PREMIUM` - целевая переменная;
- `INSURED_VALUE` - страховая стоимость;
- `INSR_TYPE` - тип страхования;
- `PROD_YEAR` - год выпуска автомобиля;
- `SEATS_NUM` - число мест;
- `CARRYING_CAPACITY` - грузоподъемность;
- `TYPE_VEHICLE`, `MAKE`, `USAGE` - категориальные признаки автомобиля и сценария использования;
- `INSR_BEGIN`, `INSR_END` - даты начала и окончания страхового периода.

## Потоковая обработка данных

Основной MVP pipeline запускается командой:

```bash
python run.py --mode update
```

Логика update:

1. Считывается следующий батч из CSV-файлов через `pandas.read_csv(..., chunksize=...)`.
2. Информация о батче сохраняется в `artifacts/data_meta.jsonl`.
3. Сам батч сохраняется в `raw_store/batch_XXXXX.csv`.
4. Состояние чтения потока сохраняется в `artifacts/state.json`.
5. Сериализованное состояние сборщика данных сохраняется в `artifacts/data_collector_state.joblib`.
6. Все накопленные батчи загружаются из `raw_store/`.
7. Выполняется контроль качества данных и обучение новой версии модели.

Размер батча и число батчей за один update управляются параметрами:

```bash
python run.py --mode update \
  --batch-size 150 \
  --batches-per-update 1
```

В GitHub Actions эти параметры задаются из YAML через переменные окружения
`MLOPS_BATCH_SIZE` и `MLOPS_BATCHES_PER_UPDATE`.

## Data quality

Data quality реализован в `src/mvp_pipeline.py` в методе `assess_and_clean_data`.

Проверки и преобразования:

- рассчитывается доля пропусков по всем данным;
- рассчитывается non-null ratio по каждой колонке;
- колонки с заполненностью ниже `min_quality_non_null_ratio` удаляются;
- целевая колонка `PREMIUM` сохраняется, если она есть в данных;
- строки без `PREMIUM` удаляются;
- строки с неположительным значением `PREMIUM` удаляются;
- результаты проверки сохраняются в `artifacts/data_quality.jsonl`.

Порог качества управляется параметром:

```bash
python run.py --mode update --min-quality-non-null-ratio 0.6
```

## Обработка признакового пространства

В MVP pipeline признаки обрабатываются через `ColumnTransformer` и `Pipeline` из
scikit-learn.

Числовые признаки:

- определяются через `X.select_dtypes(include=[np.number])`;
- пропуски заполняются медианой через `SimpleImputer(strategy="median")`;
- масштабируются через `StandardScaler`.

Категориальные признаки:

- определяются как все нечисловые колонки;
- приводятся к строковому типу;
- пропуски заполняются самым частым значением через `SimpleImputer(strategy="most_frequent")`;
- кодируются через `OneHotEncoder(handle_unknown="ignore")`.

Такой подход выбран для устойчивого инференса на новых батчах: если в новых данных
появится ранее неизвестная категория, encoder не завершится ошибкой.

## Проектирование модели

В MVP pipeline рассматриваются несколько моделей-кандидатов:

- `LinearRegression`;
- `KNeighborsRegressor`;
- `DecisionTreeRegressor`;
- `RandomForestRegressor`;
- `XGBRegressor`, если установлен `xgboost`;
- `CatBoostRegressor`, если установлен `catboost`.

Список кандидатов можно задавать из CLI или YAML:

```bash
python run.py --mode update \
  --model-candidates LinearRegression,DecisionTree,RandomForest
```

В CI/CD используется укороченный набор моделей, чтобы обучение было быстрым и
стабильным на GitHub runner:

```yaml
MLOPS_MODEL_CANDIDATES: "LinearRegression,DecisionTree,RandomForest"
MLOPS_RANDOM_FOREST_ESTIMATORS: "30"
```

Рассматриваемые гиперпараметры:

- `DecisionTreeRegressor`: `max_depth=12`, `min_samples_leaf=20`, `random_state=42`;
- `RandomForestRegressor`: `n_estimators`, `max_depth=12`, `n_jobs=-1`, `random_state=42`;
- `KNeighborsRegressor`: `n_neighbors=7`, `weights="distance"`;
- `XGBRegressor`: `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `random_state=42`;
- `CatBoostRegressor`: `iterations=300`, `depth=8`, `learning_rate=0.1`, `random_seed=42`.

Лучшая модель выбирается по максимальному `R2` на holdout-выборке. Дополнительно
сохраняются `MAE`, `RMSE`, время обучения и параметры модели.

## Валидация

Валидация проводится на train/test split:

- `train_test_split(..., test_size=0.2, random_state=42)`;
- качество считается на тестовой части;
- основная метрика выбора лучшей модели - `R2`;
- дополнительные метрики - `MAE` и `RMSE`.

В CI/CD также выполняются тесты:

```bash
pytest -q
```

Тесты проверяют:

- успешное выполнение update;
- создание `models/registry.json`;
- создание сериализованной модели `model_v001.joblib`;
- создание summary-отчета;
- успешный inference и появление колонки `predict`.

## Сохранение артефактов

После обучения сохраняются:

- `artifacts/training.log` - лог обучения в GitHub Actions;
- `artifacts/state.json` - позиция чтения потоковых CSV;
- `artifacts/data_meta.jsonl` - история считанных батчей;
- `artifacts/data_quality.jsonl` - история проверок качества;
- `artifacts/data_collector_state.joblib` - сериализованное состояние сборщика данных;
- `models/model_vXXX.joblib` - сериализованная модель;
- `models/registry.json` - реестр версий моделей, метрик и гиперпараметров;
- `reports/summary_latest.json` - summary истории обучения.

В GitHub Actions эти файлы публикуются через `actions/upload-artifact`, поэтому их
можно скачать из run workflow.

## CI/CD подход

Выбран сценарий CRON-инкрементального обучения. Причины:

- исходная MVP-система уже работает как потоковая обработка батчей;
- состояние чтения данных хранится в `artifacts/state.json`;
- накопленные батчи хранятся в `raw_store/`;
- версии моделей хранятся в `models/registry.json`;
- GitHub Actions может восстанавливать состояние предыдущего запуска через cache.

Workflow запускается:

- при `push`;
- при `pull_request`;
- вручную через `workflow_dispatch`;
- по расписанию через `schedule`.

Для scheduled-запусков workflow восстанавливает директории `artifacts/`,
`raw_store/`, `models/` и `reports/`, выполняет новый update и сохраняет состояние
обратно. Это имитирует регулярное дообучение на новых батчах данных.
