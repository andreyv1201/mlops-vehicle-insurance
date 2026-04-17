import json
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass
class Config:
    data_files: tuple[str, ...] = ("data/motor_data11-14lats.csv", "data/motor_data14-2018.csv")
    batch_size: int = 20000
    batches_per_update: int = 1
    min_quality_non_null_ratio: float = 0.6
    artifacts_dir: str = "artifacts"
    raw_store_dir: str = "raw_store"
    models_dir: str = "models"
    reports_dir: str = "reports"
    target_col: str = "PREMIUM"


class MVPPipeline:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.artifacts_dir = Path(self.cfg.artifacts_dir)
        self.raw_store_dir = Path(self.cfg.raw_store_dir)
        self.models_dir = Path(self.cfg.models_dir)
        self.reports_dir = Path(self.cfg.reports_dir)
        self.state_path = self.artifacts_dir / "state.json"
        self.meta_path = self.artifacts_dir / "data_meta.jsonl"
        self.quality_log_path = self.artifacts_dir / "data_quality.jsonl"
        self.registry_path = self.models_dir / "registry.json"

        for d in (self.artifacts_dir, self.raw_store_dir, self.models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _utc_now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _read_state(self) -> dict:
        if not self.state_path.exists():
            return {"file_idx": 0, "chunk_idx": 0, "global_batch_idx": 0}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _next_batches(self) -> list[tuple[pd.DataFrame, dict]]:
        state = self._read_state()
        file_idx = state["file_idx"]
        chunk_idx = state["chunk_idx"]
        global_batch_idx = state["global_batch_idx"]
        collected: list[tuple[pd.DataFrame, dict]] = []

        while len(collected) < self.cfg.batches_per_update and file_idx < len(self.cfg.data_files):
            file_path = Path(self.cfg.data_files[file_idx])
            local_chunk = 0
            for chunk in pd.read_csv(file_path, chunksize=self.cfg.batch_size):
                if local_chunk < chunk_idx:
                    local_chunk += 1
                    continue
                meta = {
                    "timestamp_utc": self._utc_now(),
                    "file": str(file_path),
                    "file_idx": file_idx,
                    "chunk_idx": local_chunk,
                    "global_batch_idx": global_batch_idx,
                    "rows": int(len(chunk)),
                    "cols": int(chunk.shape[1]),
                    "missing_total": int(chunk.isna().sum().sum()),
                    "missing_ratio": float(chunk.isna().sum().sum() / max(chunk.shape[0] * chunk.shape[1], 1)),
                }
                collected.append((chunk, meta))
                global_batch_idx += 1
                local_chunk += 1
                if len(collected) >= self.cfg.batches_per_update:
                    state.update({"file_idx": file_idx, "chunk_idx": local_chunk, "global_batch_idx": global_batch_idx})
                    self._write_state(state)
                    return collected
            file_idx += 1
            chunk_idx = 0

        state.update({"file_idx": file_idx, "chunk_idx": chunk_idx, "global_batch_idx": global_batch_idx})
        self._write_state(state)
        return collected

    def collect_stream_batches(self) -> list[dict]:
        batches = self._next_batches()
        saved_meta: list[dict] = []
        for df_batch, meta in batches:
            batch_path = self.raw_store_dir / f"batch_{meta['global_batch_idx']:05d}.csv"
            df_batch.to_csv(batch_path, index=False)
            meta["raw_store_path"] = str(batch_path)
            self._append_jsonl(self.meta_path, meta)
            saved_meta.append(meta)
        return saved_meta

    def _load_raw_store(self) -> pd.DataFrame:
        files = sorted(self.raw_store_dir.glob("batch_*.csv"))
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_csv(p) for p in files), ignore_index=True)

    def assess_and_clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if df.empty:
            quality = {"timestamp_utc": self._utc_now(), "rows_before": 0, "rows_after": 0, "status": "empty"}
            self._append_jsonl(self.quality_log_path, quality)
            return df, quality

        non_null_ratio = (1 - df.isna().mean()).to_dict()
        keep_cols = [c for c, ratio in non_null_ratio.items() if ratio >= self.cfg.min_quality_non_null_ratio]
        if self.cfg.target_col not in keep_cols and self.cfg.target_col in df.columns:
            keep_cols.append(self.cfg.target_col)
        cleaned = df[keep_cols].copy()
        cleaned = cleaned.dropna(subset=[self.cfg.target_col])
        cleaned = cleaned[cleaned[self.cfg.target_col] > 0]

        quality = {
            "timestamp_utc": self._utc_now(),
            "rows_before": int(df.shape[0]),
            "rows_after": int(cleaned.shape[0]),
            "cols_before": int(df.shape[1]),
            "cols_after": int(cleaned.shape[1]),
            "dropped_columns": sorted(list(set(df.columns) - set(cleaned.columns))),
            "mean_missing_ratio_before": float(df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1)),
            "mean_missing_ratio_after": float(cleaned.isna().sum().sum() / max(cleaned.shape[0] * cleaned.shape[1], 1)),
            "status": "ok",
        }
        self._append_jsonl(self.quality_log_path, quality)
        return cleaned, quality

    def _build_models(self) -> dict[str, object]:
        models: dict[str, object] = {
            "LinearRegression": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
            "DecisionTree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
            ),
        }
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
        except Exception:
            pass

        try:
            from catboost import CatBoostRegressor

            models["CatBoost"] = CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=0,
            )
        except Exception:
            pass
        return models

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )

    def train_validate_and_register(self, df: pd.DataFrame) -> dict:
        if df.empty or self.cfg.target_col not in df.columns:
            raise ValueError("Недостаточно данных для обучения.")

        X = df.drop(columns=[self.cfg.target_col]).copy()
        y = df[self.cfg.target_col].astype(float)
        obj_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if obj_cols:
            X[obj_cols] = X[obj_cols].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preprocessor = self._build_preprocessor(X_train)
        models = self._build_models()
        results = []
        trained = {}

        for name, model in models.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            start = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_time_sec = time.perf_counter() - start

            pred = pipe.predict(X_test)
            metrics = {
                "MAE": float(mean_absolute_error(y_test, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "R2": float(r2_score(y_test, pred)),
            }
            results.append({"model": name, **metrics, "train_time_sec": train_time_sec})
            trained[name] = pipe

        best = max(results, key=lambda x: x["R2"])
        best_model = trained[best["model"]]

        registry = []
        if self.registry_path.exists():
            registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        version = len(registry) + 1
        model_path = self.models_dir / f"model_v{version:03d}.joblib"
        joblib.dump(best_model, model_path)

        entry = {
            "version": version,
            "timestamp_utc": self._utc_now(),
            "best_model_name": best["model"],
            "metrics": {"MAE": best["MAE"], "RMSE": best["RMSE"], "R2": best["R2"]},
            "hyperparams": best_model.named_steps["model"].get_params(),
            "train_time_sec": best["train_time_sec"],
            "model_path": str(model_path),
            "candidates": results,
        }
        registry.append(entry)
        self.registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        return entry

    def update(self) -> bool:
        batch_meta = self.collect_stream_batches()
        if not batch_meta:
            print("Новых батчей не осталось, update пропущен.")
            return False

        df = self._load_raw_store()
        cleaned, quality = self.assess_and_clean_data(df)
        if quality["rows_after"] < 100:
            print("После очистки мало данных для обучения.")
            return False

        model_entry = self.train_validate_and_register(cleaned)
        print(
            f"Update завершен: version={model_entry['version']}, "
            f"best={model_entry['best_model_name']}, R2={model_entry['metrics']['R2']:.4f}"
        )
        return True

    def _latest_model_path(self) -> Path:
        if not self.registry_path.exists():
            raise FileNotFoundError("Реестр моделей не найден. Сначала выполните update.")
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not registry:
            raise FileNotFoundError("Реестр пуст. Сначала выполните update.")
        return Path(registry[-1]["model_path"])

    def inference(self, file_path: str) -> str:
        model = joblib.load(self._latest_model_path())
        df = pd.read_csv(file_path).copy()
        obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if obj_cols:
            df[obj_cols] = df[obj_cols].astype(str)
        pred = model.predict(df)
        out = df.copy()
        out["predict"] = pred

        output_path = self.reports_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out.to_csv(output_path, index=False)
        return str(output_path)

    def summary(self) -> str:
        data_meta = []
        if self.meta_path.exists():
            data_meta = [json.loads(line) for line in self.meta_path.read_text(encoding="utf-8").splitlines() if line]

        quality_log = []
        if self.quality_log_path.exists():
            quality_log = [json.loads(line) for line in self.quality_log_path.read_text(encoding="utf-8").splitlines() if line]

        registry = []
        if self.registry_path.exists():
            registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

        summary_payload = {
            "generated_at_utc": self._utc_now(),
            "data_stream": {
                "batches_seen": len(data_meta),
                "rows_collected_total": int(sum(item.get("rows", 0) for item in data_meta)),
                "avg_batch_missing_ratio": float(np.mean([item.get("missing_ratio", 0.0) for item in data_meta])) if data_meta else None,
            },
            "data_quality_timeline": quality_log,
            "model_timeline": [
                {
                    "version": r["version"],
                    "timestamp_utc": r["timestamp_utc"],
                    "best_model_name": r["best_model_name"],
                    "metrics": r["metrics"],
                    "train_time_sec": r["train_time_sec"],
                    "hyperparams": r["hyperparams"],
                }
                for r in registry
            ],
        }
        output_path = self.reports_dir / "summary_latest.json"
        output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output_path)
