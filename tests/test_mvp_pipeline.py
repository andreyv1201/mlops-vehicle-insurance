import json
from pathlib import Path

import pandas as pd

from scripts.generate_sample_data import build_frame
from src.mvp_pipeline import Config, MVPPipeline


def test_update_registers_model_and_summary(tmp_path):
    data_path = tmp_path / "stream.csv"
    build_frame(rows=160, seed=1).to_csv(data_path, index=False)

    config = Config(
        data_files=(str(data_path),),
        batch_size=120,
        batches_per_update=1,
        model_candidates=("LinearRegression", "DecisionTree"),
        artifacts_dir=str(tmp_path / "artifacts"),
        raw_store_dir=str(tmp_path / "raw_store"),
        models_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports"),
    )
    pipeline = MVPPipeline(config)

    assert pipeline.update() is True

    registry_path = tmp_path / "models" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(registry) == 1
    assert registry[0]["best_model_name"] in {"LinearRegression", "DecisionTree"}
    assert (tmp_path / "models" / "model_v001.joblib").exists()

    summary_path = pipeline.summary()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert summary["data_stream"]["batches_seen"] == 1
    assert summary["model_timeline"][0]["version"] == 1


def test_inference_writes_predictions(tmp_path):
    data_path = tmp_path / "stream.csv"
    inference_path = tmp_path / "inference.csv"
    frame = build_frame(rows=180, seed=2)
    frame.to_csv(data_path, index=False)
    frame.drop(columns=["PREMIUM"]).head(10).to_csv(inference_path, index=False)

    config = Config(
        data_files=(str(data_path),),
        batch_size=140,
        model_candidates=("LinearRegression",),
        artifacts_dir=str(tmp_path / "artifacts"),
        raw_store_dir=str(tmp_path / "raw_store"),
        models_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports"),
    )
    pipeline = MVPPipeline(config)

    assert pipeline.update() is True
    output_path = pipeline.inference(str(inference_path))
    output = pd.read_csv(output_path)
    assert "predict" in output.columns
    assert len(output) == 10
