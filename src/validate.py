import time
import numpy as np
from sklearn.model_selection import cross_val_score


def validate_model(model, X_train, y_train, X_test):
    print("[Этап 6] Кроссвалидация (5-fold)...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"  R2 по фолдам: {scores.round(4)}")
    print(f"  R2 среднее:   {scores.mean():.4f} +/- {scores.std():.4f}")

    print("\n  Тестирование (1000 предсказаний)")
    sample = X_test[:1]
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(
        f"  Latency: median={np.median(latencies):.2f} ms, "
        f"p95={np.percentile(latencies, 95):.2f} ms, "
        f"p99={np.percentile(latencies, 99):.2f} ms"
    )

    t0 = time.perf_counter()
    model.predict(X_test)
    batch_time = (time.perf_counter() - t0) * 1000
    print(f"  Batch ({len(X_test)} записей): {batch_time:.0f} ms")

    return {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "latency_median_ms": np.median(latencies),
        "latency_p95_ms": np.percentile(latencies, 95),
    }
