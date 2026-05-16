import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prod_year = rng.integers(1998, 2019, size=rows)
    insured_value = rng.normal(450_000, 120_000, size=rows).clip(50_000, 1_500_000)
    seats = rng.choice([2, 4, 5, 7, 12], size=rows, p=[0.06, 0.45, 0.32, 0.12, 0.05])
    carrying_capacity = rng.choice([0, 2, 4, 6, 8, 12], size=rows)
    ccm_ton = rng.normal(1800, 650, size=rows).clip(600, 5000)
    insr_type = rng.choice([1201, 1202, 1204], size=rows)
    vehicle_type = rng.choice(["Automobile", "Pick-up", "Truck", "Station Wagones"], size=rows)
    make = rng.choice(["TOYOTA", "NISSAN", "ISUZU", "HYUNDAI", "UNKNOWN"], size=rows)
    usage = rng.choice(["Private", "Own Goods", "Taxi", "General Cartage"], size=rows)
    sex = rng.choice([0, 1], size=rows)

    vehicle_age = 2018 - prod_year
    premium = (
        insured_value * 0.012
        + vehicle_age * 42
        + seats * 90
        + carrying_capacity * 35
        + (insr_type == 1204) * 450
        + rng.normal(0, 180, size=rows)
    ).clip(250, None)

    return pd.DataFrame(
        {
            "SEX": sex,
            "INSR_BEGIN": "01-JAN-18",
            "INSR_END": "31-DEC-18",
            "EFFECTIVE_YR": 18,
            "INSR_TYPE": insr_type,
            "INSURED_VALUE": insured_value.round(2),
            "PREMIUM": premium.round(2),
            "OBJECT_ID": np.arange(seed * 1_000_000, seed * 1_000_000 + rows),
            "PROD_YEAR": prod_year,
            "SEATS_NUM": seats,
            "CARRYING_CAPACITY": carrying_capacity,
            "TYPE_VEHICLE": vehicle_type,
            "CCM_TON": ccm_ton.round(1),
            "MAKE": make,
            "USAGE": usage,
            "CLAIM_PAID": rng.choice([np.nan, 5000, 12000, 25000], size=rows, p=[0.82, 0.08, 0.06, 0.04]),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate small vehicle-insurance CSV files for CI.")
    parser.add_argument("--output-dir", default="data", help="Directory for generated CSV files")
    parser.add_argument("--rows-per-file", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_frame(args.rows_per_file, seed=11).to_csv(output_dir / "motor_data11-14lats.csv", index=False)
    build_frame(args.rows_per_file, seed=14).to_csv(output_dir / "motor_data14-2018.csv", index=False)
    print(f"Generated sample data in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
