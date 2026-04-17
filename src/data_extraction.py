import os
import pandas as pd


def load_data(data_dir: str = "data") -> pd.DataFrame:
    f1 = os.path.join(data_dir, "motor_data11-14lats.csv")
    f2 = os.path.join(data_dir, "motor_data14-2018.csv")

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df = pd.concat([df1, df2], ignore_index=True)

    print(f"[Этап 1] Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"  Файл 1: {len(df1)} строк")
    print(f"  Файл 2: {len(df2)} строк")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.info())
