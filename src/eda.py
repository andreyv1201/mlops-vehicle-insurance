import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df: pd.DataFrame, output_dir: str = "reports") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    stats = {}
    stats["shape"] = df.shape
    stats["missing"] = df.isna().sum().to_dict()
    stats["dtypes"] = df.dtypes.astype(str).to_dict()

    print(f"[Этап 2] Размер данных: {df.shape}")
    print("  Пропуски (топ-5):")
    missing = df.isna().sum().sort_values(ascending=False).head(5)
    for col, cnt in missing.items():
        print(f"    {col}: {cnt} ({cnt/len(df)*100:.1f}%)")

    num_cols = ["INSURED_VALUE", "PREMIUM", "PROD_YEAR", "SEATS_NUM", "CARRYING_CAPACITY", "CCM_TON"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["PREMIUM"].dropna().hist(bins=100, ax=axes[0])
    axes[0].set_title("Распределение PREMIUM")
    axes[0].set_xlabel("PREMIUM")
    df["PREMIUM"].dropna().apply(lambda x: x if x > 0 else 1).apply(
        pd.np.log10 if hasattr(pd, "np") else __import__("numpy").log10
    ).hist(bins=100, ax=axes[1])
    axes[1].set_title("Распределение log10(PREMIUM)")
    axes[1].set_xlabel("log10(PREMIUM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "premium_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Корреляционная матрица числовых признаков")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flat, ["TYPE_VEHICLE", "USAGE", "MAKE", "SEX"]):
        top = df[col].value_counts().head(10)
        top.plot.barh(ax=ax)
        ax.set_title(f"Топ-10: {col}")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_features.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="PREMIUM", by="INSR_TYPE", ax=ax, showfliers=False)
    ax.set_title("PREMIUM по типам страхования")
    ax.set_xlabel("INSR_TYPE")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "premium_by_insr_type.png"), dpi=150)
    plt.close()

    desc = df[num_cols].describe()
    desc.to_csv(os.path.join(output_dir, "descriptive_stats.csv"))
    stats["describe"] = desc.to_dict()

    print(f"  Графики сохранены в {output_dir}/")
    return stats


if __name__ == "__main__":
    from data_extraction import load_data
    df = load_data()
    run_eda(df)
