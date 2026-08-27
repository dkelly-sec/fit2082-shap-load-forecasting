"""
Exploratory data analysis on the merged FIT2082 dataset.

Run from the project root (alongside src/ and data/):

    python eda.py

Produces:
  - Console summary: row counts, missing values, basic stats
  - eda_plots/demand_over_time.png       -- full 2-year demand series
  - eda_plots/demand_by_hour.png         -- average demand by hour, weekday vs weekend
  - eda_plots/demand_vs_temperature.png  -- scatter, does demand relate to temp as expected?
  - eda_plots/demand_vs_irradiance.png   -- scatter, same for irradiance
  - eda_plots/demand_by_season.png       -- boxplot by season
  - Console: candidate anomaly windows (flatlines, extreme outliers) to help
    decide EXCLUDED_PERIODS -- most likely this will come back empty, which
    is a legitimate result, not a failure to find something.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/interim/merged_full.csv")
OUT_DIR = Path("eda_plots")
OUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run fetch_aemo.py, load_bom.py, "
            "fetch_irradiance.py, and clean_merge.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    print("Missing values per column:")
    print(df.isna().sum())
    print()

    print("Demand summary statistics:")
    print(df["demand"].describe())
    print()

    # --- Plot 1: full demand time series ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["demand"], linewidth=0.3)
    ax.set_title("VIC1 Electricity Demand, 2024-2025 (5-min resolution)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demand_over_time.png", dpi=120)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'demand_over_time.png'}")

    # --- Plot 2: average demand by hour, weekday vs weekend ---
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    hourly_weekday = df[~df["is_weekend"]].groupby(df["timestamp"].dt.floor("h").dt.hour)["demand"].mean()
    hourly_weekend = df[df["is_weekend"]].groupby(df["timestamp"].dt.floor("h").dt.hour)["demand"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_weekday.index, hourly_weekday.values, label="Weekday", marker="o")
    ax.plot(hourly_weekend.index, hourly_weekend.values, label="Weekend", marker="o")
    ax.set_title("Average Demand by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demand_by_hour.png", dpi=120)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'demand_by_hour.png'}")

    # --- Plot 3: demand vs temperature ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["temperature"], df["demand"], s=1, alpha=0.1)
    ax.set_title("Demand vs Daily Temperature")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Demand (MW)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demand_vs_temperature.png", dpi=120)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'demand_vs_temperature.png'}")

    # --- Plot 4: demand vs irradiance ---
    if "irr_electricity" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["irr_electricity"], df["demand"], s=1, alpha=0.1)
        ax.set_title("Demand vs Solar PV Output (renewables.ninja)")
        ax.set_xlabel("Simulated PV capacity factor")
        ax.set_ylabel("Demand (MW)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "demand_vs_irradiance.png", dpi=120)
        plt.close(fig)
        print(f"Saved {OUT_DIR / 'demand_vs_irradiance.png'}")

    # --- Plot 5: demand by season ---
    fig, ax = plt.subplots(figsize=(8, 5))
    season_order = ["summer", "autumn", "winter", "spring"]
    data_by_season = [df[df["season"] == s]["demand"].dropna().values for s in season_order]
    try:
        ax.boxplot(data_by_season, tick_labels=season_order, showfliers=False)
    except TypeError:
        # older matplotlib versions use 'labels' instead of 'tick_labels'
        ax.boxplot(data_by_season, labels=season_order, showfliers=False)
    ax.set_title("Demand Distribution by Season")
    ax.set_ylabel("Demand (MW)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demand_by_season.png", dpi=120)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'demand_by_season.png'}")

    # --- Anomaly checks: flatlines and extreme outliers ---
    print()
    print("=" * 60)
    print("ANOMALY CHECKS (for deciding EXCLUDED_PERIODS)")
    print("=" * 60)

    # Flatline check: identical demand value repeated for a long stretch
    # (a real sensor/market glitch, not just a quiet period, tends to look
    # like this -- genuine demand always has some minute-to-minute noise)
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    same_as_prev = df_sorted["demand"] == df_sorted["demand"].shift(1)
    # count consecutive identical values
    run_id = (~same_as_prev).cumsum()
    run_lengths = df_sorted.groupby(run_id).size()
    long_flatlines = run_lengths[run_lengths >= 12]  # >= 1 hour of identical values

    if len(long_flatlines) == 0:
        print("No flatlines >= 1 hour found. Good sign -- likely no sensor/market glitches.")
    else:
        print(f"Found {len(long_flatlines)} flatline run(s) >= 1 hour:")
        for run_num, length in long_flatlines.items():
            rows = df_sorted[run_id == run_num]
            print(f"  {rows['timestamp'].min()} to {rows['timestamp'].max()} "
                  f"({length} rows, value={rows['demand'].iloc[0]:.1f})")

    print()

    # Extreme outlier check: demand values implausibly far from the mean
    mean, std = df["demand"].mean(), df["demand"].std()
    outliers = df[(df["demand"] < mean - 6 * std) | (df["demand"] > mean + 6 * std)]
    if len(outliers) == 0:
        print("No extreme outliers (>6 std from mean) found.")
    else:
        print(f"Found {len(outliers)} extreme outlier rows (>6 std from mean):")
        print(outliers[["timestamp", "demand"]].to_string(index=False))

    print()
    print("If both checks came back clean, EXCLUDED_PERIODS can reasonably "
          "stay empty -- that's a legitimate finding, not a gap in your work.")


if __name__ == "__main__":
    main()
