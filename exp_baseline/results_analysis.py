#!/usr/bin/env python3
import sys
import pandas as pd


def summarize(series: pd.Series):
    s = series.astype(float)
    return {
        "mean": s.mean(),
        "p50": s.quantile(0.50),
        "p95": s.quantile(0.95),
    }


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    for phase in df["phase"].unique():
        d = df[df["phase"] == phase]

        cpu = summarize(d["cpu_percent"])
        rss = summarize(d["rss_mb"])
        rt  = summarize(d["round_time_s"])

        print(f"\nPhase: {phase}")
        print(f"  CPU%      mean={cpu['mean']:.2f}, p50={cpu['p50']:.2f}, p95={cpu['p95']:.2f}")
        print(f"  RSS (MB)  mean={rss['mean']:.2f}, p50={rss['p50']:.2f}, p95={rss['p95']:.2f}")
        print(f"  Round(s)  mean={rt['mean']:.4f}, p50={rt['p50']:.4f}, p95={rt['p95']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summarize_benchmark_stats.py <csv_file>")
        sys.exit(1)

    main(sys.argv[1])
