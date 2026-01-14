#!/usr/bin/env python3
import sys
import pandas as pd


def summarize(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": s.mean(),
        "p50": s.quantile(0.50),
        "p95": s.quantile(0.95),
    }


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # ---- Timing columns that actually exist
    timing_cols = [
        "infer_s",
        "detector_total_s",
        "update_gen_s",
        "score_s",
        "decision_s",
        "log_sample_s",
        "round_total_s",
    ]

    for phase in df["phase"].unique():
        d = df[df["phase"] == phase]

        print(f"\n=== Phase: {phase} ===")

        # ---- Resource stats (sparse sampling)
        cpu = summarize(d["cpu_percent"])
        rss = summarize(d["rss_mb"])
        temp = summarize(d["cpu_temp_c"])
        freq = summarize(d["cpu_freq_mhz"])

        print(
            f"CPU%      mean={cpu['mean']:.2f}, "
            f"p50={cpu['p50']:.2f}, p95={cpu['p95']:.2f}"
        )
        print(
            f"RSS (MB)  mean={rss['mean']:.2f}, "
            f"p50={rss['p50']:.2f}, p95={rss['p95']:.2f}"
        )
        print(
            f"TEMP (°C) mean={temp['mean']:.2f}, "
            f"p50={temp['p50']:.2f}, p95={temp['p95']:.2f}"
        )
        print(
            f"FREQ (MHz) mean={freq['mean']:.0f}, "
            f"p50={freq['p50']:.0f}, p95={freq['p95']:.0f}"
        )

        # ---- Timing stats
        print("\nTiming breakdown:")
        for col in timing_cols:
            stats = summarize(d[col])
            print(
                f"  {col:<18} "
                f"mean={stats['mean']:.6f}s "
                f"p50={stats['p50']:.6f}s "
                f"p95={stats['p95']:.6f}s"
            )

        # ---- Detector overhead as fraction of round
        valid = d[["detector_total_s", "round_total_s"]].dropna()
        if len(valid) > 0:
            frac = valid["detector_total_s"] / valid["round_total_s"]
            print(
                "\nDetector overhead:"
                f"\n  detector/round  "
                f"mean={frac.mean()*100:.2f}% "
                f"p50={frac.quantile(0.50)*100:.2f}% "
                f"p95={frac.quantile(0.95)*100:.2f}%"
            )

        # ---- Context switches (per round deltas)
        if "vol_cs_delta" in d.columns:
            vol = summarize(d["vol_cs_delta"])
            invol = summarize(d["invol_cs_delta"])
            print(
                "\nContext switches (per round):"
                f"\n  voluntary     mean={vol['mean']:.2f}, p95={vol['p95']:.2f}"
                f"\n  involuntary   mean={invol['mean']:.2f}, p95={invol['p95']:.2f}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summarize_benchmark_stats.py <csv_file>")
        sys.exit(1)

    main(sys.argv[1])
