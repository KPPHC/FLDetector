python3 - << 'PY'
import pandas as pd
df = pd.read_csv("pi4_benchmark.csv")
for phase in df["phase"].unique():
    d = df[df["phase"]==phase]
    print(phase,
          "cpu% mean", d["cpu_percent"].mean(),
          "rss MB mean", d["rss_mb"].mean(),
          "round time mean", d["round_time_s"].mean())
PY
