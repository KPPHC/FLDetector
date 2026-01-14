import time
import argparse
import os
import csv

import numpy as np
import psutil
from sklearn.cluster import KMeans

import torch
import torchvision.models as models

from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class RoundTimes:
    phase: str
    round: int
    infer_s: float = 0.0
    detector_s: float = 0.0
    update_gen_s: float = 0.0
    score_s: float = 0.0
    decision_s: float = 0.0
    log_s: float = 0.0
    total_s: float = 0.0
    detected: int = 0

class Timer:
    __slots__ = ("t0", "dt")
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0


def read_cpu_temp_c() -> Optional[float]:
    # Works on most Raspberry Pi OS images
    p = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(p, "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None

def read_cpu_freq_mhz() -> Optional[float]:
    # Usually present; returns kHz
    p = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    try:
        with open(p, "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def per_round_z_detect(
    curr: float,
    hist: list[float],
    warmup: int,
    z_thresh: float,
) -> int:
    """
    Decide attack from current score only, using past scores to form a baseline.
    - During warmup, always return 0.
    - After warmup, flag if curr is z_thresh std dev above mean of history.
    """
    if len(hist) < warmup:
        return 0
    mu = float(np.mean(hist))
    sd = float(np.std(hist)) + 1e-12
    z = (curr - mu) / sd
    return 1 if z >= z_thresh else 0


def gap_stat_attack_detect(samples: np.ndarray) -> int:
    """Gap-statistic: decide if k=1 (no change) or k>1 (distribution looks multi-modal)."""
    nrefs = 10
    x = samples.astype(np.float64)
    n = x.shape[0]
    if n < 2:
        return 0

    mn, mxv = x.min(), x.max()
    if mxv - mn < 1e-12:
        return 0
    x = (x - mn) / (mxv - mn)

    k_max = min(7, n)
    ks = range(1, k_max + 1)

    eps = 1e-12
    gaps = np.zeros(len(ks))
    sdk = np.zeros(len(ks))
    gapDiff = np.zeros(max(0, len(ks) - 1))

    select_k = 1
    for i, k in enumerate(ks):
        est = KMeans(n_clusters=k, n_init=10, random_state=0)
        est.fit(x.reshape(-1, 1))
        labels = est.labels_
        centers = est.cluster_centers_.reshape(-1)

        Wk = np.sum((x - centers[labels]) ** 2)
        Wk = max(Wk, eps)

        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, n)
            est2 = KMeans(n_clusters=k, n_init=10, random_state=0)
            est2.fit(rand.reshape(-1, 1))
            labels2 = est2.labels_
            centers2 = est2.cluster_centers_.reshape(-1)
            wkref = np.sum((rand - centers2[labels2]) ** 2)
            WkRef[j] = max(wkref, eps)

        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]

    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break

    return 0 if select_k == 1 else 1




def run_phase(
    phase_name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    rounds: int,
    infer_per_round: int,
    do_detector: bool,
    update_dim: int,
    noise: float,
    attack_scale: float,
    sample_interval_s: float,
    writer: csv.DictWriter,
    proc: psutil.Process,
    detect_mode: str,
    window_len: int,
    warmup: int,
    z_thresh: float,
    ) -> None:

    score_hist_1d: list[float] = []

    # Prime CPU% calculation (psutil needs a first call)
    proc.cpu_percent(interval=None)
    phase_t0 = time.perf_counter()
    next_sample = phase_t0

    last_cpu_pct = ""
    last_rss_mb = ""
    last_temp_c = ""
    last_freq_mhz = ""
    last_load1 = ""
    last_load5 = ""
    last_load15 = ""
    last_disk_read_delta = "0"
    last_disk_write_delta = "0"
    last_net_sent_delta = "0"
    last_net_recv_delta = "0"

    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    prev_ctx = proc.num_ctx_switches()

    for r in range(rounds):
        # Process-level counters (cheap)
        ctx = proc.num_ctx_switches()  # returns (voluntary, involuntary)
        vol_cs_delta = ctx.voluntary - prev_ctx.voluntary
        invol_cs_delta = ctx.involuntary - prev_ctx.involuntary
        prev_ctx = ctx
        round_start = time.perf_counter()

        # ---- Inference timing and workload
        with Timer() as t_infer:
            with torch.no_grad():
                for _ in range(infer_per_round):
                    _ = model(x)
        infer_s = t_infer.dt

        detected = 0

        # Detector breakdown (default zeros)
        detector_total_s = 0.0
        update_gen_s = 0.0
        score_s = 0.0
        decision_s = 0.0

        if do_detector:
            det_start = time.perf_counter()

            # per-client synthetic "update"
            with Timer() as t_upd:
                update = np.random.normal(0, noise, size=(update_dim,)).astype(np.float32)
                update *= attack_scale
            update_gen_s = t_upd.dt
            #update = np.random.normal(0, noise, size=(update_dim,)).astype(np.float32)

            # Optional: simulate this client being Byzantine
            # (if you want to keep "nbyz" semantics, handle it outside by launching some Pis with --attack_scale)
            #update *= attack_scale  # only if you run an "attacker" instance

            with Timer() as t_score: 
                curr_score = float(np.linalg.norm(update))
                score_hist_1d.append(curr_score)
            score_s = t_score.dt

            with Timer() as t_dec:

                if detect_mode == "gap_window":
                    W = window_len
                    if len(score_hist_1d) >= W:
                        window = np.array(score_hist_1d[-W:], dtype=np.float64)
                        detected = gap_stat_attack_detect(window)
                else:  # "per_round"
                    detected = per_round_z_detect(
                        curr=curr_score,
                        hist=score_hist_1d[:-1],   # baseline excludes current point
                        warmup=warmup,
                        z_thresh=z_thresh,
                    )
            decision_s = t_dec.dt
            detector_total_s = time.perf_counter() - det_start

        # ---- Optional resource sampling (sparse) + time spent logging
        log_s = 0.0
        now = time.perf_counter()
        if now >= next_sample:
            with Timer() as t_log:
                last_cpu_pct = f"{proc.cpu_percent(interval=None):.2f}"
                last_rss_mb = f"{(proc.memory_info().rss / (1024 * 1024)):.2f}"

                temp_c = read_cpu_temp_c()
                freq_mhz = read_cpu_freq_mhz()
                load1, load5, load15 = os.getloadavg()

                last_temp_c = "" if temp_c is None else f"{temp_c:.2f}"
                last_freq_mhz = "" if freq_mhz is None else f"{freq_mhz:.0f}"
                last_load1 = f"{load1:.2f}"
                last_load5 = f"{load5:.2f}"
                last_load15 = f"{load15:.2f}"

                disk = psutil.disk_io_counters()
                net = psutil.net_io_counters()

                disk_read_delta = disk.read_bytes - prev_disk.read_bytes
                disk_write_delta = disk.write_bytes - prev_disk.write_bytes
                net_sent_delta = net.bytes_sent - prev_net.bytes_sent
                net_recv_delta = net.bytes_recv - prev_net.bytes_recv

                prev_disk = disk
                prev_net = net

                last_disk_read_delta = str(disk_read_delta)
                last_disk_write_delta = str(disk_write_delta)
                last_net_sent_delta = str(net_sent_delta)
                last_net_recv_delta = str(net_recv_delta)

            log_s += t_log.dt
            next_sample = now + sample_interval_s

        round_total_s = time.perf_counter() - round_start

        # ---- Single CSV row every round
        with Timer() as t_write:
            writer.writerow({
                "timestamp_s": f"{time.perf_counter() - phase_t0:.6f}",
                "phase": phase_name,
                "round": r,

                "infer_s": f"{infer_s:.6f}",

                "detector_total_s": f"{detector_total_s:.6f}",
                "update_gen_s": f"{update_gen_s:.6f}",
                "score_s": f"{score_s:.6f}",
                "decision_s": f"{decision_s:.6f}",

                "log_sample_s": f"{log_s:.6f}",
                "round_total_s": f"{round_total_s:.6f}",

                "cpu_percent": last_cpu_pct,
                "rss_mb": last_rss_mb,

                "detected": detected,

                "cpu_temp_c": last_temp_c,
                "cpu_freq_mhz": last_freq_mhz,
                "load1": last_load1,
                "load5": last_load5,
                "load15": last_load15,

                "vol_cs_delta": str(vol_cs_delta),
                "invol_cs_delta": str(invol_cs_delta),

                "disk_read_bytes": last_disk_read_delta,
                "disk_write_bytes": last_disk_write_delta,
                "net_sent_bytes": last_net_sent_delta,
                "net_recv_bytes": last_net_recv_delta,
            })

        csv_write_s = t_write.dt

        if (r + 1) % max(1, rounds // 10) == 0:
            print(
                f"{phase_name}: round {r+1}/{rounds} | "
                f"total={round_total_s:.4f}s infer={infer_s:.4f}s det={detector_total_s:.4f}s "
                f"(upd={update_gen_s:.4f}s score={score_s:.4f}s dec={decision_s:.4f}s) "
                f"detected={detected}"
            )




        # # Periodic resource sampling (CPU% and RSS)
        # now = time.perf_counter()
        # if now >= next_sample:
        #     cpu_pct = proc.cpu_percent(interval=None)  # since last call
        #     rss_mb = proc.memory_info().rss / (1024 * 1024)
        #     writer.writerow({
        #         "timestamp_s": f"{now:.3f}",
        #         "phase": phase_name,
        #         "round": r,
        #         "cpu_percent": f"{cpu_pct:.2f}",
        #         "rss_mb": f"{rss_mb:.2f}",
        #         "round_time_s": f"{(t1 - round_start):.4f}",
        #         "detected": detected,
        #     })
        #     next_sample = now + sample_interval_s

        # # lightweight console progress
        # if (r + 1) % max(1, rounds // 10) == 0:
        #     print(f"{phase_name}: round {r+1}/{rounds} | detected={detected} | round_time={t1-round_start:.4f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="pi_benchmark.csv")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--infer_per_round", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--img", type=int, default=224)

    # Detector load knobs
    ap.add_argument("--update_dim", type=int, default=200000)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--attack_scale", type=float, default=1.0)

    # Sampling
    ap.add_argument("--sample_interval", type=float, default=0.5, help="seconds between CPU/RSS samples")

    # Detector mode and params
    ap.add_argument("--detect_mode", choices=["gap_window", "per_round"], default="gap_window")
    ap.add_argument("--window_len", type=int, default=10, help="window length for gap_window mode")
    ap.add_argument("--warmup", type=int, default=10, help="warmup rounds for per_round mode")
    ap.add_argument("--z_thresh", type=float, default=3.0, help="z threshold for per_round mode")

    # Optional: load a saved init checkpoint
    ap.add_argument("--ckpt", default="", help="optional resnet18_init.pth")

    args = ap.parse_args()

    # Model (CPU)
    model = models.resnet18(weights=None)
    model.eval()

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)

    x = torch.randn(args.batch, 3, args.img, args.img)

    proc = psutil.Process(os.getpid())

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            #fieldnames=["timestamp_s", "phase", "round", "cpu_percent", "rss_mb", "round_time_s", "detected"],
            fieldnames=[
                "timestamp_s", "phase", "round",
                "infer_s",
                "detector_total_s", "update_gen_s", "score_s", "decision_s",
                "log_sample_s", "round_total_s",
                "cpu_percent", "rss_mb",
                "detected",
                "cpu_temp_c", "cpu_freq_mhz",
                "load1", "load5", "load15",
                "vol_cs_delta", "invol_cs_delta",
                "disk_read_bytes", "disk_write_bytes",
                "net_sent_bytes", "net_recv_bytes",
            ]
        )
        writer.writeheader()

        print("Phase 1/2: baseline (inference only)")
        run_phase(
            phase_name="baseline",
            model=model,
            x=x,
            rounds=args.rounds,
            infer_per_round=args.infer_per_round,
            do_detector=False,
            update_dim=args.update_dim,
            noise=args.noise,
            attack_scale=args.attack_scale,
            sample_interval_s=args.sample_interval,
            writer=writer,
            proc=proc,
            detect_mode=args.detect_mode,
            window_len=args.window_len,
            warmup=args.warmup,
            z_thresh=args.z_thresh,
        )

        print("Phase 2/2: inference + FLDetector")
        run_phase(
            phase_name="with_detector",
            model=model,
            x=x,
            rounds=args.rounds,
            infer_per_round=args.infer_per_round,
            do_detector=True,
            update_dim=args.update_dim,
            noise=args.noise,
            attack_scale=args.attack_scale,
            sample_interval_s=args.sample_interval,
            writer=writer,
            proc=proc,
            detect_mode=args.detect_mode,
            window_len=args.window_len,
            warmup=args.warmup,
            z_thresh=args.z_thresh,
        )

    print(f"Done. Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
