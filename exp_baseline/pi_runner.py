import time
import argparse
import os
import csv

import numpy as np
import psutil
from sklearn.cluster import KMeans

import torch
import torchvision.models as models


def gap_stat_attack_detect(score_sum_last10: np.ndarray) -> int:
    """Gap-statistic style: decide if k=1 (no attack) or k>1 (attack)."""
    nrefs = 10

    score = score_sum_last10.astype(np.float64)

    # n_samples = number of workers / clients
    n = score.shape[0]
    if n < 2:
        return 0

    # Normalize
    mn, mxv = score.min(), score.max()
    if mxv - mn < 1e-12:
        return 0
    score = (score - mn) / (mxv - mn)

    # IMPORTANT: cannot have more clusters than samples
    k_max = min(7, n)          # was hardcoded to 7
    ks = range(1, k_max + 1)

    eps = 1e-12               # avoid log(0)
    gaps = np.zeros(len(ks))
    sdk = np.zeros(len(ks))
    gapDiff = np.zeros(max(0, len(ks) - 1))

    select_k = 1
    for i, k in enumerate(ks):
        est = KMeans(n_clusters=k, n_init="auto")
        est.fit(score.reshape(-1, 1))
        labels = est.labels_
        centers = est.cluster_centers_.reshape(-1)

        Wk = np.sum((score - centers[labels]) ** 2)
        Wk = max(Wk, eps)

        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            est2 = KMeans(n_clusters=k, n_init="auto")
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
    nworkers: int,
    nbyz: int,
    update_dim: int,
    noise: float,
    attack_scale: float,
    sample_interval_s: float,
    writer: csv.DictWriter,
    proc: psutil.Process,
) -> None:
    score_hist = []

    # Prime CPU% calculation (psutil needs a first call)
    proc.cpu_percent(interval=None)

    next_sample = time.perf_counter()

    for r in range(rounds):
        t0 = time.perf_counter()

        # Inference workload
        with torch.no_grad():
            for _ in range(infer_per_round):
                _ = model(x)

        detected = 0
        if do_detector:
            # Synthetic "client updates"
            updates = np.random.normal(0, noise, size=(nworkers, update_dim)).astype(np.float32)
            if nbyz > 0:
                updates[:nbyz] *= attack_scale

            # Score and detection
            score = np.linalg.norm(updates, axis=1)
            score_hist.append(score)
            if len(score_hist) >= 10:
                detected = gap_stat_attack_detect(np.sum(score_hist[-10:], axis=0))

        t1 = time.perf_counter()

        # Periodic resource sampling (CPU% and RSS)
        now = time.perf_counter()
        if now >= next_sample:
            cpu_pct = proc.cpu_percent(interval=None)  # since last call
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            writer.writerow({
                "timestamp_s": f"{now:.3f}",
                "phase": phase_name,
                "round": r,
                "cpu_percent": f"{cpu_pct:.2f}",
                "rss_mb": f"{rss_mb:.2f}",
                "round_time_s": f"{(t1 - t0):.4f}",
                "detected": detected,
            })
            next_sample = now + sample_interval_s

        # lightweight console progress
        if (r + 1) % max(1, rounds // 10) == 0:
            print(f"{phase_name}: round {r+1}/{rounds} | detected={detected} | round_time={t1-t0:.4f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="pi_benchmark.csv")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--infer_per_round", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--img", type=int, default=224)

    # Detector load knobs
    ap.add_argument("--nworkers", type=int, default=10)
    ap.add_argument("--nbyz", type=int, default=2)
    ap.add_argument("--update_dim", type=int, default=200000)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--attack_scale", type=float, default=20.0)

    # Sampling
    ap.add_argument("--sample_interval", type=float, default=0.5, help="seconds between CPU/RSS samples")

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
            fieldnames=["timestamp_s", "phase", "round", "cpu_percent", "rss_mb", "round_time_s", "detected"],
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
            nworkers=args.nworkers,
            nbyz=args.nbyz,
            update_dim=args.update_dim,
            noise=args.noise,
            attack_scale=args.attack_scale,
            sample_interval_s=args.sample_interval,
            writer=writer,
            proc=proc,
        )

        print("Phase 2/2: inference + FLDetector")
        run_phase(
            phase_name="with_detector",
            model=model,
            x=x,
            rounds=args.rounds,
            infer_per_round=args.infer_per_round,
            do_detector=True,
            nworkers=args.nworkers,
            nbyz=args.nbyz,
            update_dim=args.update_dim,
            noise=args.noise,
            attack_scale=args.attack_scale,
            sample_interval_s=args.sample_interval,
            writer=writer,
            proc=proc,
        )

    print(f"Done. Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
