"""Phase 10 gate: one style-conditioned model for several FiveK experts, and few-shot styles.

Experts A-D are "known" styles; expert E is held out.

1. Joint vs separate (A-D, ``--n`` pairs each): one StyleHead with a learned
   embedding per expert (plus the style encoder trained jointly) vs four
   independent Heads. Gate: joint >= separate (mean test PSNR).
2. Few-shot E, no retraining of the shared weights:
     (a) encoder:   code = StyleEncoder(features of n E *after* images); unpaired;
     (b) embedding: a new embedding fitted on n E pairs, everything else frozen;
     (c) separate:  a new Head trained from scratch on n E pairs;
   for n in {5, 20, 50}, plus (c) at the full n as the reference.
   Gate: best of (a)/(b) at n = 20 within 1 dB of (c) at the full n.

Usage:
    uv run python -m bench.phase10 --out bench/results/phase10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench.learning_curve import Pairs, hwc, small
from bench.metrics import psnr
from photostyle.condition import StyleEncoder, StyleHead
from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.render import GlobalRenderer
from photostyle.train import fit, paired_loss

KNOWN, HELD = ["a", "b", "c", "d"], "e"


def load(expert: str, fx) -> dict:
    d = Pairs(Path(f"data/fivek_landscape_{expert}"), expert, fx)
    for it in d.items:
        it["src_t"], it["tgt_t"] = small(it, "src"), small(it, "tgt")
        it["tfeat"] = fx.cached(it["tgt_t"], f"tgt|{expert}|{it['file']}")
    pool = d.split("train") + d.split("validation")
    random.Random(1234).shuffle(pool)
    test = random.Random(99).sample(d.split("test"), min(100, len(d.split("test"))))
    return {"val": pool[:20], "pool": pool[20:], "test": test}


@torch.no_grad()
def test_psnr(theta_fn, r, items) -> float:
    vals = []
    for it in items:
        th = theta_fn(it["feat"][None])
        out = r(it["src"][None], th)[0]
        vals.append(psnr(np.clip(hwc(out), 0, 1), hwc(it["tgt"])))
    return float(np.mean(vals))


def train_joint(data, r, n, steps=3000, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    head = StyleHead(FEATURE_DIM, r.num_params, n_styles=len(KNOWN))
    enc = StyleEncoder(FEATURE_DIM)
    tr = {x: data[x]["pool"][:n] for x in KNOWN}
    allf = torch.stack([it["feat"] for x in KNOWN for it in tr[x]])
    head.set_norm(allf)
    enc.set_norm(torch.stack([it["tfeat"] for x in KNOWN for it in tr[x]]))
    opt = torch.optim.AdamW(list(head.parameters()) + list(enc.parameters()), lr=3e-3, weight_decay=1e-2)

    def batch_loss(items, code):
        th = head(torch.stack([it["feat"] for it in items]), code.expand(len(items), -1))
        return sum((r(it["src_t"][None], t[None])[0] - it["tgt_t"]).abs().mean() for it, t in zip(items, th)) / len(items)

    def val_loss():
        with torch.no_grad():
            return float(sum(batch_loss(data[x]["val"], head.embed.weight[i]) for i, x in enumerate(KNOWN)))

    best, state, bad = float("inf"), None, 0
    for step in range(1, steps + 1):
        head.train()
        enc.train()
        i = random.randrange(len(KNOWN))
        x = KNOWN[i]
        items = random.sample(tr[x], 8)
        ex = random.sample(tr[x], random.choice([5, 10, 20]))
        code_enc = enc(torch.stack([e["tfeat"] for e in ex]))
        loss = batch_loss(items, head.embed.weight[i]) + batch_loss(items, code_enc)
        # contrastive: the encoder code should point at the right embedding
        logits = F.normalize(code_enc, dim=0) @ F.normalize(head.embed.weight, dim=1).T / 0.1
        loss = loss + 0.05 * F.cross_entropy(logits[None], torch.tensor([i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            head.eval()
            enc.eval()
            v = val_loss()
            if v < best - 1e-5:
                best, bad = v, 0
                state = ({k: t.clone() for k, t in head.state_dict().items()},
                         {k: t.clone() for k, t in enc.state_dict().items()})
            else:
                bad += 1
                if bad >= 12:
                    break
    head.load_state_dict(state[0])
    enc.load_state_dict(state[1])
    head.eval()
    enc.eval()
    return head, enc, {"steps": step, "val": best}


def fit_embedding(head, r, items, seed=0, steps=400):
    """Condition-only fine-tuning: a new code vector, all weights frozen."""
    torch.manual_seed(seed)
    code = torch.nn.Parameter(head.embed.weight.mean(0).clone())
    opt = torch.optim.Adam([code], lr=1e-2)
    for _ in range(steps):
        b = random.sample(items, min(8, len(items)))
        th = head(torch.stack([it["feat"] for it in b]), code.expand(len(b), -1))
        loss = sum((r(it["src_t"][None], t[None])[0] - it["tgt_t"]).abs().mean() for it, t in zip(b, th)) / len(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return code.detach()


def train_separate(r, items, val, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    h = Head(FEATURE_DIM, r.num_params)
    h.set_norm(torch.stack([it["feat"] for it in items + val]))
    fit(h, r, items, val, paired_loss)
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bench/results/phase10"))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fx = FeatureExtractor(cache_dir=Path("data/cache/features"))
    data = {x: load(x, fx) for x in KNOWN + [HELD]}
    n = min(args.n, *(len(data[x]["pool"]) for x in KNOWN + [HELD]))
    print({x: {k: len(v) for k, v in d.items()} for x, d in data.items()}, f"n={n} [{time.time() - t0:.0f}s]",
          flush=True)
    r = GlobalRenderer("per_channel")
    res = {"n": n, "joint": {}, "separate": {}, "identity": {}, "fewshot": {}}

    head, enc, info = train_joint(data, r, n)
    res["joint_train"] = info
    for i, x in enumerate(KNOWN):
        res["joint"][x] = test_psnr(lambda f, i=i: head(f, head.embed.weight[i][None]), r, data[x]["test"])
        res["identity"][x] = test_psnr(lambda f: torch.zeros(1, r.num_params), r, data[x]["test"])
        h = train_separate(r, data[x]["pool"][:n], data[x]["val"])
        res["separate"][x] = test_psnr(h, r, data[x]["test"])
        print(f"{x}: identity {res['identity'][x]:.2f}  separate {res['separate'][x]:.2f}  "
              f"joint {res['joint'][x]:.2f} [{time.time() - t0:.0f}s]", flush=True)

    E = data[HELD]
    res["identity"][HELD] = test_psnr(lambda f: torch.zeros(1, r.num_params), r, E["test"])
    full = train_separate(r, E["pool"][:n], E["val"])
    res["fewshot"]["separate_full"] = test_psnr(full, r, E["test"])
    for k in (5, 20, 50):
        row = {}
        for seed in (0, 1, 2):
            sub = random.Random(seed).sample(E["pool"][:n], k)
            with torch.no_grad():
                code = enc(torch.stack([it["tfeat"] for it in sub]))
            row.setdefault("encoder", []).append(test_psnr(lambda f, c=code: head(f, c[None]), r, E["test"]))
            c2 = fit_embedding(head, r, sub, seed)
            row.setdefault("embedding", []).append(test_psnr(lambda f, c=c2: head(f, c[None]), r, E["test"]))
            if k >= 20:
                row.setdefault("separate", []).append(test_psnr(train_separate(r, sub, E["val"], seed), r, E["test"]))
        res["fewshot"][k] = {m: float(np.mean(v)) for m, v in row.items()}
        print(f"E n={k}: {res['fewshot'][k]}  (separate@{n}: {res['fewshot']['separate_full']:.2f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
        (args.out / "results.json").write_text(json.dumps(res, indent=1))

    joint_mean = float(np.mean(list(res["joint"].values())))
    sep_mean = float(np.mean(list(res["separate"].values())))
    best20 = max(res["fewshot"][20]["encoder"], res["fewshot"][20]["embedding"])
    res["gate"] = {"joint_ge_separate": joint_mean >= sep_mean - 0.05, "joint_mean": joint_mean, "separate_mean": sep_mean,
                   "fewshot20_within_1db": res["fewshot"]["separate_full"] - best20 <= 1.0,
                   "fewshot20_best": best20}
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    torch.save({"head": head.state_dict(), "encoder": enc.state_dict(), "styles": KNOWN},
               Path("checkpoints") / "phase10_stylehead.pt")
    print("gate:", res["gate"], f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
