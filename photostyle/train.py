"""Style learning: paired (before/after) and unpaired (example photos).

Shared by the benchmarks (bench/) and the user-facing engine/CLI/Studio.

Paired   L1 between rendered input and target, AdamW, early stopping on a
         held-out split (the Phase 7 protocol).
Unpaired distort-and-recover pseudo-pairs built from the examples, plus a
         sliced-Wasserstein colour-distribution loss to the examples, plus an
         optional look profile and fidelity terms (the Pilot A recipe).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch
import torch.nn.functional as F

from photostyle.features import FEATURE_DIM, FeatureExtractor
from photostyle.head import Head
from photostyle.looks import LookProfile, fidelity_loss, lab_pixels, profile_loss, sliced_wasserstein
from photostyle.render import GlobalRenderer

TRAIN_EDGE = 256


def shrink(t: torch.Tensor, edge: int) -> torch.Tensor:
    _, h, w = t.shape
    s = edge / max(h, w)
    if s >= 1:
        return t
    return F.interpolate(t[None], size=(round(h * s), round(w * s)), mode="bilinear",
                         antialias=True, align_corners=False)[0]


def align_pair(src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Make an (original, edited) pair pixel-aligned, or return None.

    Tries the original's 0/90/180/270 degree rotations (edited exports are often
    rotated while originals are not), keeps the one whose shape is within 2 px
    and whose structure matches best, then resamples the target to the
    original's exact size. Crops cannot be aligned and return None.
    """
    best, best_err = None, float("inf")
    for k in range(4):
        cand = torch.rot90(src, k, dims=(1, 2))
        if abs(cand.shape[1] - tgt.shape[1]) > 2 or abs(cand.shape[2] - tgt.shape[2]) > 2:
            continue
        t = F.interpolate(tgt[None], size=cand.shape[1:], mode="bilinear", align_corners=False)[0]
        a = F.adaptive_avg_pool2d(cand.mean(0, keepdim=True)[None], 16)
        b = F.adaptive_avg_pool2d(t.mean(0, keepdim=True)[None], 16)
        a, b = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
        err = float((a - b).abs().mean())
        if err < best_err:
            best, best_err = (cand.contiguous(), t), err
    if best is None or best_err > 0.5:
        return None
    return best


# --------------------------------------------------------------------------- paired


def paired_loss(model, renderer, items: list[dict]) -> torch.Tensor:
    theta = model(torch.stack([it["feat"] for it in items]))
    loss = 0.0
    for it, th in zip(items, theta):
        loss = loss + (renderer(it["src_t"][None], th[None])[0] - it["tgt_t"]).abs().mean()
    reg = renderer.regularizer() if hasattr(renderer, "regularizer") else 0.0
    return loss / len(items) + 1e-4 * theta.pow(2).mean() + reg


def fit(model, renderer, train_items, val_items, loss_fn: Callable = paired_loss, steps=3000, batch=8,
        lr=3e-3, weight_decay=1e-2, patience=20, eval_every=25, progress: Callable | None = None) -> dict:
    """AdamW with early stopping on ``loss_fn(val_items)``. Items need 'feat' (+ fields loss_fn uses)."""
    params = list(model.parameters()) + list(renderer.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best, best_state, bad, step = float("inf"), None, 0, 0
    for step in range(1, steps + 1):
        model.train()
        loss = loss_fn(model, renderer, random.sample(train_items, min(batch, len(train_items))))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model, renderer, val_items))
            if progress:
                progress(step, steps, v)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = ({k: t.clone() for k, t in model.state_dict().items()},
                              {k: t.clone() for k, t in renderer.state_dict().items()})
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        renderer.load_state_dict(best_state[1])
    model.eval()
    return {"val_loss": best, "steps": step}


def learn_paired(pairs: list[tuple[torch.Tensor, torch.Tensor]], fx: FeatureExtractor, kind: str = "per_channel",
                 val_frac: float = 0.15, seed: int = 0, progress: Callable | None = None):
    """Learn a style from (before, after) float tensors. Returns (head, renderer, feats, info)."""
    random.seed(seed)
    torch.manual_seed(seed)
    items = []
    for src, tgt in pairs:
        al = align_pair(src, tgt)
        if al is None:
            continue
        s, t = al
        items.append({"feat": fx(s), "src_t": shrink(s, TRAIN_EDGE), "tgt_t": shrink(t, TRAIN_EDGE)})
    if len(items) < 3:
        raise ValueError(f"need at least 3 alignable pairs, got {len(items)}")
    random.shuffle(items)
    nv = max(1, int(len(items) * val_frac))
    val, tr = items[:nv], items[nv:]
    r = GlobalRenderer(kind)
    head = Head(FEATURE_DIM, r.num_params)
    feats = torch.stack([it["feat"] for it in items])
    head.set_norm(feats)
    info = fit(head, r, tr, val, progress=progress)
    return head, r, feats, {**info, "n_pairs": len(items), "n_dropped": len(pairs) - len(items)}


# --------------------------------------------------------------------------- unpaired


def distort_recover_pairs(examples: list[torch.Tensor], renderer: GlobalRenderer, fx: FeatureExtractor,
                          k: int = 16, seed: int = 0) -> list[dict]:
    """Random global re-grades of each example; the model learns to undo them."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for ex in examples:
        for _ in range(k):
            th = torch.zeros(1, renderer.num_params)
            q = renderer.unpack(th)
            q["curve"].copy_(torch.randn(q["curve"].shape, generator=g) * 0.12)
            if renderer.kind == "per_channel":
                q["curve"].view(1, 3, renderer.K)[:, :, 0] = torch.randn(1, 3, generator=g) * 0.03
            q["dM"].copy_(torch.randn(q["dM"].shape, generator=g) * 0.04)
            q["bias"].copy_(torch.randn(q["bias"].shape, generator=g) * 0.02)
            with torch.no_grad():
                d = renderer(ex[None], th)[0]
            out.append({"feat": fx(d), "src_t": d, "tgt_t": ex})
    return out


def learn_unpaired(examples: list[torch.Tensor], inputs: list[torch.Tensor], fx: FeatureExtractor,
                   profile: LookProfile | None = None, regime_fn: Callable | None = None, steps: int = 1200,
                   seed: int = 0, progress: Callable | None = None):
    """Learn a look from example photos (no pairs).

    examples: in-style photos; inputs: typical *unedited* photos to be edited
    (the model's input domain). If ``profile`` is given, a look profile loss
    (per regime from ``regime_fn``) is added. Returns (head, renderer, feats, info).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    r = GlobalRenderer("per_channel")
    ex = [shrink(e, TRAIN_EDGE) for e in examples]
    pseudo = distort_recover_pairs(ex, r, fx, k=16, seed=seed)
    ins = [{"feat": fx(x), "src_t": shrink(x, TRAIN_EDGE)} for x in inputs]
    for it in ins:
        it["regime"] = regime_fn(it["src_t"]) if regime_fn else "day"
    ex_pix = torch.cat([lab_pixels(e[None], 4096) for e in ex])
    head = Head(FEATURE_DIM, r.num_params)
    feats = torch.stack([it["feat"] for it in ins + pseudo])
    head.set_norm(feats)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-2)
    for step in range(1, steps + 1):
        head.train()
        pb = random.sample(pseudo, min(8, len(pseudo)))
        pt = head(torch.stack([p["feat"] for p in pb]))
        rec = sum((r(p["src_t"][None], th[None])[0] - p["tgt_t"]).abs().mean() for p, th in zip(pb, pt)) / len(pb)
        loss = 2.0 * rec + 1e-3 * pt.pow(2).mean()
        if ins:
            bb = random.sample(ins, min(8, len(ins)))
            th = head(torch.stack([b["feat"] for b in bb]))
            outs = [r(b["src_t"][None], t[None]) for b, t in zip(bb, th)]
            swd = sliced_wasserstein(torch.cat([lab_pixels(o, 1024) for o in outs]), ex_pix)
            fid = sum(fidelity_loss(o, b["src_t"][None]) for o, b in zip(outs, bb)) / len(bb)
            loss = loss + 0.5 * swd + fid + 1e-3 * th.pow(2).mean()
            if profile is not None:
                loss = loss + sum(profile_loss(o, [b["regime"]], profile, ref=b["src_t"][None])
                                  for o, b in zip(outs, bb)) / len(bb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if progress and step % 25 == 0:
            progress(step, steps, float(loss))
    head.eval()
    return head, r, feats, {"steps": steps, "n_examples": len(examples), "n_inputs": len(inputs)}
