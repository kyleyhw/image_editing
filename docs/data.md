# Data and slider models

The editor's learned parts (the *Auto-correct* layer and the *Expert styles* looks) and its fitted
parts (the *Film stocks* looks) are slider settings of the image pipeline
([pipeline.md](pipeline.md)), learned or fitted from the data below. This page describes the data,
the models and how they were chosen; the measurements are in the
[Phase 20](../tests/reports/phase20_data.md) and [Phase 21](../tests/reports/phase21_models.md)
reports.

## Datasets

| Dataset | Content | Licence | Use |
|---|---|---|---|
| MIT-Adobe FiveK [[1]](#ref-fivek) | 5,000 photos, each edited by five retouchers (A–E) in Lightroom; split 3,500 / 500 / 1,000 (train / validation / test) | research use | Auto-correct (expert C); expert styles A–E; film fits (neutral photos) |
| RawTherapee Film Simulation Collection [[2]](#ref-rt) | 294 film-stock colour tables (HaldCLUTs) | CC BY-SA 4.0 | Film stocks |
| PPR10K [[3]](#ref-ppr) | 11,161 portraits, three retouchers | research use | not obtained: the host refuses downloads (quota) |
| raw.pixls.us | sample raw files | CC0 | raw-decoding tests only |

The datasets stay under `data/` and are not committed; `data/manifests/` (committed) lists every
file with its source, so the data can be rebuilt with `tools/fetch_fivek.py`.

### FiveK as stored here

The HuggingFace copy [[4]](#ref-hf) differs from the MIT release in two ways that matter
(Phase 20 report, §2):

- The *originals* are LibRaw renders of the raw files (camera white balance, BT.709 curve,
  auto-brightening), not Adobe's renders. Published FiveK scores are therefore not comparable.
- The *expert edits* are 8-bit **ProPhoto RGB**. They are decoded colour-managed to linear sRGB
  (`tools/fetch_fivek.decode_linear_srgb`); reading them as sRGB desaturates every target.

Pairs are resized in linear light to 128 px on the short side for training. One training pair
whose frames do not register (correlation of the aligned luminance $r < 0.6$) is excluded.

## Slider models

A slider model predicts a vector of pipeline sliders $s \in \mathbb{R}^n$ for a photo $x$:

$$
s = h(\phi(x)),
$$

where $\phi(x) \in \mathbb{R}^{1280}$ is the descriptor used throughout the project (frozen
ImageNet ResNet-18 pooled features, 512-D, plus per-channel cumulative histograms) and $h$ is a
small multilayer perceptron (one hidden layer). The browser computes $\phi$ with the same ONNX
backbone it already loads for the learned looks, so a model adds only its head (tens of kB).

### Loss

Training renders each photo through the differentiable reference pipeline $R$ and compares the
result with the expert's edit $y$ in Oklab [[5]](#ref-oklab), a space in which Euclidean distance
approximates perceived difference:

$$
\mathcal{L} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \big\lVert \mathrm{Oklab}(R(x; s))_p - \mathrm{Oklab}(y)_p \big\rVert_2
\; + \; \lambda \sum_{i=1}^{n} \frac{|s_i|}{r_i},
$$

with $\Omega$ the pixels and $r_i$ the range of slider $i$. The $\ell_1$ term is needed because
the map $s \mapsto R(x; s)$ is not injective: exposure, whites and a curve can produce nearly the
same picture, so the image term alone leaves $s$ undetermined. The penalty picks the sparsest
explanation, which is also the most readable one for the user. $\lambda = 0.003$ was chosen on the
validation split.

### Oracle

For a fixed slider set, the *oracle* is the per-photo optimum
$s^\star(x) = \arg\min_s \mathcal{L}(x, y; s)$, found by gradient descent. It bounds what any model
predicting those sliders can reach, and separates two failure causes: a slider set too weak to
express the expert's edit (oracle far from the expert) and a model too weak to predict it (model far
from the oracle). For expert C the ten correction sliders reach 28.8 dB against 20.4 dB unedited,
so the slider set is adequate; the shipped model reaches 22.4 dB, so prediction is the limit.

### Auto-correct

Ten correction sliders (temperature, tint, exposure, contrast, highlights, shadows, whites,
blacks, vibrance, saturation), trained on expert C, the expert most often used as the FiveK
reference. Hidden width 64, dropout 0.3, weight decay $10^{-3}$, 12 epochs: a larger head (256)
over-fitted (validation flat while training loss fell). On the 1,000 test photos it improves
the unedited photos from 20.4 to 22.4 dB PSNR and from 9.95 to 7.92 mean ΔE00.

### Expert styles

The same model with the 24 colour-mixer sliders added (34 outputs), trained once per retoucher.
Each is a look in the editor's *Expert styles* group: the sliders are predicted for the photo when
the look is chosen, after which it is an ordinary slider look. Results are in the
[Phase 21 report](../tests/reports/phase21_models.md#expert-styles).

### Film stocks

Twelve stocks are fitted as fixed slider settings (tone, white balance, colour mixer, grading,
calibration and R, G, B curves) by minimising the same Oklab distance between the pipeline's
output and the film table's output, over 32 well-exposed FiveK photos; the error is measured on
16 others. Every stock is within 0.7–1.7 ΔE00 of its table, below the threshold of 2 at which a
residual colour table would have been added (plan A6.4).

## References

<span id="ref-fivek">[1]</span> Bychkovsky, V., Paris, S., Chan, E., Durand, F. (2011). *Learning photographic global tonal adjustment with a database of input/output image pairs.* CVPR. [Link](https://data.csail.mit.edu/graphics/fivek/)

<span id="ref-rt">[2]</span> Pat David, Pavlov Dmitry, Michael Ezra. *RawTherapee Film Simulation Collection*, CC BY-SA 4.0. [Link](https://rawpedia.rawtherapee.com/Film_Simulation)

<span id="ref-ppr">[3]</span> Liang, J., Zeng, H., Cui, M., Xie, X., Zhang, L. (2021). *PPR10K: A large-scale portrait photo retouching dataset with human-region mask and group-level consistency.* CVPR. [Link](https://github.com/csjliang/PPR10K)

<span id="ref-hf">[4]</span> `logasja/mit-adobe-fivek`, HuggingFace dataset. [Link](https://huggingface.co/datasets/logasja/mit-adobe-fivek)

<span id="ref-oklab">[5]</span> Ottosson, B. (2020). *A perceptual color space for image processing.* [Link](https://bottosson.github.io/posts/oklab/)
