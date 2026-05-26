# Architecture

This document derives the network and renderer from first principles and
records each design decision. It complements `PROJECT_PLAN.md` (status
ledger) and `tests/reports/phase3_to_phase6_report.md` (verification).

## 1. End-to-end pipeline

For an input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ and
a chosen *style* (Fujifilm, cyberpunk, tilt-shift, ...) the system
produces a styled image $\tilde{\mathbf{I}}$ via:

```
                          predicted
                          parameters
        ┌────────────┐     theta       ┌──────────────────┐
   I ─► │  Network   │ ───────────────►│  Renderer        │ ─► I_tilde
        │ phi(I)     │                 │  R(I; theta)     │
        └────────────┘                 └──────────────────┘
```

- The **network** $\varphi$ is a regression head on top of a global
  feature extractor. It returns a finite-dimensional parameter vector
  $\theta \in \mathbb{R}^P$.
- The **renderer** $R$ is a *differentiable* image-processing pipeline
  that consumes $\theta$ and produces $\tilde{\mathbf{I}} = R(\mathbf{I}; \theta)$.

The full pipeline is differentiable end-to-end, so a pixel-space loss
between $\tilde{\mathbf{I}}$ and a paired target $\mathbf{I}^\star$
back-propagates through $R$ into $\varphi$'s weights.

## 2. Feature extraction $\varphi$

The feature extractor concatenates two complementary descriptors of the
input image.

### 2.1 Differentiable CDF

For each colour channel $c \in \{R, G, B\}$ define a soft histogram
through Gaussian binning. Letting $\{b_k\}_{k=0}^{N-1}$ be $N=256$ bin
centres uniformly spaced in $[0, 1]$ and $\sigma$ a bandwidth:

$$
h_{c,k}(\mathbf{I}) \;=\; \sum_{x \in \mathbf{I}} \exp\!\left(-\frac{(\mathbf{I}_{x,c} - b_k)^2}{\sigma^2}\right),
\qquad
p_{c,k} = \frac{h_{c,k}}{\sum_{k'} h_{c, k'} + \varepsilon},
\qquad
F_{c,k} = \sum_{k' \le k} p_{c, k'}.
$$

$F_{c, \cdot} \in \mathbb{R}^{256}$ is the empirical CDF of channel $c$.
Soft binning replaces $\delta(b_k - \mathbf{I}_{x,c})$ with a Gaussian,
so the histogram is differentiable in $\mathbf{I}$ - essential for
back-propagating a CDF loss through the renderer.

The CDF block returns the concatenation
$\mathbf{F} = [F_R | F_G | F_B] \in \mathbb{R}^{768}$.

### 2.2 Spatial encoder

A ResNet-18 pre-trained on ImageNet, with its final fully-connected
layer removed, maps $\mathbf{I}$ to a 512-dimensional global feature
vector via the network's own adaptive-average-pooling layer. The
backbone is *not* frozen during training.

### 2.3 Combined descriptor

$$
\varphi(\mathbf{I}) \;=\; \big[\mathbf{F}(\mathbf{I}) \,\|\, \text{ResNet18}(\mathbf{I})\big] \;\in\; \mathbb{R}^{1280}.
$$

## 3. Transformation heads

Three regression heads share the same MLP shape
$1280 \to 512 \to 256 \to P$ (ReLU, BatchNorm, Dropout 0.3 after the
first hidden layer), with the final linear layer **zero-initialised**
so the renderer starts at the identity map at training step 0.

| Head | File | Output dim $P$ | Renderer |
|---|---|---|---|
| Fujifilm-specific | `models/transformation_head.py` | 7 | `DifferentiableFujifilm` |
| Generic | `models/generic_head.py` | $K-2 + 14 = 21$ for $K=9$ | `DifferentiableGenericRenderer` |
| Tilt-shift composite | `models/tilt_shift.py` | $21 + 3 = 24$ | `DifferentiableTiltShiftComposite` |

The tilt-shift head's bias on its three last outputs is *not* zero but
$(0.5, 0.2, 0.5)$ -- a warm start that places the focus band at the
image centre. The motivation is explained in §6.

## 4. Fujifilm-specific renderer (Phase 1/2 legacy)

Reproduces the Fujifilm "Classic Chrome" recipe with seven scalars:
highlight tone, shadow tone, saturation, WB red, WB blue, grain,
vignette. The colour pipeline uses *HSV-faithful* identities derived in
`models/differentiable_renderer.py`:

- Saturation scaling ($H, V$ preserved, $S \to \beta S$):

  $$c' = V - \beta\,(V - c), \quad V = \max_c c, \;\; \beta = 1 + 0.1\,\text{color}.$$

- Chrome effect ($H, S$ preserved, $V \to (1 - \alpha S)\,V$):

  $$c' = c \cdot \left(1 - \alpha\,\frac{V - m}{V}\right), \quad m = \min_c c.$$

These closed forms avoid a differentiable RGB$\leftrightarrow$HSV
round-trip (which has hue-branch discontinuities). The same chrome
constant $\alpha$ used during data generation is read from
`CHROME_STRENGTHS` in `data_generation/styles/fujifilm.py` and passed
to the renderer at construction, so train-time and inference-time
pipelines stay in lockstep.

## 5. Generic renderer (Phase 3)

Four primitives composed sequentially: tone curve → colour grade →
grain → vignette.

### 5.1 ToneCurve $(K = 9)$ control points

Knot positions $x_k = k/(K-1)$ are fixed. The endpoints are anchored at
$y_0 = 0$ and $y_{K-1} = 1$; the network supplies the $K-2 = 7$
interior offsets $\Delta y_k$ so that $y_k = x_k + \Delta y_k$. At
$\Delta y_k \equiv 0$ the curve is the identity.

For each input pixel value $x \in [0, 1]$:

$$
i = \lfloor x \,(K - 1) \rfloor, \quad u = x\,(K-1) - i, \quad
f(x) = y_i + u\,(y_{i+1} - y_i).
$$

Piecewise-linear interpolation is exactly differentiable in both
$\Delta y$ and $x$.

### 5.2 ColorMatrix

Per-pixel affine map:

$$
\mathbf{c}' = (\mathbf{I}_3 + d\mathbf{M})\,\mathbf{c} + \mathbf{b}, \qquad d\mathbf{M} \in \mathbb{R}^{3 \times 3}, \;\; \mathbf{b} \in \mathbb{R}^3.
$$

The network predicts $d\mathbf{M}$ as an offset from the identity, so
zero output keeps the colour unchanged. Models WB shifts, saturation,
hue rotation, and arbitrary linear grades.

### 5.3 Grain

$\tilde{\mathbf{I}} = \mathrm{clip}(\mathbf{I} + g\,\boldsymbol{\eta},\,0, 1),
\boldsymbol{\eta} \sim \mathcal{N}(0, 1)$, applied unconditionally so
train and inference share the same noise statistics as the data
generator.

### 5.4 Vignette

$$
\mathrm{mask}(x, y) = \mathrm{clip}\!\left(1 - s \cdot \frac{X^2 + Y^2}{2},\,0,\,1\right), \quad (X, Y) \in [-1, 1]^2.
$$

### 5.5 Parameter layout

$$
\theta_{\text{generic}} = \underbrace{[\Delta y_1, \ldots, \Delta y_{K-2}]}_{7} \;\Vert\; \underbrace{\mathrm{vec}(d\mathbf{M})}_{9} \;\Vert\; \underbrace{\mathbf{b}}_{3} \;\Vert\; \underbrace{g}_{1} \;\Vert\; \underbrace{s}_{1} \;\in\; \mathbb{R}^{21}.
$$

## 6. Tilt-shift composite (Phase 5)

The generic renderer is composed with a spatially-variant blur stage:

$$
R_{\text{tilt}}(\mathbf{I}; \theta) = T\!\left(R_{\text{generic}}(\mathbf{I}; \theta_{1:21});\; \theta_{22:24}\right).
$$

### 6.1 The tilt-shift primitive

Three scalars $(c_y, w, \sigma_{\!s}) = \theta_{22:24}$ define a
horizontal focus band centred at $y = c_y$ of half-width $w/2$.
A pre-computed Gaussian blur $B(\mathbf{I})$ (single, fixed
$\sigma_{\max}$) is blended with the input via a smooth mask:

$$
m(y) = 1 - \mathcal{S}\!\left(\frac{|y - c_y| - w/2}{f}\right), \qquad
\mathcal{S}(t) = 3t^2 - 2t^3 \;\;\text{clamped to}\;\;[0, 1],
$$

$$
T(\mathbf{I})_{x, y} = \mathbf{I}_{x, y} \,\big(1 - (1 - m(y))\,\sigma_{\!s}\big) + B(\mathbf{I})_{x, y} \, (1 - m(y))\, \sigma_{\!s}.
$$

$f$ is a fixed feather length (0.15 by default).
$\mathcal{S}$ is $C^1$ continuous, so the blend has no visible seam.

### 6.2 Why scalar parameters, not parameter maps

The project plan invited investigation of per-pixel parameter maps via
a U-Net-style decoder. The current design parametrises only the
*spatial structure* (which pixels get blurred) with three numbers and
lets the renderer compute the per-pixel blur weight by closed form.
This keeps the head shape identical to the generic case (no encoder
decoder) at the cost of restricting the spatial pattern to a horizontal
focus band. A genuine U-Net head remains future work; the current code
is the minimum spatially-variant extension.

### 6.3 Warm-start for the focus band

At zero head outputs the focus band would have $c_y = 0, w = 0$
(degenerate, at the image corner). The L1 gradient of the blend with
respect to $\sigma_{\!s}$ is then inconsistent across the image and
training collapses to $\sigma_{\!s} \to 0$. The fix is to leave the
final layer's *weight* at zero but bias the last three outputs to
$(0.5, 0.2, 0.5)$, so training begins from a moderate centred band
that the gradient can refine. See `TiltShiftStyleNet.TILT_SHIFT_BIAS_INIT`.

## 7. Composite loss

For predicted $\tilde{\mathbf{I}}$ and target $\mathbf{I}^\star$:

$$
\mathcal{L}(\tilde{\mathbf{I}}, \mathbf{I}^\star) = \lambda_{\text{pixel}}\,\|\tilde{\mathbf{I}} - \mathbf{I}^\star\|_1 \;+\; \lambda_{\text{percep}}\,\frac{1}{|\mathcal{T}|}\sum_{\ell \in \mathcal{T}}\|\Phi_\ell(\tilde{\mathbf{I}}) - \Phi_\ell(\mathbf{I}^\star)\|_1 \;+\; \lambda_{\text{cdf}}\,\|F(\tilde{\mathbf{I}}) - F(\mathbf{I}^\star)\|_1,
$$

where:

- $\Phi_\ell$ are activations of a frozen ImageNet-pretrained VGG-16 at
  $\mathcal{T} = \{\text{relu1\_2}, \text{relu2\_2}, \text{relu3\_3}, \text{relu4\_3}\}$;
- $F$ is the differentiable CDF from §2.1.

Default weights $(\lambda_{\text{pixel}}, \lambda_{\text{percep}}, \lambda_{\text{cdf}}) = (1, 0.05, 1)$ put the three components in similar order of magnitude
at the start of training.

Each component answers a different question:

| Component | Penalises mismatches in |
|---|---|
| $L_1$ pixel | direct numerical agreement |
| $L_1$ VGG features | perceptually salient structure (colour, texture, edges) at multiple scales |
| $L_1$ CDF | global tonal / colour distribution per channel |

## 8. Identity at initialisation

A standing invariant: at the start of training, regardless of style,
the renderer must produce $\tilde{\mathbf{I}} = \mathbf{I}$ (or, for the
warm-started tilt-shift head, a faithful centred-focus-band tilt-shift).
This is enforced by:

1. Zero-initialising the final linear layer of every head's weight.
2. Zero-initialising the bias for the generic / Fujifilm dimensions.
3. The renderer's primitives being identity at zero parameters
   (e.g. $\Delta y_k = 0 \Rightarrow$ tone curve $= x$, $d\mathbf{M} = 0 \Rightarrow$ colour matrix $= I_3$, $g = 0 \Rightarrow$ no grain, $s = 0 \Rightarrow$ unit vignette mask).

The first loss the optimiser sees is exactly $\mathcal{L}(\mathbf{I}, \mathbf{I}^\star)$, i.e. the do-nothing baseline. Any loss decrease
unambiguously corresponds to a learned non-trivial transformation.

## 9. References

<span id="ref-he-2016">[1]</span> He, K., Zhang, X., Ren, S. & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. [Link](https://doi.org/10.1109/CVPR.2016.90) — ResNet-18 spatial backbone.

<span id="ref-simonyan-2014">[2]</span> Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556. [Link](https://arxiv.org/abs/1409.1556) — VGG-16 perceptual features.

<span id="ref-johnson-2016">[3]</span> Johnson, J., Alahi, A. & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* ECCV. [Link](https://arxiv.org/abs/1603.08155) — VGG feature taps `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` used here.

<span id="ref-bychkovsky-2011">[4]</span> Bychkovsky, V., Paris, S., Chan, E. & Durand, F. (2011). *Learning Photographic Global Tonal Adjustment with a Database of Input/Output Image Pairs.* CVPR. [Link](https://doi.org/10.1109/CVPR.2011.5995332) — MIT-Adobe FiveK dataset, used by `MIT5KDataset`.
