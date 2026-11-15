# Image pipeline

The editor, the looks and training all use one pipeline (plan amendment A6.3). Its reference
implementation is [`photostyle/pipeline.py`](../photostyle/pipeline.py): differentiable PyTorch,
the source of every constant. The browser runs a WebGL2 port,
[`studio/web/src/lib/pipeline.js`](../studio/web/src/lib/pipeline.js), which reads the same
constants from `pipeline_spec.json` (written by `python -m photostyle.pipeline`). The port agrees
with the reference to within 1 code value in 129 of 130 test cases
([report](../tests/reports/phase19_editor.md)).

## Order of operations

```
display sRGB ──► geometry ──► [look, Phase A5] ──► linear Rec. 2020
   │                                                    │
   │    1 white balance ─► 2 exposure ─► 3 tone ─► 4 point curves ─► 5 colour (Oklab) ─► 6 vignette
   │                                                                                       │
   └──────────── output: linear sRGB, luminance-preserving gamut clip, sRGB encode, dither ◄┘
```

The order follows the convention of raw editors: correct the light (white balance, exposure), then
shape tone, then colour, then the finishing that depends on position. Each stage is described
below with the reason for its formulation.

## Working space

Let $x \in [0,1]^3$ be a display-encoded sRGB pixel. It is linearised with the sRGB EOTF
[[1]](#ref-iec)

$$
\operatorname{dec}(x) = \begin{cases} x/12.92, & x \le 0.04045 \\ \left(\frac{x+0.055}{1.055}\right)^{2.4}, & x > 0.04045 \end{cases}
$$

and mapped to linear Rec. 2020 [[2]](#ref-bt2020) by $c = M_{\text{sRGB}\to2020}\,\operatorname{dec}(x)$,
with $M_{\text{sRGB}\to2020} = M_{2020\to XYZ}^{-1} M_{\text{sRGB}\to XYZ}$.

- **Linear light**, because exposure and white balance are physically multiplications of radiance.
- **A wide gamut**, so that saturating a colour or moving a hue does not clip on the way. Only the
  final output is mapped into sRGB.
- **32-bit float** throughout the shader, so raising shadows does not band. Only the input texture
  and the canvas are 8-bit, and the output is dithered by ±1 code value (triangular).

Above 1 the power branch of $\operatorname{dec}$ continues, so values pushed over white survive a
round trip and can be recovered by the tone stage.

## 1. White balance

White balance is a von Kries adaptation: a diagonal gain $c \leftarrow \operatorname{diag}(k)\,c$.

**Temperature.** The slider $T \in [-100, 100]$ moves the assumed illuminant along the Planckian
locus in mired (reciprocal megakelvin), the scale on which colour-temperature differences are
roughly perceptually even:

$$
\text{mired}_{\text{src}} = \frac{10^6}{6504} - 0.8\,T .
$$

The span of ±80 mired runs from 4278 K to 13 559 K. That is about the gap between tungsten-halogen
light and daylight, the range a temperature slider for already-rendered photos is expected to
correct. The blackbody chromaticity $(x_K, y_K)$ comes from the cubic fit of Kim et al.
[[3]](#ref-kim). The white $w(K) = M_{2020\to XYZ}^{-1}\,(x_K/y_K,\ 1,\ (1-x_K-y_K)/y_K)$ then gives

$$
k = \frac{w(6504\,\text{K})}{w(K_{\text{src}})} ,
$$

so $T = 0$ is the identity. $T > 0$ assumes a bluer illuminant, which divides blue down: the photo
gets warmer, as in Lightroom.

**Tint.** The green gain is multiplied by $2^{-0.35\,t/100}$. $t = +100$ is magenta (green −22 %),
as in Lightroom.

**Normalisation.** Finally $k \leftarrow k / (\ell^\top k)$, with $\ell$ the luminance row of
$M_{2020\to XYZ}$. A neutral keeps its luminance, and black stays black. The previous additive
"warmth" offset tinted black; this is the fix.

## 2. Exposure

$c \leftarrow 2^{E} c$, with $E \in [-5, 5]$ stops. +1 doubles linear light exactly
(`tests/test_pipeline.py::test_exposure_doubles_linear_light`).

## 3. Tone

The six tone sliders (contrast, highlights, shadows, whites, blacks, plus the exposure already
applied) act on **luminance only**. This removes the hue shifts that per-channel curves cause
(the "green sky" failure of the A5 looks).

Let $Y = \ell^\top c$ and the perceptual luminance $L = \operatorname{enc}(Y)$, using the sRGB
encoding as the perceptual scale. The sliders displace the ordinates of a curve with abscissae

$$
x = (0,\ 0.06,\ 0.18,\ 0.4614,\ 0.75,\ 0.92,\ 1,\ 1.35) .
$$

- 0.4614 is $\operatorname{enc}(0.18)$, mid grey and the contrast pivot.
- 1.35 is $\operatorname{enc}(2)$, one stop over white. With a point there, a photo pushed over
  white by exposure can be brought back.

The ordinates are $y = x + \sum_s \frac{v_s}{100} r_s$, where $r_s$ is the displacement row of
slider $s$ (`TONE_ROWS` in the code). For example, shadows +100 lifts the shadow point by 0.10.
Highlights below 0 also pull the over-white point down by $0.35 \cdot |v|/100$, which recovers one
stop at −100. The ordinates are made strictly increasing by $y_i \leftarrow \max(y_i, y_{i-1} +
10^{-4})$, then interpolated by **PCHIP**, the monotone cubic of Fritsch and Carlson
[[4]](#ref-fc). With slopes

$$
m_i = \frac{w_1 + w_2}{w_1/d_{i-1} + w_2/d_i}\ \text{if}\ d_{i-1} d_i > 0,\ \text{else}\ 0, \qquad
w_1 = 2h_i + h_{i-1},\quad w_2 = h_i + 2h_{i-1},
$$

where $d_i$ are the secant slopes and $h_i$ the knot spacings, the interpolant is monotone
wherever the data are. **Tones therefore never invert**, whatever the sliders
(`test_tone_is_monotone`).

The new luminance $Y' = \operatorname{dec}(L')$ is applied as a ratio:

$$
c' = \kappa\, c + (Y' - \kappa Y), \qquad \kappa = \frac{Y'}{\max(Y,\ 10^{-5})} .
$$

Wherever $Y \ge 10^{-5}$ this is a pure scaling $c' = (Y'/Y)\,c$, which keeps chromaticity and
hence hue (`test_tone_preserves_hue`: under 2° of Oklab hue). Below the floor, the rest of the new
luminance is added as grey, so a lifted black becomes grey rather than dividing by zero. The floor
$10^{-5}$ is 0.03 of a code value in sRGB, below visibility.

## 4. Point curves

These are Lightroom's point curves: **RGB** (the same curve on each channel), then **R**, **G**,
**B**. They act on display-encoded values, as the panel shows them, with PCHIP between up to 16
points. Below the first point and above the last the curve is flat, so a raised black point lifts
everything below it, as in Lightroom. Unlike the tone sliders these are per-channel on purpose:
"faded" and cross-processed looks rely on per-channel endpoints, which is why tutorials use them.

## 5. Colour (Oklab)

Colour work happens in **Oklab** [[5]](#ref-oklab), a perceptual space whose hue angle is far more
uniform than HSV's: rotating a hue keeps lightness, and scaling chroma keeps hue. With
$C = \sqrt{a^2+b^2}$ and $h = \operatorname{atan2}(b, a)$:

- **Calibration** (before Oklab): $c \leftarrow Y + M_{\text{cal}}(c - Y)$, a matrix on chroma
  only, so neutrals are untouched. Each hue slider moves up to 35 % of a primary into its
  neighbour, as in [`develop.py`](../photostyle/develop.py).
- **Saturation and vibrance:**

  $$
  C \leftarrow C\,(1 + s)\,\bigl(1 + v\,(1 - \min(C/0.25,\,1))^2\bigr)
  $$

  with $s, v \in [-1, 1]$. Saturation −100 gives exactly zero chroma (grey). Vibrance acts most on
  muted colours and not at all from $C = 0.25$, about the chroma of the sRGB primaries.
- **Colour mixer (HSL).** There are eight bands at the Oklab hues of fully saturated sRGB red,
  orange, yellow, green, aqua, blue, purple and magenta (29°, 53°, 110°, 143°, 195°, 264°, 294°,
  344°). A hue lying a fraction $u$ of the way from band $k$ to band $k+1$ has weights
  $w_k = \cos^2(\pi u/2)$ and $w_{k+1} = \sin^2(\pi u/2)$. These weights are smooth and sum to 1
  at every hue (`test_band_weights_partition_of_unity`), so neighbouring bands blend without seams.
  Hue ±100 rotates by ±30°, saturation scales chroma by $1 + \sum_k w_k s_k$, and luminance scales
  Oklab $L$ by up to ±30 %. Hue and luminance changes are weighted by
  $\operatorname{smoothstep}(C/0.08)$, so greys are unaffected: chroma 0.08 is roughly skin.
- **Black & white.** $C \leftarrow 0$, with $L \leftarrow L\,(1 + 0.4 \sum_k w_k m_k)$ weighted
  the same way, so each band's grey can be brightened or darkened.
- **Colour grading.** Shadow, midtone and highlight masks are $(1-p)^{2/\beta}$,
  $1 - (1-p)^{2/\beta} - p^{2/\beta}$ and $p^{2/\beta}$. Here $p = L + \text{balance}/200$ and
  $\beta = 0.5 + \text{blending}/100$, so the default blending of 50 gives $(1-p)^2$, $2p(1-p)$
  and $p^2$. Each wheel adds up to 0.08 of Oklab chroma toward its hue. The wheel's hue is in
  conventional (HSV) degrees and is mapped through the band anchors, so "30° orange" means sRGB
  orange.

## 6. Vignette

$c \leftarrow c \cdot 2^{2\,(a/100)\,m(r)}$, an exposure-like change of up to ±2 stops in the
corners. The mask $m = \operatorname{smoothstep}\bigl((r - r_0)/(r_1 - r_0)\bigr)$ runs from
midpoint and feather. The radius $r$ is a $p$-norm distance normalised to 1 in the corners:
roundness +100 is a circle, 0 follows the frame and −100 is squarer. The vignette is measured on
the cropped frame, like Lightroom's post-crop vignette.

## Output

The working colour is converted to linear sRGB. Out-of-range colours are brought into gamut along
the line to their own luminance,

$$
c \leftarrow Y + t\,(c - Y), \qquad t = \max\{t \le 1 : Y + t(c - Y) \in [0,1]^3\},
$$

which keeps hue and luminance and gives up chroma, instead of clipping channels, which shifts hue.
The result is then sRGB-encoded.

## Geometry

Crop and straighten come first. A crop is $(x, y, w, h)$ as fractions of the frame, plus an angle
$\theta$. Output pixel $(u, v)$ samples the source at

$$
s = B\,R_\theta\,C\,P\,(u, v, 1)^\top,
$$

where $P$ scales the crop to pixels, $C$ centres, $R_\theta$ rotates and $B$ returns to texture
coordinates. Straightening by $\theta$ keeps the largest centred crop of the same aspect inside the
rotated frame, at scale

$$
\min\!\left(\frac{W}{W\cos\theta + H\sin\theta},\ \frac{H}{W\sin\theta + H\cos\theta}\right).
$$

## Looks and adjustments

The picture is the look first (the A5 renderer: per-channel curves, matrix, vignette, scaled by
Strength), then the photo's adjustments above. The adjustments belong to the photo and are kept
when the look changes, and undo covers look changes. In Phase 21 the looks move into the same
slider space (A6.4).

## Constants

Every constant is in [`photostyle/pipeline.py`](../photostyle/pipeline.py) with the reason for its
value. The amounts per slider (for example shadows +100 = +0.10 of perceptual luminance) follow
[`develop.py`](../photostyle/develop.py), which was written to follow Lightroom's documented
behaviour. Adobe does not publish its exact maths, so agreement with Lightroom's scales is
approximate and unverified (A6.6).

## References

<span id="ref-iec">[1]</span> IEC 61966-2-1:1999. *Multimedia systems and equipment — Colour measurement and management — Default RGB colour space — sRGB.*

<span id="ref-bt2020">[2]</span> ITU-R BT.2020-2 (2015). *Parameter values for ultra-high definition television systems.* [Link](https://www.itu.int/rec/R-REC-BT.2020)

<span id="ref-kim">[3]</span> Kim, Y.-S., Cho, B.-H., Kang, B.-S., & Hong, D.-I. (2002). *Color temperature conversion system and method using the same.* US Patent 7,024,034.

<span id="ref-fc">[4]</span> Fritsch, F. N., & Carlson, R. E. (1980). *Monotone piecewise cubic interpolation.* SIAM Journal on Numerical Analysis, 17(2), 238–246. [Link](https://doi.org/10.1137/0717021)

<span id="ref-oklab">[5]</span> Ottosson, B. (2020). *A perceptual color space for image processing.* [Link](https://bottosson.github.io/posts/oklab/)
