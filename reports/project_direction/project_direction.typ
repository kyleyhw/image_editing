// Research report: revised project direction.
// Build: python -c "import typst; typst.compile('project_direction.typ', output='project_direction.pdf')"

#set document(
  title: "Content-Adaptive, Editable Photo Styles: Revising the Project Direction",
  date: datetime(year: 2026, month: 9, day: 23),
)
#set page(paper: "a4", margin: (x: 1.5cm, top: 1.4cm, bottom: 1.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 9.2pt)
#set par(justify: true, leading: 0.52em, spacing: 0.62em)
#set heading(numbering: "1")
#show heading.where(level: 1): it => block(above: 0.8em, below: 0.45em)[
  #set text(size: 10pt, weight: "bold")
  #counter(heading).display() #h(0.4em) #it.body
]
#set math.equation(numbering: "(1)")
#show math.equation: set text(size: 9.2pt)
#set list(spacing: 0.45em, indent: 0.4em)
#set table(stroke: none, inset: (x: 3pt, y: 1.8pt))

#align(center)[
  #text(size: 14pt, weight: "bold")[Content-Adaptive, Editable Photo Styles:\ Revising the Project Direction]
  #v(0.15em)
  #text(size: 9.5pt)[Research report · `kyleyhw/image_editing` · 23 September 2026]
]

#v(0.3em)
#block(inset: (x: 1.2cm))[
  #set text(size: 8.8pt)
  #set par(leading: 0.48em)
  *Abstract.* The project set out to build a "general-purpose stylization engine" in which a fixed
  network learns any style from data. We argue this goal is ill-posed: its synthetic styles are
  fixed functions that need no learning. We reframe it as learning the *per-image* parametric edit
  that takes a photo to a target look, specified by paired or unpaired examples. We then review 18
  learned enhancement methods on MIT-Adobe FiveK. Image adaptivity, not spatial processing, carries
  most of the gain: a fixed 3D LUT reaches 20.37 dB PSNR, Zeng et al.'s image-adaptive blend 25.21 dB,
  and spatial extensions add at most ≈ 0.5 dB. We recommend a SepLUT-style renderer (per-channel curves,
  colour matrix, residual basis-LUT blend) driven by a style-conditioned head on frozen features, and a
  staged evaluation that begins with a data-scaling curve.
]

#show: rest => columns(2, gutter: 0.55cm, rest)

= Introduction
The current system predicts $P = 21$ renderer parameters (a tone curve shared by all channels, a
$3 times 3$ colour matrix and bias, grain, vignette) from a 1280-D descriptor: per-channel differentiable
CDFs concatenated with ResNet-18 features. It has been trained on three synthetic styles and on FiveK
expert C. The synthetic targets are closed-form maps $G(I)$ that apply identical parameters to every
image, so the network can at best imitate $G$, and running $G$ is faster and exact. Only the FiveK run
required learning: 80 training pairs reduced held-out $L_1$ to the expert from 0.0917 to 0.0741 (4/5
images improved), but 500 pairs raised it to 0.1030 (1/5). That result rests on five test images and one
seed, and training was confounded: a fully fine-tuned ResNet with BatchNorm at batch 4, no ImageNet
normalisation, and unequal epochs.

= Problem formulation
A *style* $s$ is a target appearance, not an operation. For an input $I$ the model predicts
$ theta = f(I, s), quad hat(I) = R(I; theta), $
where $R$ is a differentiable, editable renderer. The style is specified either by *paired* data
${(I_i, I_i^star)}$ (a photographer's edits) or by an *unpaired* set $cal(S)_s$ of in-style photos.
The defining property is content adaptivity: $theta$ must vary with $I$, because a dark street and a
bright beach need different edits to reach the same look. Constraints: CPU training, $10^1$–$10^2$
examples per style, outputs exportable as curves or a `.cube` LUT, no generated content.

= Background: image-adaptive 3D LUTs
Zeng et al. @zeng learn $N = 3$ basis LUTs $Phi_n$ on a $33^3$ RGB lattice. A 269K-parameter CNN
sees a $256^2$ thumbnail and predicts weights $w_n (I)$; the image's LUT is
$ Phi(I) = sum_(n=1)^N w_n (I) Phi_n , $
applied at full resolution by trilinear interpolation. Training minimises
$cal(L)_"MSE" + 10^(-4) cal(R)_s + 10 cal(R)_m$, where $cal(R)_s$ is a total-variation smoothness term
and $cal(R)_m$ penalises non-monotone outputs. The model has 593.5K parameters and runs in 1.66 ms per
4K image. On FiveK (expert C, 480p) it scores 25.21 dB paired and 22.86 dB when trained unpaired with a
WGAN-GP critic. Its ablation is the key evidence for this project: one *fixed* LUT scores 20.37 dB,
one adaptive LUT 23.15 dB, three 25.21 dB.

= Comparative analysis
@tab lists representative methods. Accuracy has converged within ≈ 0.5 dB, so size, conditioning and
editability decide.

#figure(
  placement: none,
  caption: [FiveK expert C PSNR (480p; † other protocol). MS: multi-style;
    UP: unpaired; Ed: editable output; n/r: not reported.],
  text(size: 7.6pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (left, right, right, center, center, center),
      table.hline(stroke: 0.6pt),
      [*Method*], [*dB*], [*Params*], [*MS*], [*UP*], [*Ed*],
      table.hline(stroke: 0.4pt),
      [3D LUT @zeng], [25.21], [593K], [–], [✓], [LUT],
      [AdaInt @adaint], [25.49], [620K], [–], [–], [–],
      [SepLUT-S @seplut], [25.42], [47K], [–], [–], [curves+LUT],
      [SA-3DLUT @sa3dlut], [25.50], [4.5M], [–], [–], [–],
      [CSRNet @csrnet], [25.17], [36K], [–], [–], [–],
      [HDRNet @hdrnet], [24.66], [482K], [–], [–], [–],
      [StarEnhancer @star], [25.29†], [n/r], [✓], [few-shot], [curves],
      [PieNet @pienet], [25.28†], [n/r], [✓], [few-shot], [–],
      [Exposure @exposure], [21.32], [8.6M], [–], [✓], [filters],
      [InstructP2P @ip2p], [n/a], [>860M], [text], [–], [–],
      table.hline(stroke: 0.6pt),
    )
  ],
) <tab>

Four findings follow. (i) *Adaptivity dominates*: spatially aware LUTs add 0.26 dB on FiveK
@sa3dlut, and global CSRNet beats local HDRNet. (ii) *Separable is cheaper*: per-channel 1D curves
before a $9^3$ LUT match Zeng at one-twelfth the size, and SepLUT's LUT generator *is* Zeng's
basis blend @seplut. (iii) *Only StarEnhancer* combines multiple styles, few-shot new styles from
example images, and editable curves @star; CSRNet shows fine-tuning only the conditioning matches
retraining for a new expert @csrnet. (iv) *Unpaired training is costly*: −2.35 dB for Zeng, and most
unpaired rivals barely beat Camera Raw's auto setting (21.61 dB) @zeng. Pixel generators and diffusion
editors can alter content @ip2p and are excluded.

= Proposed approach
*Features.* Keep ResNet-18 + CDF, frozen and cached, with ImageNet normalisation. *Renderer.* A cascade
of three per-channel monotone curves, the $3 times 3$ matrix and bias, and a residual $9^3$ LUT
$sum_n w_n Phi_n$ over $N approx 4$–$8$ shared bases with Zeng's regularisers, followed by grain and
vignette. The composite bakes into one $33^3$ `.cube`. *Conditioning.* A small head without
BatchNorm, modulated by FiLM on a style code: a learned embedding for known styles, or the mean
embedding of $n$ example images for new ones @star @pienet. *Unpaired styles.* A batch-level CDF /
sliced-Wasserstein colour loss, an identity anchor and distort-and-recover pseudo-pairs, with a small
WGAN-GP critic only if needed.

= Limitations and evaluation plan
No study reports these methods below 675 training pairs; the combination is untested; style codes
from few images of subtle photographers are noisy (StarEnhancer's expert Recall\@1 is as low as 24.6%);
the LUT stage is not slider-editable. We therefore gate each step: (1) remove the confounds and
measure a learning curve from 25 to 4,500 pairs (3 seeds, ≥ 100 test images, identity, mean-edit,
histogram-matching and per-image oracle baselines); (2) ablate renderers against CSRNet; (3) train on
experts A–D and hold out E; (4) test unpaired losses on disjoint FiveK images; (5) verify `.cube` export.

= Conclusion
The project's value lies where no closed-form edit exists: content-dependent looks learned from
examples. The evidence favours a compact, global, image-adaptive and style-conditioned renderer over
larger spatial or generative models. The first deliverable is the learning curve, not a new model.

#colbreak(weak: true)
#set text(size: 8pt)
#set heading(numbering: none)
#bibliography("refs.yml", title: "References", style: "ieee")
