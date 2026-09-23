# Unpaired Style Learning, Multi-Style / Personalized Conditioning, and Editable Outputs for Photo Retouching

Scope: methods that (a) learn a retouching style from an unpaired set of in-style images, (b) condition one model on several styles or photographers, (c) adapt to a new style from a few references, and (d) produce editable, non-hallucinating (parametric) outputs. All numbers below were taken from the primary papers' PDFs (text extracted and read directly). "FiveK" means MIT-Adobe FiveK, with Expert C as the target unless stated otherwise.

**Cross-paper caveat on FiveK numbers:** papers use different FiveK renditions (the "480p/UPE" split vs the "5K-Dark" input rendition from Hu et al.), different train/test splits (4500/500 paired vs 2250/2250/500 unpaired), and sometimes re-train baselines while other times reusing released weights. The same method can therefore get very different PSNRs in different papers. For example, Exposure/White-Box scores 21.32 dB in Zeng et al., 19.74 dB in UEGAN, 18.57 dB in StarEnhancer, and 18.59 dB in NeurOp. Compare numbers only within one table.

---

## 1. Exposure (Hu et al., ACM TOG 2018): RL + GAN over differentiable, resolution-independent filters, trained on unpaired data

### Takeaway
Exposure is the original "white-box" unpaired style learner. An RL actor picks a sequence of 5 global, differentiable filters and their parameters from a 64×64 thumbnail, and a WGAN-GP critic scores the result against an unpaired set of in-style photos. It learns a style from only about 370–2,000 target images in under 3 GPU-hours, and the filters apply at any resolution and stay editable. Its FiveK fidelity numbers are weak and training needs several stabilisation tricks.

### Cited Findings
- **Design principles:** filters must be (1) differentiable, (2) resolution-independent: parameters are estimated on a 64×64 downsample and applied to the full-res RAW, and (3) understandable, "so that the generated operation sequence can be understood by users… It would also enable them to further adjust the parameters." The authors argue a black-box output "might not be invertible, leaving users unable to undo any unwanted effects." — [Hu et al. 2018, arXiv:1709.09602](https://arxiv.org/abs/1709.09602)
- **Filters (8):** exposure (pO = 2^E·pI), white balance (per-channel gains), gamma, saturation, contrast, black & white, tone curve, and color curves. Curves are monotonic piecewise-linear with L parameters, and the prefix-sum formulation keeps them differentiable. All are global pixel-wise mappings. — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **RL formulation:** the policy has two parts. π1 is a stochastic discrete choice of filter; π2 is a deterministic choice of that filter's continuous parameters. The state includes 8 boolean "filter used" planes plus a step counter. Episodes are fixed at 5 edits, because "doing so makes training more stable than having the network learn when to stop itself." The reward is the incremental improvement in discriminator score minus penalties. — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Adversarial part:** a WGAN with gradient penalty (Gulrajani et al.). The discriminator gets more iterations and a higher learning rate (n_critic = 5; lr 5e-5 vs 1.5e-5 for the policy) so that it "saturates" and gives a tight EMD bound. — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Stability tricks, which the authors say are needed because "Both RL algorithms and GANs are known to be hard to train":**
  - An entropy penalty on π1: R′ = R − 0.05(log|F| + Σ π1 log π1).
  - A −1 penalty for reusing a filter.
  - An "out-of-order" trajectory buffer (2,048 images in flight), which acts like experience replay for RL and like a history buffer for the GAN.
  
  — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Data:**
  - FiveK split into three disjoint parts: 2,000 RAW inputs, 2,000 Expert-C retouches of *other* images, and 1,000 RAW test images.
  - Two 500px.com artists with 369 and 397 photos.
  
  — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Cost:** training takes "less than 3 hours for all the experiments". Inference is 30 ms on a TITAN X (Maxwell) and the model is under 30 MB (TensorFlow). For comparison, CycleGAN took 30 h to train at 500×333. — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Evaluation, since there are no pairs:** histogram intersection of luminance, contrast and saturation distributions against the target set, plus AMT ratings from 1 to 5 (100 images, 5 ratings each).

  | Target | Method | Luminance | Contrast | Saturation | AMT |
  |---|---|---|---|---|---|
  | FiveK C | Exposure | 71.3% | 83.7% | 69.7% | 3.43 |
  | FiveK C | CycleGAN | 61.4% | 71.1% | 82.6% | 2.47 |
  | FiveK C | Pix2pix (paired) | 92.4% | 83.3% | 86.5% | 3.37 |
  | FiveK C | Expert C | — | — | — | 3.66 |
  | 500px artist A | Exposure | 82.4% | 80.0% | 71.5% | 3.39 |
  | 500px artist A | CycleGAN | 63.6% | 45.2% | 71.8% | 2.69 |
  | 500px artist B | Exposure | 85.2% | 91.7% | 83.5% | 3.22 |
  | 500px artist B | CycleGAN | 60.1% | 79.4% | 83.4% | 2.86 |

  — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Limitations the authors state:**
  - Poor face tones.
  - Can't fix bad content, composition or lighting.
  - No denoising, so brightening amplifies shadow noise.
  - Global operations only. Local masks (gradient or luminance masks) are proposed as future work.
  - Only about 2×10³ training images.
  
  — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)
- **Later PSNR comparisons show Exposure as the weakest of the parametric methods:**
  - 21.32 dB / 0.864 SSIM / ΔE 12.65 on FiveK 480p unpaired, retrained ([Zeng et al., arXiv:2009.14468](https://arxiv.org/abs/2009.14468)).
  - 19.74 dB, using the released model ([UEGAN, arXiv:2012.15020](https://arxiv.org/abs/2012.15020)).
  - 18.57 dB on MIT-Adobe-5K-UPE ([StarEnhancer, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)).
  - 18.59 dB with 8.56M params on 5K-Dark ([NeurOp, arXiv:2207.08080](https://arxiv.org/abs/2207.08080)).
- Zeng et al. attribute White-Box's weaker results to "the unstability of reinforcement learning", and say that it and the other unpaired methods "are not stable in different scenes", producing over- or under-enhanced images. — [Zeng et al.](https://arxiv.org/abs/2009.14468)

### Inferences
- Exposure's recipe fits a small-data, "style from a folder of examples" setting: a parametric renderer, a discriminator on the output distribution, and a content anchor that is implicit because a global filter cannot hallucinate. The RL part is the fragile piece. Later work (Zeng's 3D-LUT GAN, below) shows a single-shot differentiable predictor plus a GAN can replace RL and be more stable.
- Training fit in under 3 GPU-hours on a 2017 GPU with a 64×64 predictor input. CPU training is plausible but slow; the paper does not test it.

### Gaps
- The Exposure paper reports no PSNR/SSIM. The later PSNRs come from other groups using different splits.
- The paper does not report per-filter usage statistics or a quantitative measure of variance across training runs.

---

## 2. Zeng et al. image-adaptive 3D LUT: unpaired GAN variant (TPAMI 2020/22)

### Takeaway
The image-adaptive 3D-LUT (a tiny CNN predicts weights that blend N basis LUTs) can be trained unpaired simply by replacing the MSE loss with a WGAN-GP loss plus an L2 content term, keeping the smoothness and monotonicity LUT regularizers. On FiveK it is the best unpaired method in the authors' table (22.86 dB at 480p), about 2.35 dB below its own paired version (25.21 dB).

### Cited Findings
- **Unpaired setup:** the generator G is the basis LUTs plus the CNN weight predictor. The discriminator D has the same architecture as the weight predictor, minus instance norm and dropout, with a 1-dim FC output. — [Zeng et al., arXiv:2009.14468](https://arxiv.org/abs/2009.14468)
- **Losses:**
  - Generator: L_G = E[−D(G(x))] + λ1·E‖G(x) − x‖², with λ1 = 1000 following DPE. The L2 term "ensures that the output image preserves the same content as the input".
  - Discriminator: WGAN-GP with λ2 = 10.
  - Total: L_unpaired = L_gan + λs·R_s (smoothness) + λm·R_m (monotonicity), with λs = 1e-4 and λm = 10.
  - Optimiser: Adam, batch size 1, learning rate 2e-4 unpaired (1e-4 paired). G and D are updated at an equal pace.
  
  — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **Data (DPE split):** 2,250 source RAW inputs and 2,250 Expert-C retouches of *different* images, with 500 test images. All model settings are identical to the paired case. — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **FiveK unpaired results (Table 4):**

  | Method | 480p PSNR | 480p SSIM | 480p ΔE | Full-res PSNR | Full-res SSIM | Full-res ΔE |
  |---|---|---|---|---|---|---|
  | Camera Raw auto | 21.61 | 0.854 | 11.83 | 21.55 | 0.861 | 11.98 |
  | Pix2Pix | 19.21 | 0.814 | 14.76 | N.A. | N.A. | N.A. |
  | CycleGAN | 20.98 | 0.831 | 13.28 | N.A. | N.A. | N.A. |
  | White-Box (Exposure) | 21.32 | 0.864 | 12.65 | 21.17 | 0.875 | 12.81 |
  | DPE | 21.99 | 0.875 | 11.40 | N.A. (OOM) | N.A. | N.A. |
  | UIE | 22.11 | 0.879 | 11.21 | 22.03 | 0.882 | 11.46 |
  | **3D-LUT unpaired** | **22.86** | **0.887** | **10.28** | **22.78** | **0.898** | **10.42** |

  — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **Paired counterpart (Table 3, same FiveK 480p):** 25.21 dB / 0.922 / ΔE 7.61. At full resolution: 25.10 / 0.930 / 7.72. Paired baselines: HDRNet 24.32, DPE 23.75. — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **Unpaired methods barely beat the no-training baseline:** "most of the unpaired enhancement methods, except ours, do not have obvious advantage over the automatic adjustment of Camera Raw… The main reason is that most unpaired methods are not stable enough." — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **Unpaired imaging-pipeline task (FiveK 480p, Table 6):**
  - Every method, including 3D-LUT at 21.60 dB, is worse than or equal to Camera Raw at 21.61 dB.
  - The other methods sit at 14.9–17.8 dB.
  - The authors explain that the L2 content constraint "can hardly hold" when input and target differ greatly in dynamic range, and that relaxing it hurts GAN stability.
  
  — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **Model size:** the model is small, about 593.5K params including LUTs, as tabulated by NeurOp. — [NeurOp, arXiv:2207.08080](https://arxiv.org/abs/2207.08080)

### Inferences
- The paired-to-unpaired gap is roughly 2.3–2.4 dB PSNR and about 2.7 ΔE on FiveK. That quantifies the cost of dropping pairs even with a strong parametric renderer.
- The L2(G(x), x) content anchor is crude: it penalises any large tonal change, which fights strong styles. A distribution loss on colour statistics, or an identity loss on in-style inputs as UEGAN uses, may be a better anchor for strong looks.
- The basis LUTs are global colour mappings, so the output is content-preserving by construction. The learned LUT can be exported (.cube), but it is not a small set of named sliders.

### Gaps
- Zeng et al. report no variance or seed sensitivity for GAN training, and no small-data (for example 50-image) experiment.

---

## 3. Other unpaired enhancers: DPE, EnlightenGAN, UEGAN (all output pixels, not parameters)

### Takeaway
DPE, EnlightenGAN and UEGAN are all encoder–decoder / U-Net image generators trained with GAN losses. They output pixels, not edit parameters, so they can change or distort content and are resolution-limited. In FiveK comparisons they score at or below the parametric unpaired 3D-LUT.

### Cited Findings
- **DPE (Chen et al., CVPR 2018):** a two-way GAN (CycleGAN-like) with three changes:
  - A U-Net generator "augmented with global features".
  - "WGAN with an adaptive weighting scheme", which the authors call less parameter-sensitive than WGAN-GP.
  - Individual batch-norm layers for the two generators.
  
  Input is fixed at 512×512. — [Chen et al. CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Deep_Photo_Enhancer_CVPR_2018_paper.pdf)
- **DPE numbers:** unpaired FiveK 480p is 21.99 dB / 0.875 in Zeng's table. At original resolution it is "unavailable because the DPE method is too memory" costly (OOM). Using the released pretrained model, UEGAN measured 22.36 dB / 0.8674. — [Zeng et al.](https://arxiv.org/abs/2009.14468); [UEGAN](https://arxiv.org/abs/2012.15020)
- **EnlightenGAN (Jiang et al., IEEE TIP 2021):** low-light enhancement trained with no paired supervision on a mix of 914 low-light and 1,016 normal-light unpaired images. It uses:
  - An attention-guided U-Net generator.
  - A global–local discriminator.
  - A "self-regularized" perceptual loss and attention.
  
  Retrained on FiveK by the UEGAN authors, it scored 16.96 dB, below the unprocessed input (17.42 dB). — [EnlightenGAN, arXiv:1906.06972](https://arxiv.org/abs/1906.06972); [UEGAN](https://arxiv.org/abs/2012.15020)
- **UEGAN (Ni et al., IEEE TIP 2020):** a single-generator encoder–decoder with a global attention module and modulation. Its losses:
  - A relativistic-discriminator **quality loss** that also treats real low-quality images as "fake".
  - A VGG-feature **fidelity loss** between input and output, used instead of a pixel L2 because "the generated high-quality image is typically different from the input… due to contrast stretching and color rendering".
  - An L1 **identity loss** on inputs that are already high quality.
  - Weights: λ_qua 0.05, λ_fid 1, λ_idt 0.1.
  
  — [Ni et al., arXiv:2012.15020](https://arxiv.org/abs/2012.15020)
- **UEGAN data and training:** FiveK split 2,250 input / 2,250 other-image Expert-C / 500 test, plus a separate target set of 2,000 Flickr "HDR"-tagged images. Images are resized so the long side is 512, with 256² crops. Trained 150 epochs on one RTX 2080 Ti. — [Ni et al.](https://arxiv.org/abs/2012.15020)
- **UEGAN FiveK results (Table I):**

  | Method | PSNR | SSIM | NIMA |
  |---|---|---|---|
  | Input | 17.42 | 0.8037 | 4.46 |
  | CycleGAN | 20.72 | 0.7825 | 4.37 |
  | Exposure | 19.74 | 0.8442 | 4.62 |
  | EnlightenGAN | 16.96 | 0.7562 | 4.25 |
  | DPE | 22.36 | 0.8674 | 4.54 |
  | **UEGAN** | **22.88** | **0.8882** | **4.76** |

  Exposure and DPE were run from released models; CycleGAN and EnlightenGAN were retrained. — [Ni et al.](https://arxiv.org/abs/2012.15020)
- **Pixel methods distort content:** Exposure's authors observed that CycleGAN and Pix2pix "generate vivid color but lead to edge distortions and degraded image quality", while Exposure "has no distortion and no limit on image resolution". — [Hu et al. 2018](https://arxiv.org/abs/1709.09602)

### Inferences
- UEGAN's loss design transfers directly to a parametric renderer (curves or LUT): a quality/adversarial loss against the style set, a perceptual fidelity loss, and an identity loss on already-in-style images. It avoids the over-strict pixel L2 anchor used by Zeng/DPE.
- UEGAN (22.88) and unpaired 3D-LUT (22.86) are near-identical in PSNR, but they come from different papers and splits. They are not directly comparable, and the parametric one scales to full resolution.

### Gaps
- There is no FiveK comparison of UEGAN vs unpaired 3D-LUT on the same split.
- EnlightenGAN was designed for low light, so its poor FiveK score may reflect task mismatch.

---

## 4. StarEnhancer (Song, Qian, Du; ICCV 2021): multi-style curves with a reference-image style encoder

### Takeaway
StarEnhancer is the closest published match to "one model, many photographers, new style from a handful of examples, editable output":
- A metric-learned style encoder turns n example images into a latent code (the mean of L2-normalised embeddings).
- A mapping network turns source and target codes into Dual-AdaIN statistics that modulate a curve encoder.
- The output is a set of 15 per-image curves, which are editable and run at more than 200 FPS at 4K.

Training is paired (L1 against the target-style image), but adapting to a new style needs only unpaired example images.

### Cited Findings
- **Enhancer:** a CNN curve encoder sees a K×K downsample and predicts knot points for 15 curves, one for each input channel i∈{r,g,b,x,y} to output channel j∈{r,g,b}. The x and y curves are *spatial* coordinate curves, which let it mimic gradient or elliptical filters. The curves are rendered via piecewise cubic interpolation and indexing into a residual image, O = R + I. It can apply a low-colour-depth mapping to high-bit-depth images. — [Song et al. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Style encoder:** first trained as a style classifier with a normalised-softmax (cosine, scale s) loss over style classes. At inference, "we feed n images of the specific style into the style encoder" and take the L2-normalised mean embedding as that style's latent code. — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Conditioning:** a mapping network maps source-style and target-style latent codes to per-layer (µ, σ) style codes. Dual AdaIN then computes F′ = σ_b·(F − µ_a)/σ_a + µ_b, using mapped statistics rather than feature statistics, because ordinary normalisation layers hurt performance. — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **User awareness and few-shot:** to avoid overfitting to class-centre codes, training also uses embeddings computed from *subsets* of a style's images. The paper states: "New users can select their preferred images in the shared gallery or use their collection to generate new target latent codes… paired images are not necessary for this procedure." — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Editability:** "For experts, all knot points of the predicted curves can be adjusted manually, just like the curve tool in Lightroom". Non-experts get per-curve strength sliders β (u′ = β·u) with real-time feedback and no CNN re-inference. — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Data:** all 12 FiveK styles: 5 experts, 4 camera input renditions, and 3 auto-retouch styles. The multi-style benchmark uses 10 styles: A–E, O/P/Q camera, and X/Y auto, where Y was regenerated with a recent Lightroom. Eight other styles are held out as *unseen*. — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Single-style FiveK-UPE results** (Expert C, 4500/500, test at original resolution; FPS at 4K on a TITAN RTX):

  | Method | PSNR | SSIM | LPIPS | FPS |
  |---|---|---|---|---|
  | Exposure | 18.57 | 0.701 | — | 0.11 |
  | DPE | 22.15 | 0.850 | — | — |
  | CURL | 24.20 | 0.880 | 0.108 | — |
  | DeepLPF | 24.48 | 0.887 | 0.103 | — |
  | HDRNet | 23.20 | 0.917 | 0.120 | 22 |
  | DeepUPE | 23.24 | 0.893 | 0.158 | 4.7 |
  | A3DLUT | 24.92 | 0.934 | 0.093 | 602 |
  | Basic (single-style) enhancer | 25.46 | 0.948 | 0.083 | 205 |
  | StarEnhancer (multi-style model) | 25.29 | 0.943 | 0.086 | — |

  The FPS for the StarEnhancer row was not recoverable from the text; it reads as sharing the curve renderer, and the abstract claims "over 200 FPS" at 4K. — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Multi-style results:**
  - The style encoder's Recall@1 per class ranges from 24.6% (A) to 85.0% (P). Expert styles are "significantly more difficult to distinguish than camera input styles".
  - Transfers between camera styles are easiest. Expert A is the hardest target.
  - The input style still leaks into the output: from style P the outputs "always have a cooler tone".
  - Example per-image PSNRs in Fig. 7 range from about 24.4 to 34.9 dB. The full A–E matrix is shown only as a figure.
  
  — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Unseen styles (Fig. 9):** the experiment was repeated 10 times at each sample size. The authors report:
  - "the more samples used tend to yield more reliable latent codes".
  - An unseen *source* is harder than an unseen *target*, and both unseen is hardest.
  - Even so, "our enhancer can outperform most of the enhancers fine-tuned on the train sets of these styles".
  
  — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)
- **Deployment:** the style encoder and mapping network run once. Style codes can be computed server-side, so devices keep only the curve-encoder weights. Encoders are shallow ResNets without batch norm (Fixup init). — [Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)

### Inferences
- StarEnhancer's supervision is paired: it needs the same image in both source and target style for the L1 loss. Its few-shot adaptation, however, is unpaired, because a new style only needs example images to compute a code. Combining this style-code conditioning with an unpaired adversarial or distribution loss (as in Exposure or unpaired 3D-LUT) would give a fully unpaired multi-style model. I found no paper that tests this exact combination.
- The low Recall@1 for expert styles (25–50%) suggests that a handful of references gives a noisy code for subtle photographer styles.

### Gaps
- The exact parameter count is not given in the text I extracted.
- The numeric unseen-style PSNR curve and the full multi-style matrix appear only as figures, so I could not read exact values.

---

## 5. PieNet (Kim, Koh, Kim; ECCV 2020): personalized enhancement via metric-learned preference vectors

### Takeaway
PieNet learns a "preference vector" for each user in a triplet-loss embedding space. A new user picks roughly 10–20 liked images, and their vector is the average of those images' embeddings, with no retraining. A U-Net-style decoder conditioned on that vector outputs a residual image. It is personalised and few-shot, but outputs pixels, not parameters.

### Cited Findings
- **New-user flow:** a user selects "about 10∼20 preferred images from a random set of images". — [Kim et al. ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
- **Metric learning:**
  - Twin embedding networks, ImageNet-pretrained, give 512-D L2-normalised features.
  - Per-user learnable preference vectors are trained with a triplet loss (margin 0.2) so that a user's liked images lie near their vector.
  - The five FiveK experts serve as training users. For each expert, their own retouches are positives, and the inputs plus other experts' retouches are negatives.
  - Training: 25k mini-batches of 64 triplets.
  
  — [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
- **Enhancer:**
  - A ResNet-18 encoder takes a 512×512 input.
  - The decoder uses "personalized up-sample blocks", each of which receives the preference vector.
  - The output is a delta image, Ĩ = I + ΔI.
  - Losses include a perceptual loss in the preference-embedding space (λp 0.4) and a TV-style loss on the delta (λt 0.01).
  - Preference vectors are perturbed with ‖n‖=0.1 noise for robustness.
  
  — [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
- **Few-shot schemes:**
  1. Optimise a new preference vector by triplet loss with the embedding frozen, using positive and negative images.
  2. The default: average the embeddings of preferred images only.
  
  With a large number of preferred images (N = 2,000), scheme 1 wins. With few images, averaging is preferred (text truncated in my extraction; bar plots in Fig. 6). — [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
- **FiveK results (Table 1):**

  | Method | Single user (C) PSNR | Single user (C) SSIM | Multi-user (A–E) mPSNR | Multi-user (A–E) mSSIM |
  |---|---|---|---|---|
  | White-Box | 18.36 | 0.810 | 17.83 | 0.799 |
  | Distort-and-Recover | 20.97 | 0.841 | 18.65 | 0.834 |
  | HDRNet | 23.44 | 0.882 | 21.64 | 0.872 |
  | DPE | 22.34 | 0.873 | 21.09 | 0.858 |
  | DeepUPE | 23.61 | 0.887 | 21.74 | 0.881 |
  | **PieNet** | **25.28** | **0.908** | **24.28** | **0.907** |

  Baselines give one output per input, so their multi-user score reuses that single output against every expert. — [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
- **Personalisation benchmark:** FiveK expanded to 28 "users". 20 are for training (11 Lightroom presets, 5 algorithms, experts A–D) and 8 are held out (4 presets, 3 algorithms, expert E). Evaluation is mPSNR/mSSIM vs N_pref, repeated 10×. — [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)
- **Earlier personalisation work:** PieNet's related work cites Kang et al., where users enhance about 25 representative images by controlling parameters and metric learning transfers those parameters, and Caicedo et al., who extend this with collaborative filtering across similar users. PieNet contrasts itself by needing only image *selection*, not parameter editing. — [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)

### Inferences
- PieNet's preference-vector mechanism (a metric-learned embedding, averaged over the user's liked images) matches StarEnhancer's style code in spirit. It could drive a parametric renderer instead of a pixel decoder. StarEnhancer is effectively that parametric version.
- A 512×512 pixel decoder with a residual output means results at full resolution require upsampling the delta, and are not editable sliders.

### Gaps
- I did not find the exact few-shot mPSNR values; they are in bar plots only.
- I did not find parameter counts or runtime in the text I read.
- I did not separately verify the original Bychkovsky et al. 2011 personalisation (collaborative-filtering) experiment from its primary source.

---

## 6. Editable white-box retouchers with strong paired numbers: NeurOp (ECCV 2022) and RSFNet (ICCV 2023)

### Takeaway
Both NeurOp and RSFNet are *paired, single-style* models. They are relevant as editable renderers that could be put under a style-conditioning or unpaired loss:
- NeurOp uses 3 learned "neural color operators", each controlled by one scalar strength slider, with only 28K params.
- RSFNet predicts region attention maps and named filter arguments (saturation, contrast, hue, temperature, and so on) that are summed in parallel. It is white-box and local.

### Cited Findings
- **NeurOp:** a neural colour operator maps an RGB value plus a scalar strength v∈[−1,1] to a new RGB value. It is built as encoder → feature translation scaled by v → decoder, and designed to be approximately homomorphic, like chaining exposure edits. A CNN strength predictor sets each operator's scalar from global feature statistics of intermediate images. The model has "only 28k parameters", and users can adjust "strengths using three sliders in real-time". — [Wang et al., arXiv:2207.08080](https://arxiv.org/abs/2207.08080)
- **NeurOp on MIT-Adobe-5K-Dark (Expert C):**

  | Method | PSNR | SSIM | ΔE | Params |
  |---|---|---|---|---|
  | NeurOp | 24.32 | 0.907 | 10.10 | 28,108 |
  | CSRNet | 23.86 | 0.897 | 10.57 | 36,489 |
  | 3D-LUT | 23.12 | 0.874 | 11.26 | 593,516 |
  | HDRNet | 22.65 | 0.880 | 11.83 | 482,080 |
  | White-Box | 18.59 | 0.797 | 17.42 | 8,561,762 |

  - On 5K-Lite, NeurOp scores 25.09 / 0.911 / 9.93.
  - On PPR10K, it scores 25.45–26.32 dB depending on subset, or up to 26.46 with HRP.
  - Runtime is about 4 ms at 500×333 and 19 ms for 1 MP.
  
  — [Wang et al.](https://arxiv.org/abs/2207.08080)
- **NeurOp ablation:** initialising the operators from standard operators beats random initialisation. NeurOp also beats fixed functional standard operators (23.78 dB) — the ablation reported random init PSNR 22.61 vs 24.32 for the full model. — [Wang et al.](https://arxiv.org/abs/2207.08080)
- **RSFNet:**
  - A parameter predictor outputs region attention maps and K×N filter arguments.
  - A renderer applies about 10 traditional filters (contrast, saturation, hue, temperature, shadows, midtones, highlights, shift, and others, from DaVinci-Resolve-style tools) per region and sums the increments linearly, rather than cascading them.
  - Variants constrain masks with palette, saliency or semantic segmentation for user editing.
  
  — [Ouyang et al. ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouyang_RSFNet_A_White-Box_Image_Retouching_Approach_using_Region-Specific_Color_Filters_ICCV_2023_paper.pdf)
- **RSFNet-map on FiveK (Expert C):**

  | Setting | PSNR | SSIM | ΔE | Runtime |
  |---|---|---|---|---|
  | 480p "zeroed with C's" | 25.49 | 0.924 | 7.23 | — |
  | 480p "zeroed as shot" | 24.64 | 0.915 | 9.16 | 9.98 ms |
  | Full resolution | 24.39 | 0.894 | — | 12.35 ms |

  - 3D-LUT+AdaInt scores 25.49 / 0.926 / 7.47 at 480p (C's rendition), and 24.24 dB at full resolution in 1.80 ms.
  - RSFNet-global, with no maps, scores 24.31 dB.
  - On PPR10K, RSFNet-map scores 25.58 dB (expert a), 24.81 (b) and 25.52 (c).
  
  — [Ouyang et al.](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouyang_RSFNet_A_White-Box_Image_Retouching_Approach_using_Region-Specific_Color_Filters_ICCV_2023_paper.pdf)

### Inferences
- Modern white-box models close the gap to black-box LUT/HDRNet methods on paired FiveK (about 24.3–25.5 dB) while staying editable. That makes them better candidates than Exposure's 8-filter set as the renderer in an unpaired or multi-style system.
- Neither paper tests unpaired training or multi-style conditioning.

### Gaps
- I did not find the RSFNet parameter count in the extracted text.
- Neither NeurOp nor RSFNet has a multi-expert or personalisation evaluation.

---

## 7. Neural Preset (Ke et al., CVPR 2023): DNCM, self-supervised training, style from a reference image

### Takeaway
Neural Preset transfers a colour style from a single reference image using Deterministic Neural Color Mapping (DNCM): a per-image k×k matrix (k=16, so 256 predicted numbers) applied identically to every pixel. It is trained self-supervised on COCO with random LUT/filter perturbations, needing no paired or style-labelled data. It runs at about 52 fps at 4K with 1.96 GB memory and 5.15M params. It is global only and is not a named-slider representation.

### Cited Findings
- **DNCM:** each pixel's RGB value is projected by P (3×k), multiplied by an image-adaptive T (k×k) predicted by an encoder, then projected back by Q (k×3). This is "arbitrary" colour mapping with "only a few hundred learnable parameters" per image, compared with about 10K for a 32-level 3D LUT, and it operates per pixel, so "avoiding artifacts". — [Ke et al., arXiv:2303.13511](https://arxiv.org/abs/2303.13511)
- **Two stages:**
  - nDNCM normalises the input's colour style.
  - sDNCM applies style parameters r_s extracted from a style image.
  - r_s "can be stored as presets" and reused for fast style switching.
  
  — [Ke et al.](https://arxiv.org/abs/2303.13511)
- **Self-supervised training:**
  - Two colour-perturbed copies I_i and I_j of the same image are made with random filters or LUTs, from about 5,000 LUT files plus a random filter adjustment strategy.
  - An L2 consistency loss makes their normalised versions match.
  - Style parameters are swapped between the copies, and an L1 reconstruction loss is applied: Y_i = sDNCM(Z_j, r_i) ≈ I_i.
  - Total: L = L_rec + 10·L_con.
  - Encoder: EfficientNet-B0 at 256² input. Adam, 32 epochs, batch 24.
  
  — [Ke et al.](https://arxiv.org/abs/2303.13511)
- **Efficiency on an RTX 3090 (FP32):**
  - 4K: 0.019 s / 1.96 GB. The text states "about 52 fps at 4K".
  - 8K: 0.061 s. The authors say 8K runs "over 16 fps".
  - Memory stays constant because processing is patch-wise.
  - Model size: 5.15M params, the lowest among the compared methods.
  - It is about 28× faster than PhotoWCT2 at 2K.
  - CPU timings are in their Appendix C.6, which I did not read.
  
  — [Ke et al.](https://arxiv.org/abs/2303.13511)
- **Other uses without fine-tuning:** low-light enhancement, underwater correction, dehazing and harmonisation. — [Ke et al.](https://arxiv.org/abs/2303.13511)
- **Limitations the authors state:**
  - It amplifies JPEG artifacts.
  - It may fail between images "with very different inherent colors".
  - It "cannot perform local-adaptive color mapping" — for example, it cannot map blue sky and water to different colours.
  
  — [Ke et al.](https://arxiv.org/abs/2303.13511)

### Inferences
- Neural Preset's self-supervised trick removes any need for style labels or pairs: learn to undo random colour perturbations of the same image. For "style from examples", the style code can be computed from one or many references and averaged. The paper demonstrates single reference images; averaging sDNCM parameters over a photographer's set is untested.
- A DNCM matrix is exportable, and since the whole map is a function of RGB only, it can be baked into a 3D LUT. It is not human-readable like curves or sliders.
- Reference-based transfer copies one image's palette, not a photographer's content-dependent *policy* (how they treat skies vs faces). This is a key difference from StarEnhancer and PieNet, which condition an image-adaptive predictor.

### Gaps
- There are no FiveK expert-style numbers; the paper evaluates colour style transfer with its own metrics and a user study.

---

## 8. NILUT (Conde et al., AAAI 2024): multiple LUT styles in one implicit MLP, with blending

### Takeaway
NILUT fits a small coordinate MLP, RGB→RGB, to reproduce existing 3D LUTs. A conditional variant (CNILUT) adds a one-hot style vector so that one network holds 3–5 LUT styles and can blend them by using non-one-hot condition vectors. It is supervised LUT *emulation*, not learning from photos, and it is not image-adaptive.

### Cited Findings
- **Training:** a 4096×4096 "RGB map" containing all 16M colours is edited in Photoshop with real photographer LUTs, and the MLP overfits the mapping. "We do not require natural images to learn real 3D LUTs". — [Conde et al., arXiv:2306.11920](https://arxiv.org/abs/2306.11920)
- **Architectures and results:** MLP, SIREN and residual MLP (MLP-Res) were compared.
  - MLP-Res 128×2 has 33.9K params and scores 45.34 dB / ΔE 0.97, averaged over 5 LUTs.
  - The 64×2 version has 8.7K params and scores 43.84 dB / 1.11.
  - All configurations score above 40 dB and below ΔE 1.5.
  - For comparison, a 33³ 3D LUT is about 107K float parameters.
  - MLP-Res reaches "almost perfect mapping in 4 minutes".
  
  — [Conde et al.](https://arxiv.org/abs/2306.11920)
- **CNILUT:** the input is RGB concatenated with a one-hot style vector c. Blending uses condition vectors such as [0.5, 0, 0.5]. Results:
  - Three LUTs in MLP-Res 128×2: 43.72 dB on average.
  - Five LUTs in MLP-Res 256×2: 45.34 dB and ΔE 0.85 on average.
  - There is a "slight performance degradation" vs separate NILUTs and it needs "longer and more complex training".
  - Blending "happens implicitly without additional computational cost".
  
  — [Conde et al.](https://arxiv.org/abs/2306.11920)

### Inferences
- The one-hot or blend conditioning is a cheap way to store several photographers' *global* looks in one tiny renderer, and it is CPU-trainable in minutes. It could be combined with an image-adaptive predictor that outputs the condition or blend weights. The paper does not test learning a NILUT from unpaired photos; that would need a distribution or adversarial loss (see section 10).
- Blends interpolate in the MLP's condition space, which is not guaranteed to equal a linear blend of the LUT outputs. The paper shows this only qualitatively.

### Gaps
- There is no FiveK expert-style learning experiment. The evaluation is LUT-reproduction fidelity only.

---

## 9. Contrast: diffusion / instruction editors (InstructPix2Pix, Brooks et al., CVPR 2023)

### Takeaway
InstructPix2Pix regenerates pixels with a latent diffusion model trained on 454K synthetic (GPT-3 + Stable Diffusion + Prompt-to-Prompt) edit pairs. It is powerful for semantic and stylistic edits but can make "undesired excessive changes". Its output is a new raster with no edit parameters, it costs about 9 s per image on an A100, and its training cost 25.5 h on 8×A100.

### Cited Findings
- **Data:** a fine-tuned GPT-3 writes instructions and edited captions, and Stable Diffusion with Prompt-to-Prompt generates before/after pairs, giving 454,445 examples. — [Brooks et al., arXiv:2211.09800](https://arxiv.org/abs/2211.09800)
- **Compute:** training was 10,000 steps on 8×40GB A100 over 25.5 hours. Inference with 100 denoising steps takes about 9 s on one A100. — [Brooks et al.](https://arxiv.org/abs/2211.09800)
- **Guidance trade-off:** two classifier-free guidance scales, s_I and s_T, trade consistency with the input image against edit strength. The authors tune them per example. — [Brooks et al.](https://arxiv.org/abs/2211.09800)
- **Failure cases:** it "can make undesired excessive changes to the image" and "can sometimes fail to isolate the specified object". The model is "limited by the visual quality of the generated dataset" and by Stable Diffusion. When style is changed, the paper notes a baseline "struggles to preserve identity". — [Brooks et al.](https://arxiv.org/abs/2211.09800)

### Inferences
- For retouching toward a photographer's look, pixel regeneration risks content drift and texture hallucination. It is also resolution-bound (the paper shows 512–768 px generations) and non-editable. Parametric renderers (curves, LUTs, filters, DNCM) cannot add or remove content by construction, and they are 3–5 orders of magnitude cheaper per image.

### Gaps
- InstructPix2Pix has no FiveK or expert-style benchmark, so no direct quantitative comparison exists.

---

## 10. Distribution-matching losses for unpaired styles with a parametric renderer

### Takeaway
In the verified literature, the dominant unpaired signal for parametric retouchers is adversarial:
- WGAN-GP in Exposure and in unpaired 3D-LUT.
- A relativistic GAN in UEGAN, which is a pixel method.

Each is paired with a content anchor: L2 to the input, VGG-feature fidelity, or an identity loss. Explicit colour-histogram or optimal-transport losses (sliced Wasserstein, CDF matching) exist in colour transfer and low-light work. I found no primary paper in this scope that trains image-adaptive LUTs or curves to a photographer's *set* with a sliced-Wasserstein histogram loss.

### Cited Findings
- **Exposure:** WGAN-GP critic over the whole output distribution, with rewards from its improvement. Evaluation used histogram intersection of luminance, contrast and saturation distributions; this is a metric, not a loss. — [Hu et al.](https://arxiv.org/abs/1709.09602)
- **Unpaired 3D-LUT:** WGAN-GP plus λ·‖G(x) − x‖² plus LUT smoothness and monotonicity regularisers. — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- **UEGAN:** relativistic-GAN quality loss, VGG fidelity loss, and L1 identity loss on in-style inputs. — [Ni et al.](https://arxiv.org/abs/2012.15020)
- **Neural Preset:** avoids distribution losses altogether with self-supervised perturb-and-reconstruct. It also notes that prior colour-transfer metrics use VGG Gram matrices, which it considers to "contain semantic" information not suited to colour-style comparison, and proposes its own measures. — [Ke et al.](https://arxiv.org/abs/2303.13511)
- **Multiscale sliced Wasserstein distance** (He et al., ECCV 2024) is proposed as a perceptual colour-difference measure that "show[s] its promise as a loss function for image and video color transfer tasks". — [He et al., arXiv:2407.10181](https://arxiv.org/abs/2407.10181)
- **Zero-DCE** trains per-image curve estimators with *no* paired or unpaired target data, using non-reference losses (exposure control, colour constancy, spatial consistency, illumination smoothness). It is an example of curve renderers trained purely by statistical losses, but it targets low light, not a style. — [Guo et al. CVPR 2020, arXiv:2001.06826](https://arxiv.org/abs/2001.06826)

### Inferences
- A viable unpaired recipe for a small style set (tens to hundreds of images) looks like this:
  - A parametric renderer: curves, a basis-LUT blend, DNCM, or NeurOp-style operators.
  - A small discriminator on downsampled outputs, *or* a sliced-Wasserstein / CDF loss between the colour and luminance statistics of the output batch and the target-set batch.
  - A content anchor: identity on in-style inputs, plus a mild L2 or perceptual term.
  
  Histogram and SWD losses are deterministic and far more stable than GANs, which suits small data and CPU training. They only match marginal or joint colour statistics, though, not content-conditional behaviour (skin vs sky). A GAN, or conditioning on semantics, is needed for that. This is inference, not a verified result.

### Gaps
- I found no primary paper using sliced-Wasserstein, CDF or Gram losses specifically to train image-adaptive LUTs or curves from an unpaired photographer set with FiveK numbers. The search was brief; this should be treated as "not found", not "does not exist".
- I did not verify the sliced optimal-transport colour transfer literature (Pitié et al.; Bonneel et al.) from primary sources in this pass.

### Cross-method summary matrix (derived from the cited findings in sections 1–10)

| Method | (a) Unpaired training | (b) Multi-style in one model | (c) Few-shot new style | (d) Editable / non-hallucinating output | FiveK headline |
|---|---|---|---|---|---|
| Exposure (2018) | Yes (WGAN-GP + RL) | No (one model per style) | No (retrain; ~370 images suffice) | Yes: 8 named global filters, any resolution | 21.32 dB (Zeng table, unpaired) |
| 3D-LUT unpaired (Zeng) | Yes (WGAN-GP + L2 anchor) | No | No | Global LUT, exportable, no named sliders | 22.86 dB unpaired vs 25.21 paired (480p) |
| DPE / UEGAN / EnlightenGAN | Yes | No | No | No (pixel generators; DPE OOM at full res) | DPE 21.99; UEGAN 22.88 (own table) |
| StarEnhancer | Training paired; new styles need only unpaired examples | Yes (style codes + Dual AdaIN) | Yes (average embedding of n images) | Yes: 15 curves incl. spatial x/y, knot and slider editing, >200 FPS at 4K | 25.29 dB (single-style UPE) |
| PieNet | Training paired; new users select liked images | Yes (preference vectors) | Yes (~10–20 liked images) | No (residual pixel image, 512² input) | 25.28 C; 24.28 mean A–E |
| NeurOp | No (paired) | No | No | Yes: 3 strength sliders, 28K params | 24.32 (5K-Dark) |
| RSFNet | No (paired) | No | No | Yes: region maps + named filters | 25.49 (480p) |
| Neural Preset | Yes (self-supervised, no labels) | Yes (any reference style) | Yes (1 reference image) | Deterministic global colour map (bakeable to a LUT), not sliders; no local edits | N/A (colour-transfer metrics) |
| NILUT / CNILUT | N/A (fits existing LUTs) | Yes (one-hot, blendable) | No | Global, deterministic, not image-adaptive | N/A (>42 dB LUT fidelity) |
| InstructPix2Pix | N/A (synthetic pairs) | Via text | Via text | No (regenerates pixels; excessive changes; ~9 s/A100) | N/A |
