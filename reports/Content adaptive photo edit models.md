# Keep Zeng's basis blend, drop its big LUT

Zeng et al.'s image-adaptive 3D LUT is the right *idea* for this project but not the right *implementation*. Its core trick is a small network that outputs a few blend weights for a shared set of learned "basis" LUTs. That trick is sound, cheap, cannot invent content, exports to `.cube`, and has the best unpaired FiveK score of its generation. But its 33³ lattice and its from-scratch 269K-parameter CNN are the wrong size for tens-to-hundreds of pairs on a CPU. The best-supported choice for this repo is a hybrid built on the existing ResNet-18 + CDF features and MLP head. The renderer becomes a **SepLUT-style cascade**: three per-channel tone curves, then the existing 3×3 matrix, then a residual **9³ 3D LUT built as a weighted blend of a few basis LUTs**, keeping grain and vignette as separate sliders. On top of that sit **StarEnhancer/PieNet-style style codes**: a learned embedding per photographer, plus a code averaged from a few example images for new styles, fed in through FiLM. On the published FiveK protocol, SepLUT matches or beats Zeng with 5–12× fewer parameters (25.42–25.47 dB vs 25.29 dB in the same table) and has published CPU timings. StarEnhancer is the only published system that combines several styles in one model, few-shot new styles from example images, and curve-editable output. Two caveats matter. First, **no paper tests any of these methods at tens-to-hundreds of pairs**. Second, **unpaired training costs about 2.3 dB against paired training** even in Zeng's best case. The staged plan at the end is designed to test this recommendation cheaply before committing to it.

Conventions: every number below comes from a cited primary paper or official repo. Sentences that start with or contain **"Inference:"** are my reasoning, not published results. FiveK PSNRs are comparable only within one protocol. The main protocol is "P1" (Zeng's 480p, 4,500/500 split, expert C), and the others are named where used. Cross-paper differences of about 0.1 dB are noise.

## Zeng et al. blends three learned LUTs with 593K parameters

**How it works.** A 3D LUT is a lattice over RGB space: each lattice point stores an output colour. Any input pixel is mapped by finding its surrounding 8 lattice points and **trilinearly interpolating** between them. With the usual 33 points per axis, a LUT stores 3×33³ ≈ **108K numbers** ([Zeng et al.](https://arxiv.org/abs/2009.14468)). Zeng et al. learn **N = 3 basis LUTs**. A tiny CNN looks at a 256×256 bilinear downsample of the image and outputs 3 scalar weights. It has five conv + LeakyReLU + InstanceNorm blocks, dropout 0.5, and an FC layer, **269K parameters** in all. The image's own LUT is the weighted sum of the bases, and it is applied to the full-resolution image. Because the bases are shared and only the weights change, the edit is **content-adaptive but still global**: every pixel with the same RGB value gets the same output ([Zeng et al.](https://arxiv.org/abs/2009.14468)). Initialisation makes the model start as a no-op: the first basis is the identity, the others are zero, and the FC bias is set so the output ≈ the input. The total is about **593.5K parameters**, of which about 55% is LUT storage ([AdaInt Table 2](https://arxiv.org/abs/2204.13983)).

**Regularisers.** Two regularisers keep the LUT physically sensible. The first is **smoothness**: a squared total variation (TV) penalty between neighbouring lattice cells, plus an L2 penalty on the predicted weights. The second is **monotonicity**: a ReLU penalty whenever an output decreases along any axis. The authors motivate monotonicity partly because "training data may be insufficient to cover the entire color space"; it gives gradient to lattice cells that no training pixel touches. The paired loss is MSE + 1e-4·R_s + 10·R_m. PSNR degrades if λs > 1e-4 and is insensitive to λm ([Zeng et al. Eqs. 11–15](https://arxiv.org/abs/2009.14468)). That coverage problem is real: SepLUT measured that one of Zeng's 33³ LUTs uses only **about 5.5% of its cells** on a typical image ([SepLUT §4.4](https://arxiv.org/abs/2207.08351)).

**Results and speed.** On FiveK expert C at 480p, paired training gives **25.21 dB / SSIM 0.922 / ΔE 7.61**, against HDRNet's 24.32 in the same table. At full resolution it gives 25.10 dB. Independent re-runs land at 25.19–25.29 dB ([Zeng et al. Table 3](https://arxiv.org/abs/2009.14468); [AdaInt](https://arxiv.org/abs/2204.13983); [ICELUT](https://arxiv.org/abs/2403.19238)). The ablation over N is informative for this project:

- A single *fixed* learned LUT with no CNN scores **20.37 dB**. One adaptive LUT scores 23.15, N = 2 scores 24.86, and N = 3 scores 25.21.
- More bases than 3 add almost nothing (25.29 at N = 5).

([Zeng et al. Table 2](https://arxiv.org/abs/2009.14468))

So nearly all the gain comes from image adaptivity, not from spatial processing. On GPU it takes **1.66 ms for a 4K image** on a Titan RTX ([Zeng et al. Table 7](https://arxiv.org/abs/2009.14468)). On CPU it takes **17.35 ms at 480p** on a Xeon 8163 ([SepLUT Table 5](https://arxiv.org/abs/2207.08351)), or 7.15 + 6.71 ms on a Xeon 8260L ([ICELUT Table 6](https://arxiv.org/abs/2403.19238)).

**Unpaired variant.** The MSE loss is replaced by a WGAN-GP critic, whose architecture is the weight-predictor CNN without its normalisation, plus a content anchor λ·‖G(x) − x‖² with λ = 1000. The LUT regularisers stay the same. It was trained on 2,250 inputs and 2,250 expert-C retouches of *different* images. It scores **22.86 dB / ΔE 10.28**: the best of the unpaired methods in the authors' table, but **2.35 dB below its paired self**. The authors say most unpaired rivals show no "obvious advantage over the automatic adjustment of Camera Raw" (21.61 dB). On the harder XYZ→sRGB task, even their own method only ties Camera Raw (21.60 vs 21.61), because the L2 anchor "can hardly hold" when input and target differ greatly ([Zeng et al. Tables 4, 6](https://arxiv.org/abs/2009.14468)).

**Limitations and small-data evidence.** The authors state two limitations. The same LUT applies to every region, so high-dynamic-range scenes lack local contrast. And a LUT cannot denoise, so noisy night shots get noise amplified ([Zeng et al. §4.7](https://arxiv.org/abs/2009.14468)). The smallest training set Zeng tested is **675 HDR+ pairs**, which scored 23.54 dB, below its FiveK score. The authors partly attribute the gap to having fewer pairs. There is no training-set-size ablation and no CPU training time anywhere in this LUT family (lut_family notes §10). The official trilinear op is CUDA-only, but the README points to a `grid_sample` replacement ([Zeng repo](https://github.com/HuiZeng/Image-Adaptive-3DLUT)).

**What this means for the repo.** Inference: Zeng's result explains why the repo's 80-pair model already moves images toward expert C. The repo's head is already an "image-adaptive global" model, which is exactly the regime that captures most of FiveK expert C. Several findings argue against the README's reading that the 500-pair regression is a "capacity ceiling of global primitives":

- Global CSRNet (25.17 dB) beats local HDRNet (24.66) and local DeepLPF (24.73) under P1 ([AdaInt Table 2](https://arxiv.org/abs/2204.13983)).
- Spatially-aware 3D LUTs gain only **+0.26 dB** on FiveK, although they gain +4.7 dB on HDR+ ([Wang et al. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)).

So a global ceiling is unlikely to be what failed at 500 pairs. Two things are more likely. One is the training confounds already listed: an unfrozen BatchNorm (BN) ResNet at batch 4, no ImageNet normalisation, fewer epochs, one seed, and 5 test images. The head's own `BatchNorm1d` layers at batch 4 are a further confound. The other is the renderer's expressiveness. Inference: `generic_renderer.py` applies **one tone curve shared by all three channels** plus an affine matrix. That is weaker than the per-channel curves and non-linear colour maps that every competitive global method uses.

## Eighteen alternatives compared on accuracy, size, conditioning and editability

The table groups methods by family. The "CPU / small-data fit" column is my judgement (inference), based on parameter count, custom-op dependence and any small-data evidence.

| Method | FiveK expert C (protocol) | Params | Speed | Paired / unpaired | Multi-style in one model | Few-shot new style | Spatial / local | Editability, `.cube` | CPU / small-data fit |
|---|---|---|---|---|---|---|---|---|---|
| **Zeng 3D-LUT** ([paper](https://arxiv.org/abs/2009.14468)) | 25.21 (P1); unpaired 22.86 | 593.5K | 1.66 ms 4K GPU; 17 ms 480p CPU | Both | No | No | No | Per-image LUT → `.cube`; no sliders | Medium: 33³ is sparse on small data |
| **AdaInt** ([paper](https://arxiv.org/abs/2204.13983)) | 25.49 (P1) | 619.7K | 1.29 ms GPU | Paired | No | No | No | Non-uniform lattice needs resampling for `.cube` | Low: CUDA-only op |
| **SepLUT** ([paper](https://arxiv.org/abs/2207.08351)) | 25.42 (S) / 25.47 (L) (P1) | 47.2K / 119.8K | ~1.1 ms GPU; 25 ms (16 ms 8-bit) 480p CPU | Paired | No | No | No | 3 curves + small LUT; bakes to one `.cube` | **High**: tiny, CPU/C++ code |
| **CLUT-Net** ([repo](https://github.com/Xian-Bei/CLUT)) | 25.53–25.68 (P1) | ≈290K (estimate; unverified) | 8.7 + 6.7 ms CPU | Paired | No | No | No | Reconstructs standard LUT → `.cube` | Medium: CUDA op |
| **ICELUT** ([paper](https://arxiv.org/abs/2403.19238)) | 25.27 (P1) | 780 KB storage | ~7.7 ms CPU | Paired | No | No | No | LUT-only inference | High for deployment |
| **SA-3DLUT** ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)) | 25.50 (P1, own SSIM calc) | 4.52M | ~4 ms 4K V100 | Paired | No | No | Yes: per-pixel LUT blend | Not one `.cube` | Low: no code, large |
| **4D LUT** ([paper](https://arxiv.org/abs/2209.01749)) | 24.96 (UPE-510px; not P1) | 924K | 5.75 ms GPU | Paired | No | No | Yes: context map | Not one `.cube` | Low |
| **Bilateral-grid LUTs** ([2508.16121](https://arxiv.org/abs/2508.16121)) | 25.66–25.76 (P1) | 160–464K | 1.2–3.6 ms GPU | Paired | No | No | Yes | Not one `.cube` | Medium |
| **CSRNet** ([paper](https://arxiv.org/abs/2009.10390)) | 25.17 (P1); 23.69 (P2) | 36.5K | 3.1 ms 480p, 77 ms 4K GPU | Paired | No, but condition-only fine-tune ≈ scratch | Partial (fine-tune) | No | RGB-only, so bakeable to LUT; latent code, no sliders | High |
| **NeurOp** ([paper](https://arxiv.org/abs/2207.08080)) | 24.32 (P2, vs CSRNet 23.86) | 28K | 4 ms at 500 px | Paired | No | No | No | 3 strength sliders; operators learned | High |
| **RSFNet** ([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouyang_RSFNet_A_White-Box_Image_Retouching_Approach_using_Region-Specific_Color_Filters_ICCV_2023_paper.pdf)) | 25.49 map / 24.31 global (P1) | not found | ~10 ms | Paired | No | No | Yes: region masks | Named filters + masks | Medium |
| **HDRNet** ([paper](https://arxiv.org/abs/1707.02880)) | 24.32–24.66 (P1) | 482K | 2 ms desktop, 14 ms phone CPU | Paired | No | No | Yes: bilateral grid | Grid of affine matrices; not sliders | Medium: overfit reports |
| **DeepLPF** ([paper](https://arxiv.org/abs/2003.13985)) | 24.73 (P1) | 1.7–1.8M | 32 ms 480p GPU | Paired | No | No | Yes: graduated/radial filters | Drawable masks | Low: U-Net |
| **CURL** ([paper](https://arxiv.org/abs/1911.13175)) | 24.04 (P3), 24.20 (P4) | 1.4M | not reported | Paired | No | No | TED backbone edits pixels | 10 curves, but backbone alters pixels | Medium; tested on 90 pairs (RAW task) |
| **StarEnhancer** ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)) | 25.29 multi-style, 25.46 single (UPE full-res) | not found | >200 FPS 4K | Paired training; new styles from unpaired examples | **Yes** | **Yes** (mean of n embeddings) | Partial: x/y curves | Knot-editable curves + β sliders; x/y curves block `.cube` | Medium-high |
| **PieNet** ([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)) | 25.28 C; 24.28 mean A–E | not found | not found | Paired; new users pick 10–20 images | Yes | Yes | Pixel decoder | Residual pixels; not editable | Low |
| **Exposure** ([paper](https://arxiv.org/abs/1709.09602)) | 21.32 unpaired (Zeng table) | ~8.6M | 30 ms GPU | Unpaired (RL + GAN) | No | No (≈370 images suffice) | No | 8 named filters | Low: RL fragile |
| **Neural Preset** ([paper](https://arxiv.org/abs/2303.13511)) | none reported | 5.15M | 52 fps 4K | Self-supervised | Any reference | 1 image | No | Global map, bakeable to LUT | Low: large encoder |
| **NILUT / CNILUT** ([paper](https://arxiv.org/abs/2306.11920)) | N/A (LUT emulation) | 8.7–34K | minutes to fit | Fits existing LUTs | Yes (one-hot, blendable) | No | No | Evaluates to `.cube` | High, but not image-adaptive |

Five patterns matter for this project.

First, **on paired FiveK the LUT and global-parametric families have converged within about 0.5 dB** (25.2–25.8 dB under P1). Parameter efficiency and editability are now the real differentiators, not accuracy.

Second, **spatial methods buy little on FiveK expert C**: SA-3DLUT gains 0.26 dB and bilateral-grid LUTs about 0.3–0.5 dB. RSFNet is the exception, where masks help by about 1.2 dB over its own global variant ([RSFNet](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouyang_RSFNet_A_White-Box_Image_Retouching_Approach_using_Region-Specific_Color_Filters_ICCV_2023_paper.pdf)). So local ability is worth something, but it is the second lever, not the first.

Third, **only the StarEnhancer and PieNet lineage addresses multi-style conditioning and few-shot styles**. Of the two, only StarEnhancer does it with an editable renderer. It trains on all FiveK styles, holds out 8 unseen styles, and reports that a code averaged from a few examples "can outperform most of the enhancers fine-tuned on the train sets of these styles." That comparison is reported only as a figure ([Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)).

Fourth, **every unpaired method is far below its paired counterpart**. The parametric ones (Zeng unpaired, Exposure) are the only ones that do not distort content or run out of memory at full resolution ([Zeng et al.](https://arxiv.org/abs/2009.14468); [Hu et al.](https://arxiv.org/abs/1709.09602)).

Fifth, **no row has evidence below about 675 training pairs** for FiveK-style retouching. CURL's 90-pair experiment was a RAW-to-RGB task ([CURL](https://arxiv.org/abs/1911.13175)).

## Recommendation: a SepLUT renderer under a style-conditioned version of the existing head

### Architecture mapped onto the current code

The recommendation reuses what the repo already does well and replaces the weak link.

**Features.** Keep the ResNet-18 GAP + per-channel CDF feature. CSRNet's ablation found histogram conditioning the strongest global prior ([CSRNet](https://arxiv.org/abs/2009.10390)). Change two things: **freeze the backbone and cache features**, and add ImageNet normalisation. Inference: with cached features, the head trains on CPU in seconds per epoch, even on all 4,500 FiveK pairs.

**Renderer.** Replace the renderer with this cascade:

1. **Three per-channel monotone curves** (K = 9–17 knots each), replacing the single shared curve.
2. The existing **3×3 matrix + bias**, kept because white balance and saturation read naturally as sliders.
3. A **residual 9³ 3D LUT formed as Σ wₙ·Bₙ over N ≈ 4–8 shared basis LUTs**. The bases are initialised to zero residual and regularised with Zeng's TV (1e-4) and monotonicity (10) terms.
4. **Grain and vignette** as separate spatial sliders.

**Head output.** The head now outputs curve knots, matrix, bias, N basis weights, grain and vignette, roughly 40–60 numbers. The learned bases add only N×3×729 ≈ 9–17K parameters.

This is SepLUT's decomposition. SepLUT shows the per-channel curves spread colours over the lattice so a 9³ or 17³ LUT suffices: 3D-LUT-only 33³/17³/9³ scores 25.27/25.24/25.21 dB, and adding 3×1D curves lifts a 9³ model to 25.42 dB at 47.2K parameters ([SepLUT §4.3, Table 4](https://arxiv.org/abs/2207.08351)). SepLUT also shows its LUT generator is mathematically Zeng's basis blend ([SepLUT §3.5](https://arxiv.org/abs/2207.08351)), so the previously suggested Zeng design survives as the third stage, just smaller.

**Head size.** The current head (1280→512→256→P, about 790K parameters with BN at batch 4) is larger than all of SepLUT-L. Inference: shrink it to 1280→128→P or regularise it hard, and replace BatchNorm with LayerNorm or none.

### Conditioning on photographers and styles

A style code **s** (32–64-D) conditions the head via **FiLM** (per-layer scale and shift from s). StarEnhancer's Dual-AdaIN is the published analogue, and it found ordinary normalisation layers hurt ([Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)). The code comes from one of two sources:

- **Known photographers or styles:** a learned embedding table, e.g. FiveK experts A–E plus the synthetic Fujifilm, cyberpunk and tilt-shift looks.
- **New styles:** a small style encoder that maps the *same cached features* of n example "after" images to a code, then averages them. This follows StarEnhancer's and PieNet's mean-of-embeddings recipe. It is trained on style subsets so it tolerates small n ([Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf); [Kim et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf)).

A cheaper fallback exists for a new photographer with a few pairs. **Fine-tune only the style embedding** (or the FiLM layers) with everything else frozen. This mirrors CSRNet's finding that condition-network-only fine-tuning matches training from scratch on experts A, B, D and E ([CSRNet](https://arxiv.org/abs/2009.10390)).

Inference: the basis LUTs are shared across all styles, so they become a common colour vocabulary. Each style then only chooses curves, a matrix and blend weights. This is the property most likely to make tens of pairs per style enough, and it is untested.

### Unpaired training for style sets

For example-only styles such as cyberpunk or film, the literature supports an adversarial signal with a content anchor. The anchor can be Zeng's L2-to-input, or UEGAN's gentler VGG fidelity plus an identity loss on inputs that are already in style ([Zeng et al.](https://arxiv.org/abs/2009.14468); [Ni et al.](https://arxiv.org/abs/2012.15020)).

Inference: at tens-to-hundreds of images on a CPU, start with deterministic losses and add a small critic only if they fall short. The deterministic losses are:

1. A **batch-level CDF / sliced-Wasserstein colour-distribution loss**. The repo already has a CDF loss, and multiscale SWD has been proposed as a colour-transfer loss ([He et al.](https://arxiv.org/abs/2407.10181)).
2. The identity and fidelity anchors.
3. **Distort-and-recover pseudo-pairs**: randomly perturb in-style images with global edits and train the model to restore them ([D&R](https://arxiv.org/abs/1804.04450)). Neural Preset uses the same self-supervised perturb-and-reconstruct idea at scale ([Ke et al.](https://arxiv.org/abs/2303.13511)).

Histogram matching is deterministic, but it only matches colour statistics. It cannot learn "treat skies differently from faces"; that content-conditional behaviour needs pairs or a critic (inference). Note also that the existing synthetic styles are content-independent by construction. For them a content-adaptive model can at best rediscover a fixed LUT, and N = 1-style fitting (or NILUT-style emulation) is enough ([NILUT](https://arxiv.org/abs/2306.11920)).

### Why this beats each main competitor under these constraints

**Pure Zeng.** SepLUT beats it by 0.13–0.18 dB in the same P1 table with 5–12× fewer parameters ([SepLUT Table 6](https://arxiv.org/abs/2207.08351)). Its first stage is human-readable curves, and the repo can drop Zeng's 269K from-scratch CNN because it already has better global features. Inference: a 9³ lattice has 729 cells against 35,937, so tens of pairs cover it far better. Zeng's own 5.5% cell-utilisation figure is the warning sign.

**AdaInt.** AdaInt's +0.2 dB comes from a non-uniform lattice that needs a CUDA op and cannot be written as a plain `.cube` ([AdaInt](https://arxiv.org/abs/2204.13983)). Per-channel curves before the LUT provide the same "put resolution where it's needed" effect in an exportable form (inference, consistent with SepLUT's analysis).

**CLUT-Net and ICELUT.** Both are compression or deployment optimisations of the same global map. They are useful later, but they add no conditioning or editability and depend on custom kernels ([CLUT repo](https://github.com/Xian-Bei/CLUT); [ICELUT](https://arxiv.org/abs/2403.19238)).

**CSRNet and NeurOp.** These are the strongest small-model rivals, and the honest runner-up. At 28–36K parameters they are even smaller, and CSRNet's condition-only fine-tuning is the best evidence for few-parameter style adaptation. But CSRNet's edit is a latent modulation vector that is not slider-editable, and it runs 77 ms at 4K because its per-pixel MLP runs at full resolution ([AdaInt](https://arxiv.org/abs/2204.13983)). NeurOp's three learned operators are sliders, but not curves or LUTs, and its reported numbers use a different protocol (P2) ([NeurOp](https://arxiv.org/abs/2207.08080)). If editability were not a requirement, a style-conditioned CSRNet would be an equally good bet.

**StarEnhancer as a whole.** StarEnhancer is the closest match to the goal, and the recommendation borrows its conditioning wholesale. Its renderer is a real competitor: 15 input-to-output curves including spatial x/y curves, scoring 25.46 dB single-style ([Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)). Two things favour the SepLUT cascade instead. The x/y curves prevent `.cube` export. And curves summed per channel cannot represent hue-selective shifts (e.g. teal shadows only in blues) as directly as a small 3D LUT (inference). StarEnhancer's x/y curves are a good upgrade path if local gradients prove necessary.

**Spatial methods** (SA-3DLUT, 4D LUT, bilateral-grid LUTs, HDRNet, DeepLPF). They give up to about 0.5 dB on FiveK, at the cost of `.cube` export, 0.5–4.5M parameters, and, for SA-3DLUT, any public code. HDRNet has a documented purple-sky overfitting case ([Zeng et al.](https://arxiv.org/abs/2009.14468)).

**Pixel generators and diffusion** (DPE, UEGAN, PieNet's decoder, InstructPix2Pix). They violate the no-hallucination constraint by construction, and InstructPix2Pix "can make undesired excessive changes" ([Brooks et al.](https://arxiv.org/abs/2211.09800)).

### Honest weaknesses and missing evidence

The recommendation is a composition of methods that each have evidence, but **the combination itself is untested**:

- **Unpaired training with an image-adaptive LUT** has only been shown with a GAN, not with SWD or CDF losses.
- **Style-code conditioning** has only been shown with curves.
- **Small data is unmeasured.** No paper measures accuracy against training-set size between 25 and 500 pairs for any method here.
- **Multi-style codes are noisy for subtle styles.** StarEnhancer's style encoder identifies expert styles poorly (Recall@1 from **24.6%** for expert A), so codes averaged from few images of subtle photographers will be noisy ([Song et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf)).
- **Unpaired styles will be weaker.** They should be expected to land nearer Zeng's 22.9 dB than 25 dB in fidelity terms.
- **The 3D LUT stage is not slider-editable.** Users get curves, matrix sliders and a LUT-strength blend; a Lightroom-style HSL panel would need an extra hue-vs-saturation curve as in CURL ([CURL](https://arxiv.org/abs/1911.13175)).
- **The bases are per-model, not per-style.** A basis learned on FiveK may not span cyberpunk's colour moves.
- **CPU runtime is slightly worse unquantised.** SepLUT's CPU runtime was 25 ms vs Zeng's 17 ms at 480p, and only 8-bit quantisation equalises them ([SepLUT Table 5](https://arxiv.org/abs/2207.08351)). Irrelevant for training, minor for inference.

## Five cheap experiments decide the design before any rewrite

Each stage has a decision gate. All stages use cached frozen features, 3 seeds, and at least 100 held-out FiveK images reported as PSNR / ΔE at 480p, so results are comparable to P1.

| Stage | Experiment | Decision gate |
|---|---|---|
| 0. Remove confounds | Re-run 80 vs 500 pairs with frozen, normalised backbone; drop or replace head BN; equal epochs; add a 25/50/100/250/500/4,500 learning curve | If 500 ≥ 80 once fixed, the "global ceiling" story is retired and the learning curve becomes the small-data baseline |
| 1. Renderer ablation | Same head, swap renderers: current (shared curve + matrix) → 3 per-channel curves + matrix → + 9³ residual basis LUT (N = 4, 8) → Zeng 33³ ×3 | Keep the 3D LUT stage only if it beats curves + matrix by > 0.3 dB at ≤ 250 pairs; compare to a CSRNet baseline as the small-model yardstick |
| 2. Multi-photographer conditioning | Train one FiLM-conditioned model on experts A–D; hold out E; compare (a) averaged style code from n = 5/10/20 "after" images, (b) embedding-only fine-tune on n pairs, (c) separate per-expert models | Conditioning is justified if one shared model ≥ separate models at equal data per expert |
| 3. Unpaired | Use expert-C retouches of *disjoint* images as the unpaired set (Zeng's protocol, scaled down), so PSNR is measurable; compare CDF/SWD + identity, + distort-and-recover pseudo-pairs, + small WGAN-GP critic | Adopt the simplest loss within ~0.5 dB of the best; then apply to cyberpunk/film sets with a small preference test |
| 4. Export and editing | Bake curves∘matrix∘LUT to a 33³ `.cube`; check PyTorch-vs-`.cube` PSNR (target > 45 dB); open in Resolve; expose curves, matrix sliders, LUT strength | Ship if the exported render is visually identical and edits behave predictably |

Stage 0 is the cheapest and most informative. Inference: if the learning curve rises monotonically once the confounds are removed, most of the remaining plan is incremental. If it still falls, the problem is optimisation or labels, and no renderer choice will fix it.

## Conclusion

The literature has changed which question matters. Zeng et al. showed that **image adaptivity, not spatial processing, carries most of the FiveK gain**: a fixed LUT scores 20.4 dB, an adaptive global one 25.2 dB, and spatial extensions add a few tenths. The repo's head already sits in the right regime. What it lacks is a renderer with per-channel curves and a small non-linear colour stage, and a clean experiment showing whether more data helps. The Zeng-style basis blend earns its place as a small third stage and a shared colour vocabulary across styles. It does not justify a 33³ lattice or a new CNN.

The unresolved risk is data, not architecture. Every number above comes from 675–4,500 pairs, and unpaired results trail paired ones by more than 2 dB. The first deliverable should therefore be the learning curve from Stage 0, not a new model. Its shape will show whether tens of pairs per photographer is a realistic target or whether style-code conditioning over many pooled styles is what makes the small-data goal achievable.
