# Parametric / filter-based learned photo enhancement (non-LUT): HDRNet, CSRNet, DeepLPF, CURL, Distort-and-Recover, and relatives

Verification method: the primary PDFs (arXiv) were downloaded and their text extracted; the numbers below come from those tables, not from summaries. **FiveK numbers are not comparable across papers unless the protocol matches.** In practice there are four protocols:

| Protocol label (used below) | Input rendition | Split | Resolution | Used by |
|---|---|---|---|---|
| **P1 "480p / Zeng"** | sRGB, "zeroed with C's" style (Zeng) | 4,500 train / 500 test, expert C | short side 480 px (also tested at full ~12 MP / "4K") | Zeng 3D-LUT (TPAMI 2020), AdaInt (CVPR 2022), SepLUT, RSFNet |
| **P2 "5K-Dark / CSRNet"** | Hu et al. (White-Box/Exposure) pre-processing, 16-bit ProPhoto TIFF | random 4,500 train / 500 test, expert C | long edge 500 px | CSRNet (ECCV 2020), NeurOp (ECCV 2022) |
| **P3 "5K-DPE"** | DPE pre-processing | 2,250 train (500 of them held out for val) / 500 test, expert C | long edge 500 px | DeepLPF, CURL |
| **P4 "5K-UPE"** | DeepUPE pre-processing | 4,500 train / 500 test (DeepUPE's test set), expert C | full resolution, 6–25 MP | DeepLPF, CURL (competitor numbers copied from DeepUPE) |
| (P5 "Random250") | Hwang et al. | 4,750 train / 250 test, expert C | — | HDRNet paper, Distort-and-Recover paper (L2-Lab only) |

"ΔE" / "L2 error in Lab" means the mean Euclidean distance in CIELAB (ΔE*ab, CIE76) in all of these papers.

---

## 1. HDRNet (Gharbi, Chen, Barron, Hasinoff, Durand, SIGGRAPH 2017 / ACM TOG 36(4)) "Deep Bilateral Learning for Real-Time Image Enhancement"

### Takeaway
HDRNet predicts a **spatially varying** low-resolution bilateral grid of 3×4 affine color transforms (16×16 spatial × 8 intensity bins × 12 coefficients). A learned full-resolution grayscale guidance map "slices" the grid back to full resolution. The model has about 482K parameters and runs in real time on a phone. Under the shared 480p protocol it scores about 24.3–24.7 dB on FiveK. The output is interpretable as local affine color matrices, but it cannot be meaningfully exposed as sliders.

### Cited Findings
- Mechanism: a low-resolution stream (input downsampled to 256×256) runs conv layers split into local and global paths. These are fused and linearly predicted into "a 16×16 map with 96 channels", which "can be viewed as a 16×16×8 bilateral grid, where each grid cell contains 12 numbers, one for each coefficient of a 3×4 affine color transform". A full-resolution stream learns a grayscale guidance map g, and a new "slicing" node does data-dependent trilinear lookup into the grid. The per-pixel affine transforms are applied to the full-res input. The loss is computed on the full-res output, not the coefficients. — [HDRNet arXiv 1707.02880](https://arxiv.org/abs/1707.02880)
- Guidance map: a pointwise learned function, a 3×3 color matrix M followed by per-channel piecewise-linear curves and a sum. "M is initialized to the identity". — [HDRNet](https://arxiv.org/abs/1707.02880)
- Speed from the paper: on a Google Pixel phone, 1920×1080 viewfinder at 40–50 Hz. "Overall throughput is under 20 ms, with 14 ms spent on inference (CPU)", plus 18 ms GPU render, overlapped. Desktop: "2 ms" (vs seconds for competitors) and "61 ms to process a 12-megapixel image". The figure-1 teaser reports "61 ms, PSNR = 28.4 dB" on a 12 MP HDR+ task. — [HDRNet](https://arxiv.org/abs/1707.02880)
- FiveK in the original paper uses only the L2-Lab metric on the Hwang splits. Expert C Random250: **7.8** Lab (L-only 5.5), vs Yan 2016 9.85. HighVar50: 7.1. Other experts on Random250: A 11.7, B 7.4, D 10.0, E 8.8. No PSNR/SSIM is reported for FiveK. — [HDRNet](https://arxiv.org/abs/1707.02880)
- Parameter count: **482,080** per CSRNet's table and **483.1K** per AdaInt's table. The original paper gives no total. — [CSRNet arXiv 2009.10390](https://arxiv.org/abs/2009.10390); [AdaInt arXiv 2204.13983](https://arxiv.org/abs/2204.13983)
- FiveK scores under third-party protocols:
  - P1, Zeng TPAMI (retrained by Zeng): 480p **24.32 / 0.912 / ΔE 8.49**; full res 24.03 / 0.919 / 8.68. — [Zeng 3D-LUT arXiv 2009.14468](https://arxiv.org/abs/2009.14468)
  - P1, AdaInt (retrained, V100): 480p **24.66 / 0.915 / 8.06, 3.49 ms**; 4K 24.52 / 0.921 / 8.20, 56.07 ms. — [AdaInt](https://arxiv.org/abs/2204.13983)
  - P2, CSRNet (retrained): **22.65 / 0.880 / L2-Lab 11.64**. — [CSRNet](https://arxiv.org/abs/2009.10390)
  - P2, NeurOp: 22.65 / 0.880 / 11.83 (same PSNR/SSIM as CSRNet, ΔE differs). — [NeurOp arXiv 2207.08080](https://arxiv.org/abs/2207.08080)
  - P4, DeepUPE-protocol numbers copied by DeepLPF/CURL: 21.96 / 0.866. — [DeepLPF arXiv 2003.13985](https://arxiv.org/abs/2003.13985)
  - The spread is 21.96–24.66 dB for the same model, depending on protocol.
- Tone-mapping variant (16-bit XYZ input, 480p): 24.14 / 0.913 / 8.65 (Zeng) and 24.52 / 0.915 / 8.14 (AdaInt). — [Zeng](https://arxiv.org/abs/2009.14468); [AdaInt](https://arxiv.org/abs/2204.13983)
- Limitations the authors state:
  - Without the global path, "the network can make erroneous local decisions that lead to artifacts as exemplified by the large-scale variations in the sky".
  - The model "fails when the image operator strongly violates our modeling assumptions", e.g. dehazing, matting and colorization.
  - The affine relation "breaks down at larger scales (like a grid cell)".
  - — [HDRNet](https://arxiv.org/abs/1707.02880)
- Third-party weakness: some sky areas enhanced by HDRNet "are biased to purple". Zeng suggests "the HDRNet model overfits the training data". — [Zeng](https://arxiv.org/abs/2009.14468)

### Inferences
- The model is spatially varying but low-frequency: 16×16 spatial cells, with edges recovered through the guidance map. It can do local tone mapping and is edge-aware, but edits inside a cell must be affine in color.
- Interpretability: each pixel gets a 3×4 color matrix, which you can visualise as coefficient maps (the paper's Fig. 8). There is no compact global slider vector. Editing would mean editing about 24.6K grid coefficients, so it is not slider-friendly.
- About 480K parameters and a fixed 256×256 low-res input make CPU training at low resolution plausible, but no paper reports CPU training.

### Gaps
- "Halo artifacts" in HDRNet: the task framed these as known, but I found no primary-source statement about halos. The HDRNet paper mentions sky artifacts without the global path. Zeng attributes halos to DPE and a purple sky cast to HDRNet. Treat HDRNet halos as unverified folklore unless a specific source is found.
- No published HDRNet results with tens to hundreds of training pairs.

---

## 2. CSRNet (He, Liu, Qiao, Dong, ECCV 2020) "Conditional Sequential Modulation for Efficient Global Image Retouching"

### Takeaway
CSRNet is a pixel-independent 3-layer 1×1-conv MLP (the "base network"). Its features are modulated channel-wise (scale and shift, called GFM) by a 32-D condition vector from a tiny strided-conv condition network. It has **36,489 parameters** and is global only. It was state of the art in 2020 on P2 with 23.69 dB. Under P1 it scores 25.17 dB at 480p, and it was the strongest non-LUT model in that table until NeurOp and RSFNet.

### Cited Findings
- Architecture:
  - Base network: 3 conv layers, 64 channels, kernel 1×1, i.e. a per-pixel MLP.
  - Condition network: 3 conv layers with 32 channels (first kernel 7×7, then 3×3, all stride 2), followed by global average pooling to a 32-D condition vector.
  - 6 FC layers produce 3 scale and 3 shift vectors for GFM.
  - Training: L1 loss, batch size 1, lr 1e-4 halved every 1e5 iterations, 6e5 iterations, "only 5 hours" on a 2080Ti.
  - — [CSRNet](https://arxiv.org/abs/2009.10390)
- The authors interpret the base network as "color decomposition". "The condition network generates editing parameters", and "photo retouching consists of only pixel-wise operations". Strength control is done by blending the output with the input. — [CSRNet](https://arxiv.org/abs/2009.10390)
- P2 (random 4,500/500, expert C, long edge 500 px, Hu et al. pre-processing) Table 1, PSNR / SSIM / L2-Lab / params:
  - White-Box: 18.59 / 0.797 / 13.24 / 8,561,762
  - Distort-and-Recover: 19.54 / 0.800 / 12.91 / **259,263,320**
  - HDRNet: 22.65 / 0.880 / 11.64 / 482,080
  - DUPE: 20.22 / 0.829 / 13.38 / 998,816
  - Pix2Pix: 22.05 / 0.788 / 11.88
  - **CSRNet: 23.69 / 0.895 / 10.86 / 36,489**
  - On DPE's input version: DPE 23.76 / 0.881 / 10.50 / 3.3M vs CSRNet 24.23 / 0.900 / 10.29.
  - HDRNet, D&R and Pix2Pix were retrained by the CSRNet authors. White-Box, DUPE and DPE use released models.
  - — [CSRNet](https://arxiv.org/abs/2009.10390)
- Ablation: global priors fed to the condition network rank "histograms > average intensity > brightness". Histogram (768-D) conditioning gives 22.90 dB with 206K params. — [CSRNet](https://arxiv.org/abs/2009.10390)
- Style transfer by **fine-tuning only the condition network** (base fixed) roughly matches training from scratch. PSNR fine-tune vs scratch: A 22.29/22.06, B 25.61/25.52, D 23.06/23.04, E 23.95/23.81. — [CSRNet](https://arxiv.org/abs/2009.10390)
- P1 (AdaInt, V100): **36.4K params, 480p 25.17 / 0.924 / 7.75, 3.09 ms**; 4K 24.82 / 0.926 / 7.94, **77.10 ms**. This is slower than HDRNet at 4K because it runs a per-pixel MLP at full resolution. — [AdaInt](https://arxiv.org/abs/2204.13983)
- Tone mapping (480p): 25.19 / 0.921 / 7.63. — [AdaInt](https://arxiv.org/abs/2204.13983)
- Weaknesses:
  - NeurOp reports CSRNet "performs less satisfactory on PPR10K", with PSNR about 2 dB lower on PPR10K-a. — [NeurOp](https://arxiv.org/abs/2207.08080)
  - Its controllability (linear interpolation) "cannot faithfully reproduce sophisticated and highly non-linear color transformations" and gives only one degree of freedom. — [NeurOp](https://arxiv.org/abs/2207.08080)
  - Under RSFNet's "480p zeroed as shot" input CSRNet drops to 24.24 / 0.910 / 9.70. At full resolution it scores 23.04 dB and takes 80.6 ms. — [RSFNet arXiv 2303.08682](https://arxiv.org/abs/2303.08682)

### Inferences
- Its structure is close to the repo's design: a global image descriptor conditions a pixelwise color mapping. The difference is that CSRNet's mapping is a learned MLP (a black box), whereas the repo's is a hand-designed renderer. The modulation vector is not human-readable.
- 36K parameters and fine-tuning only the condition network make it the best candidate among these for small data and CPU training. No paper demonstrates this below about 4,500 pairs.

### Gaps
- No reported results with fewer than about 2,000 training pairs. No CPU timings.

---

## 3. DeepLPF (Moran, Marza, McDonagh, Parisot, Slabaugh, CVPR 2020) "Deep Local Parametric Filters for Image Enhancement"

### Takeaway
A U-Net backbone feeds small regression heads that predict parameters for three **spatially local, parametric** filters: a per-channel cubic polynomial in (x, y, intensity), graduated (linear-gradient) filters and elliptical (radial) filters. The filters are exactly the kind of tools Lightroom exposes, so each is drawable and editable. The full model has about 1.7–1.8M parameters (about 452K in the filter blocks) and is relatively slow (32 ms at 480p on a V100). Under P1 it scores 24.73 dB.

### Cited Findings
- Filter parameterisation (Table 1 of the paper):
  - Graduated: G=8 parameters, defined by three parallel lines, offsets, a scale s_g and an inversion flag.
  - Elliptical: E=8 parameters.
  - Cubic-10: 30 parameters (10 per channel). Cubic-20: 60 parameters.
  - Cubic filters depend on spatial position and intensity.
  - Multiple instances of the same filter type are fused by elementwise product. 3 graduated and 3 elliptical filters are used per channel.
  - — [DeepLPF arXiv 2003.13985](https://arxiv.org/abs/2003.13985)
- Pipeline: the U-Net output feeds the polynomial filter first. Graduated and elliptical maps are then predicted in parallel and fused. Loss: L1 in Lab plus MS-SSIM on L. — [DeepLPF](https://arxiv.org/abs/2003.13985)
- P3 (DPE split, 2,250/500, long edge 500 px), PSNR / SSIM / LPIPS / weights:
  - U-Net alone: 21.57 / 0.843 / 0.601 / 1.3M
  - +Cubic-20: 23.44
  - **Full model: 23.93 / 0.903 / 0.582 / 1.8M**
  - DPE: 23.80 / 0.900 / 0.587 / 3.3M
  - "Only ~1/4 of model parameters (i.e. around 452k) … are attributed to filter blocks."
  - — [DeepLPF](https://arxiv.org/abs/2003.13985)
- P4 (UPE split, full resolution 6–25 MP): **DeepLPF 24.48 / 0.887 / LPIPS 0.103, 800K weights**. Competitor numbers copied from DeepUPE: DeepUPE 23.04 / 0.893, HDRNet 21.96 / 0.866, DPE 22.15 / 0.850, D&R 20.97 / 0.841, White-Box 18.57 / 0.701. — [DeepLPF](https://arxiv.org/abs/2003.13985)
- P1 (AdaInt): **1.7M params, 480p 24.73 / 0.916 / 7.99, 32.12 ms**. 4K was not available: "insufficient GPU memory". — [AdaInt](https://arxiv.org/abs/2204.13983)
- NeurOp on 5K-Lite: 23.63 / 0.875 / 10.55, 1,769,347 params. — [NeurOp](https://arxiv.org/abs/2207.08080)
- RSFNet "zeroed as shot": 480p 23.38 / 0.880 / 10.03 at 44.25 ms; full resolution 23.40 / 0.863 at **1133.90 ms**. — [RSFNet](https://arxiv.org/abs/2303.08682)
- Authors' future work: "automatic estimation of the optimal sequence of filter application". — [DeepLPF](https://arxiv.org/abs/2003.13985)

### Inferences
- It is the most "slider/overlay-editable" of the spatially varying methods. Graduated and radial filters map one-to-one onto Lightroom masks. The cubic filter is a smooth global-ish spatial/intensity polynomial and is harder to expose as a slider.
- The heavy U-Net backbone and full-resolution filter evaluation are its weaknesses: memory at 4K, and the slowest runtime in these tables.

### Gaps
- No small-data results. The official code is at github.com/sjmoran/DeepLPF (not re-verified here).

---

## 4. CURL (Moran, McDonagh, Slabaugh, ICPR 2020, published 2021) "CURL: Neural Curve Layers for Global Image Enhancement"

### Takeaway
A TED (Transformed Encoder-Decoder) backbone plus a "CURL block" that regresses the knots of **global piecewise-linear scaling curves**, applied in sequence:
- Lab: L, a, b (3 curves)
- RGB: R, G, B (3 curves)
- HSV: H(H), S(S), V(V) and **S(H)**, i.e. hue-dependent saturation (4 curves)

That is 10 global curves in total. The curves are directly displayable as editable curve widgets. The model has about 1.4M parameters: P3 24.04 dB, P4 24.20 dB. Note that the TED backbone also changes pixels, so the curves alone do not fully explain the output.

### Cited Findings
- The curve is S(x) = k0 + Σ (k_{m+1} − k_m) δ(Mx − m), piecewise linear with M knots. "The neural curve outputs scale factors", i.e. the pixel is multiplied by the curve value. That allows cross-channel maps like saturation as a function of hue. — [CURL arXiv 1911.13175](https://arxiv.org/abs/1911.13175)
- Order Lab→RGB→HSV was the best of all six permutations: 27.09 dB vs 25.32–26.83 on the Samsung S7 data. All three color spaces beat any single one (27.09 vs RGB-only 26.74, Lab-only 26.98, HSV-only 25.88). A curvature regularizer L_reg is important "due to its role in constraining the flexibility of the neural retouching curves". — [CURL](https://arxiv.org/abs/1911.13175)
- P3 (DPE, 2,250/500, 500 px): **TED+CURL 24.04 / 0.900 / LPIPS 0.583, 1.4M params** vs DPE 23.80 / 0.900 / 0.587 / 3.3M. — [CURL](https://arxiv.org/abs/1911.13175)
- P4 (UPE, full resolution): **24.20 / 0.880 / 0.108, 1.4M** vs DeepUPE 23.04 / 0.893 / 0.158 / 1.0M and HDRNet 21.96 / 0.866 (copied). — [CURL](https://arxiv.org/abs/1911.13175)
- The Samsung S7 RAW→RGB experiment uses only **90 train / 10 val / 10 test** images: TED+CURL 27.04 dB vs TED alone 26.56. — [CURL](https://arxiv.org/abs/1911.13175)
- The paper claims "human-interpretable" curve adjustment. — [CURL](https://arxiv.org/abs/1911.13175)

### Inferences
- CURL is the closest published analogue to the repo's approach: an image network regresses global tone/color curve knots, the curves are applied by a differentiable renderer, and there is a regularizer on curve smoothness.
- The key difference is that CURL keeps a pixel-level encoder-decoder (TED) in front of the curves, so it is not purely global and the curves alone do not reproduce the edit.
- The 90-image S7 training set is the only evidence among these papers of training on around 100 pairs, although that is a RAW→RGB task, not FiveK retouching.

### Gaps
- Knot count M: not found in the extracted text; it may be in the supplement. Not verified.
- No runtime reported. No P1 (480p/4,500) or P2 numbers found from any paper.

---

## 5. Distort-and-Recover (Park, Lee, Yoo, Kweon, CVPR 2018) "Distort-and-Recover: Color Enhancement using Deep Reinforcement Learning"

### Takeaway
A Double-DQN agent chooses from **12 discrete global actions**, each a ±5% change to contrast, saturation, brightness or one of 6 white-balance channel pairs, until it chooses "stop". Its state is a VGG-16 fc6 feature (4096-D) plus a 20×20×20 Lab histogram. It can train from **retouched images only**, by randomly distorting them to create pseudo-pairs. The output is a fully interpretable action sequence, but accuracy is the lowest of this group (about 19.5–22 dB) and the agent is huge (259M params, mostly FC layers).

### Cited Findings
- Actions: "12 actions to adjust contrast, saturation, brightness, and white-balance. Each action increases or decreases the value by 5%". The agent has 4 FC layers (4096, 4096, 512, 12). The reward is the decrease in Lab L2 distance to the target. Training takes "at least 12 hours" on a GTX 1080. — [D&R arXiv 1804.04450](https://arxiv.org/abs/1804.04450)
- Distort-and-recover scheme: reference images are distorted by random global operations. These operations deliberately differ from the agent's action set: highlight/shadow brightness, contrast and saturation, plus C/M/Y/R/G/B adjustments. The distortion keeps Lab L2 at 10–20. Suggested sources of retouched images are Flickr, stock sites and AVA. — [D&R](https://arxiv.org/abs/1804.04450)
- Paper results (Random250, 4,750 train, expert C), L2-Lab / SSIM:
  - Paired: **10.99 / 0.905** vs Pix2Pix 10.49 / 0.857 and Yan 2016 9.85.
  - Distort-and-recover only (expert-C images as references): **12.15 / 0.910** vs Pix2Pix 13.59–14.46.
  - Input baseline: 17.07.
  - Feature ablation: VGG + Lab histogram 10.99; Lab histogram only 12.30.
  - — [D&R](https://arxiv.org/abs/1804.04450)
- The authors acknowledge the method "is mainly limited by the predefined actions". — [D&R](https://arxiv.org/abs/1804.04450)
- Third-party numbers:
  - P2 (CSRNet): 19.54 / 0.800 / 12.91, **259,263,320 params**. — [CSRNet](https://arxiv.org/abs/2009.10390)
  - P1 (Zeng, "Dis-Rec"): 480p 21.98 / 0.856 / 10.42; original resolution 21.81 / 0.862 / 10.60. Zeng notes "unstability of reinforcement learning". — [Zeng](https://arxiv.org/abs/2009.14468)
  - P4: 20.97 / 0.841. — [DeepLPF](https://arxiv.org/abs/2003.13985)
  - CSRNet observes that D&R "tends to generate over-exposure output". — [CSRNet](https://arxiv.org/abs/2009.10390)

### Inferences
- Its best idea for this repo is the self-supervised pseudo-pair generation (distort good images, learn to recover them), not the DQN itself. The repo's global parametric renderer could use exactly this augmentation.
- Most of the 259M parameters come from VGG-sized FC layers on 4096-D features. A small MLP head on frozen features, as in the repo, is the lightweight equivalent.

### Gaps
- No small-data (tens to hundreds of images) experiment reported.

---

## 6. Close relatives (brief)

### Takeaway
Later "white-box" or parametric methods (NeurOp 2022 and RSFNet 2023) now beat CSRNet with explicit, controllable operators. Zero-DCE shows that global curve estimation can be trained with no references at all.

### Cited Findings
- **NeurOp** (Wang et al., ECCV 2022):
  - Sequential learned "neural color operators" (exposure-, black-level-, vibrance-like), each controlled by one scalar strength predicted from global statistics (CSRNet-style). 28,108 params.
  - P2 5K-Dark: **24.32 / 0.907 / 10.10** vs CSRNet 23.86 / 0.897 / 10.57 in their re-run.
  - 5K-Lite: 25.09 / 0.911 / 9.93.
  - Runtime 4 ms at 500×333 and 19 ms at 1 MP.
  - Training: about 2 h initialization plus about 9 h on FiveK, on a 2080Ti.
  - — [NeurOp](https://arxiv.org/abs/2207.08080)
- **RSFNet** (Ouyang et al., ICCV 2023):
  - Parallel region-specific color filters (saturation, contrast, hue, etc.) with predicted attention masks, summed linearly.
  - RSFNet-map, P1 480p: **25.49 / 0.924 / 7.23**. "Zeroed as shot" 480p: 24.64 / 0.915 / 9.16, 9.98 ms. Full resolution: 24.39 dB, 12.35 ms.
  - RSFNet-global: 24.31 / 0.911 / 8.21.
  - The parallel design beat sequential orderings, and "nearly half of these sequences fail".
  - — [RSFNet](https://arxiv.org/abs/2303.08682)
- **White-Box / Exposure** (Hu et al., TOG 2018): RL plus GAN, with differentiable global filters; unpaired. P2 18.59 / 0.797 / 13.24 (8.56M params). Zeng P1 unpaired: 21.32 / 0.864 / 12.65. — [CSRNet](https://arxiv.org/abs/2009.10390); [Zeng](https://arxiv.org/abs/2009.14468)
- **DeepUPE** (Wang et al., CVPR 2019): HDRNet-style bilateral illumination estimation. P1 480p 21.88 / 0.853 / 10.80, 927.1K params. — [AdaInt](https://arxiv.org/abs/2204.13983)
- **Zero-DCE** (Guo et al., CVPR 2020):
  - A per-pixel quadratic "LE-curve" LE(x) = x + αx(1−x), applied iteratively. It predicts 24 parameter maps (8 iterations × 3 channels), so it is spatially varying.
  - **79,416 params**, "about 500 FPS for images of size 640×480×3".
  - Trained with **zero-reference** losses (spatial consistency, exposure, color constancy, illumination smoothness) on 2,422 multi-exposure SICE images resized to 512×512.
  - It is a low-light method, not an expert-style retouching method. No FiveK retouching numbers.
  - — [Zero-DCE arXiv 2001.06826](https://arxiv.org/abs/2001.06826)

### Inferences
- Zero-DCE's non-reference losses (exposure-level and color-constancy priors) could be added as regularizers to a small-data global-parameter model, but they encode a "well-exposed" prior, not a style.

### Gaps
- NeurOp and RSFNet were not checked on small-data regimes.

---

## 7. How the repo's approach relates, and which methods suit small data / CPU training

### Takeaway
The repo's model is:
- features: a frozen/pretrained ResNet-18 GAP (512-D) plus a differentiable per-channel CDF (768-D);
- head: an MLP (1280→512→256→21);
- output: 21 **global** parameters (7 tone-curve interior knots at K=9, a 3×3 color-matrix offset, a color bias, grain and vignette);
- renderer: a differentiable one.

Architecturally it is a **CURL-without-TED** / **CSRNet-with-explicit-renderer** hybrid, closest in spirit to CURL's global curves and CSRNet's histogram-conditioned global modulation. None of the published methods demonstrates FiveK training at tens to hundreds of pairs. The closest evidence is CURL's 90-image S7 experiment and CSRNet's condition-only fine-tuning. The repo's own 80→500-pair result is consistent with a global-capacity ceiling.

### Cited Findings
- Repo design: 1280-D descriptor into a small MLP. The parameter layout is "Tone-curve interior knots (K−2), 3×3 colour matrix offset (9), colour bias (3), grain (1), vignette (1)". The renderer is identity at zero output. Loss: L1 + VGG + CDF. — [repo README.md](/home/user/image_editing/README.md); head layers at /home/user/image_editing/models/generic_head.py (Linear input→512→256→num_params).
- Repo small-data results:
  - 80 FiveK expert-C pairs, 12 epochs: held-out L1 to expert fell from 0.0917 (input) to 0.0741, and 4/5 test pairs moved toward the expert.
  - Scaling to 500 pairs made it worse: L1 0.0825→0.1030, with 1/5 pairs improving. The authors attribute this to a capacity ceiling of global primitives.
  - — [repo README.md](/home/user/image_editing/README.md)
- CSRNet's ablation shows histogram conditioning is the strongest global prior (hist > mean intensity > brightness). This supports the repo's CDF feature choice. — [CSRNet](https://arxiv.org/abs/2009.10390)
- CURL shows curves across multiple color spaces (Lab+RGB+HSV, including a hue→saturation curve) beat single-space curves by 0.1–1.2 dB, and that a curve curvature regularizer matters. — [CURL](https://arxiv.org/abs/1911.13175)
- Purely global variants give up about 1 dB against spatially varying ones in the same paper: RSFNet-global 24.31 vs RSFNet-map 25.49 (P1 480p). CSRNet (global, 25.17) nonetheless beats HDRNet (local, 24.66) and DeepLPF (local, 24.73) on P1. — [RSFNet](https://arxiv.org/abs/2303.08682); [AdaInt](https://arxiv.org/abs/2204.13983)
- Compute reported:
  - CSRNet: 5 h on a 2080Ti (6e5 iterations, batch 1). — [CSRNet](https://arxiv.org/abs/2009.10390)
  - NeurOp: about 11 h on a 2080Ti. — [NeurOp](https://arxiv.org/abs/2207.08080)
  - D&R: 12 h or more on a GTX 1080. — [D&R](https://arxiv.org/abs/1804.04450)
  - HDRNet: inference 14 ms on a phone CPU. — [HDRNet](https://arxiv.org/abs/1707.02880)
  - No paper reports CPU training.

### Inferences
- **Small data (tens to hundreds of pairs):**
  - Best bets are low-parameter global methods: CSRNet (36K), NeurOp (28K), and the repo-style explicit parametric head on frozen features.
  - Distort-and-recover pseudo-pairs can multiply data from retouched-only images, and fine-tuning only CSRNet's condition network is a documented few-parameter adaptation path.
  - HDRNet (480K, spatial grid) and DeepLPF/CURL (1.4–1.8M with U-Net/TED backbones) are more likely to overfit at that scale. This is an inference; none of them was tested at that scale.
- **CPU training:** CSRNet and NeurOp are small enough to train on CPU at about 500 px (a per-pixel 1×1 MLP), as is a frozen-ResNet + MLP head where features can be cached. DeepLPF's U-Net and D&R's 259M-parameter FC agent are impractical.
- **Interpretability ranking** (most to least slider-able):
  1. D&R: named ±5% actions.
  2. Repo 21-D head: curve knots and a color matrix.
  3. CURL: 10 displayable curves, though TED also edits pixels.
  4. DeepLPF: drawable graduated/elliptical masks plus a polynomial.
  5. NeurOp and RSFNet: scalar operator strengths / filter arguments plus masks.
  6. HDRNet: a grid of affine matrices.
  7. CSRNet: a latent modulation vector, controllable only via blend strength.
- **Upgrades suggested by the literature:**
  - Add Lab/HSV curves, including hue→saturation (CURL).
  - Use the histogram/CDF condition (CSRNet, already done).
  - Add a curve smoothness regularizer (CURL).
  - For local edits, use a handful of explicit graduated/radial filters (DeepLPF) or region-mask plus global filter parameters (RSFNet), rather than a full per-pixel decoder.

### Gaps
- No published head-to-head of these methods at 50–500 FiveK pairs. Few-shot claims above are inferences.
- CURL's knot count and runtime, and CURL and DeepLPF results on the P2 split, were not verified.
