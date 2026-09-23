# Image-adaptive 3D LUTs (Zeng et al. 2020) and LUT-family successors for learned photo enhancement

Scope note: numbers below were read directly from the arXiv / CVF PDFs and official GitHub READMEs/code (fetched Sept 2026). Unless stated otherwise, "FiveK" means MIT-Adobe FiveK, expert C target, 4,500 train / 500 test split, sRGB-to-sRGB "photo retouching", with models trained at 480p (short side = 480) and tested at 480p and/or original ~12 MP ("4K"/full-res). PSNR/SSIM/ΔE values from different papers are not always computed identically (e.g., SA-3DLUT reports a different SSIM for the same 3DLUT baseline), so cross-paper comparisons of ±0.1 dB should be treated as noise.

## 1. Zeng et al. 2020 (TPAMI) — how it works exactly

### Takeaway
Zeng et al. learn N=3 basis 3D LUTs (33³ lattice, 3×33³ ≈ 108K params each) plus a tiny CNN (269K params) that looks at a 256×256 bilinear-downsampled copy of the image and outputs N scalar weights; the per-image LUT is the weighted sum of the basis LUTs, applied to the full-resolution image by trilinear interpolation. Total ≈ 593.5K params, <2 ms per 4K image on a Titan RTX, FiveK 480p PSNR 25.21 / SSIM 0.922 / ΔE 7.61 (paired). It is purely global: the same LUT is applied to every pixel, so no local contrast / spatial adaptation, and noise is amplified.

### Cited Findings
**Architecture**
- A 3D LUT with M points per axis stores 3M³ output values; "M ... is usually set to 33 in practice" and "When M = 33, a 3D LUT contains 108K (3M³) parameters." Indexing colors are a uniform discretization of RGB; transformation = lookup + trilinear interpolation. — [Zeng et al., arXiv 2009.14468](https://arxiv.org/abs/2009.14468)
- CNN weight predictor: input is bilinearly downsampled to 256×256 ("bicubic... and box filter downsampling ... little difference"); 5 conv blocks (conv + LeakyReLU + InstanceNorm), one dropout (rate 0.5), one FC layer outputting N weights; "The whole CNN model contains only 269K parameters when N = 3." — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- Official code (`models_x.py`, class `Classifier`): `nn.Upsample(size=(256,256), mode='bilinear')`, Conv2d(3→16, stride 2) + LeakyReLU(0.2) + InstanceNorm, then discriminator-style blocks 16→32→64→128→128, Dropout(0.5), and the "FC" is implemented as `nn.Conv2d(128, 3, 8)`. — [HuiZeng/Image-Adaptive-3DLUT models_x.py](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
- Total params: 269K CNN + 3×108K LUTs ≈ 593K; AdaInt/SepLUT/4D LUT papers all list the re-run baseline as **593.5K** params. README: "less than 600K parameters and takes less than 2 ms to process an image of 4K resolution using one Titan RTX GPU." — [Zeng README](https://github.com/HuiZeng/Image-Adaptive-3DLUT); [AdaInt Table 2](https://arxiv.org/abs/2204.13983)
- Initialization: first basis LUT = identity, others = zero; FC bias set to 1 so initial output ≈ input. — [Zeng et al.](https://arxiv.org/abs/2009.14468)
- Ablation on N (FiveK 480p): N=1 w/o CNN (a single fixed learned LUT): 20.37 dB / 0.852 / ΔE 13.47; N=1 with CNN 23.15; N=2 24.86; **N=3 25.21 / 0.922 / 7.61**; N=4 25.26; N=5 25.29 → N=3 chosen as the knee. — [Zeng et al. Table 2](https://arxiv.org/abs/2009.14468)

**Regularizers and loss**
- Smoothness: R_TV = Σ_{c∈{r,g,b}} Σ_{i,j,k} (‖c_O(i+1,j,k) − c_O(i,j,k)‖² + ‖c_O(i,j+1,k) − c_O(i,j,k)‖² + ‖c_O(i,j,k+1) − c_O(i,j,k)‖²) (L2 chosen over L1 for smoother result); full smooth term R_s = R_TV + Σ_n ‖w_n‖² (L2 on predicted weights). — [Zeng et al. Eqs. 11–12](https://arxiv.org/abs/2009.14468)
- Monotonicity: R_m = Σ_c Σ_{i,j,k} [g(c_O(i,j,k) − c_O(i+1,j,k)) + g(c_O(i,j,k) − c_O(i,j+1,k)) + g(c_O(i,j,k) − c_O(i,j,k+1))], g = ReLU; motivated by preserving relative brightness/saturation and by updating lattice cells not hit by training colors ("training data may be insufficient to cover the entire color space"). — [Zeng et al. Eq. 13](https://arxiv.org/abs/2009.14468)
- L_paired = L_mse + λs·R_s + λm·R_m; L_unpaired = L_gan + λs·R_s + λm·R_m; **λs = 0.0001, λm = 10**. Large λs (>1e-4) hurts PSNR; PSNR is insensitive to λm. — [Zeng et al. Eqs. 14–15, Fig. 3](https://arxiv.org/abs/2009.14468)
- Unpaired GAN variant: WGAN-GP-style; generator loss −D(G(x)) + λ1·E‖G(x) − x‖² with λ1 = 1,000 (content preservation), gradient penalty λ2 = 10; CNN for unpaired drops InstanceNorm and dropout. — [Zeng et al. §3.3](https://arxiv.org/abs/2009.14468)
- Training: Adam, batch size 1, lr 1e-4 (paired) / 2e-4 (unpaired); augment with random crops at scale [0.6, 1.0], horizontal flip, slight brightness/saturation jitter. Trilinear interpolation is a custom CUDA op. — [Zeng et al. §3.5](https://arxiv.org/abs/2009.14468)

**Data / splits**
- FiveK expert C; paired: 4,500 train / 500 test; unpaired: 2,250 source + 2,250 retouched targets from different images, same 500 test. Two resolutions: 480p (short side 480) and original (12–13 MP). Competitors trained on 256×256 (480p) or 1024×1024 (original) crops. — [Zeng et al. §4.1, §4.3](https://arxiv.org/abs/2009.14468)
- README: only 480p FiveK (8-bit sRGB, 16-bit XYZ inputs, 8-bit sRGB targets) is released; "A model trained on the 480p resolution can be directly applied to images of 4K (or higher) resolution without performance drop." — [Zeng README](https://github.com/HuiZeng/Image-Adaptive-3DLUT)

**Results (Zeng's own paper)**
| Setting | PSNR | SSIM | ΔE | Source |
|---|---|---|---|---|
| FiveK retouch, paired, 480p | 25.21 | 0.922 | 7.61 | Table 3 |
| FiveK retouch, paired, original res | 25.10 | 0.930 | 7.72 | Table 3 |
| (HDRNet, paired 480p / orig) | 24.32 / 24.03 | 0.912 / 0.919 | 8.49 / 8.68 | Table 3 |
| FiveK retouch, **unpaired**, 480p | 22.86 | 0.887 | 10.28 | Table 4 |
| FiveK retouch, **unpaired**, original | 22.78 | 0.898 | 10.42 | Table 4 |
| (UIE, best unpaired competitor 480p) | 22.11 | 0.879 | 11.21 | Table 4 |
| FiveK XYZ→sRGB "imaging pipeline", paired 480p | 25.06 | 0.920 | 7.63 | Table 5 |
| HDR+ (Nexus 6P subset, 675 train/250 test), paired 480p | 23.54 | 0.885 | 7.93 | Table 5 |
| FiveK XYZ→sRGB, unpaired 480p | 21.60 | 0.852 | 11.95 | Table 6 |
| HDR+, unpaired 480p | 18.98 | 0.767 | 16.21 | Table 6 |
— all from [Zeng et al.](https://arxiv.org/abs/2009.14468) (table columns reconstructed from PDF text extraction; values for "Ours" cross-checked against Table 2 ablation 25.21/0.922/7.61).
- Runtime (Titan RTX, 100 runs): 0.64 ms @1920×1080, 1.66 ms @3840×2160, 3.76 ms @6000×4000; HDRNet 45 / 210 / 590 ms on the same GPU (authors note HDRNet can be much faster with custom OpenGL shaders). — [Zeng et al. Table 7](https://arxiv.org/abs/2009.14468)
- Independent re-runs of the same model: AdaInt reports 25.29/0.923/7.55 (480p) and 25.25/0.932/7.59 (4K) at 1.17 / 1.49 ms on V100 — [AdaInt Table 2](https://arxiv.org/abs/2204.13983); ICELUT retrained it at 25.19/0.912/7.61 — [ICELUT Table 4](https://arxiv.org/abs/2403.19238).

**Limitations (authors' own §4.7)**
- "once the 3D LUT is determined for an input image, it is the same for different local areas within the image" → insufficient local contrast in high-dynamic-range scenes; authors suggest pre-applying local tone mapping (guided-filter variant) before the LUT. Second, a LUT transforms each RGB value independently, so it cannot do detail enhancement/denoising — dark noisy HDR+ night images get noise amplified. — [Zeng et al. §4.7](https://arxiv.org/abs/2009.14468)
- Cell utilization: a single 33-point 3D LUT uses only about 5.53% of its cells for a typical image (SepLUT's measurement on Zeng's model). — [SepLUT §4.4](https://arxiv.org/abs/2207.08351)

### Inferences
- The "593K" figure is ~55% LUT storage (324K) and ~45% CNN; the per-image deployable artifact is only one 33³ LUT (108K floats) plus 3 weights.
- The HDR+ paired score (23.54 dB on only 675 training pairs) is the only in-paper evidence of behaviour with a small training set; it is lower than FiveK, which the authors attribute partly to fewer training pairs.

### Gaps
- Zeng does not report CPU runtime or a training-set-size ablation.
- The TPAMI published version (vol. 44, no. 4, 2022, pp. 2058–2073 per the README citation) may differ slightly from arXiv v1; I read arXiv v1.

## 2. AdaInt (Yang et al., CVPR 2022)

### Takeaway
AdaInt keeps Zeng's framework (same backbone, N=3 basis LUTs, 33 points) but additionally predicts image-adaptive, per-channel non-uniform sampling positions for the lattice, so resolution is concentrated where the color transform is most non-linear. Cost: +26K params (593.5K → 619.7K), +~0.1 ms; gain ≈ +0.2 dB on FiveK.

### Cited Findings
- Mechanism: the shared backbone f feeds an extra FC head g predicting 3×(Ns−1) unnormalized intervals; softmax per channel → normalized intervals; cumsum with a prepended 0 → monotonically increasing coordinates in [0,1]; a differentiable "AiLUT-Transform" does lookup (binary search) + trilinear interpolation on the non-uniform lattice and backpropagates to the coordinates. g's weights init to 0 and bias to 1 (uniform start); lr of g decayed ×0.1 and frozen for the first 5 epochs. — [AdaInt arXiv 2204.13983](https://arxiv.org/abs/2204.13983)
- Training: Adam, batch 1 (FiveK) / 16 (PPR10K), 400 epochs, lr 1e-4, V100; Ns = 33; FiveK 4,500/500 at 480p train, tested at 480p and original 4K; PPR10K 8,875/2,286 at 360p. — [AdaInt](https://arxiv.org/abs/2204.13983)
- FiveK retouching (Table 2): 3D-LUT 593.5K params 25.29/0.923/7.55 @480p (1.17 ms), 25.25/0.932/7.59 @4K (1.49 ms); **3D-LUT + AdaInt 619.7K params: 25.49/0.926/7.47 @480p (1.29 ms), 25.48/0.934/7.45 @4K (1.59 ms)**. SA-3DLUT (numbers copied from its paper): 4.5M params, 25.50 @480p, 2.27 / 4.39 ms. — [AdaInt Table 2](https://arxiv.org/abs/2204.13983)
- FiveK tone mapping (16-bit XYZ → sRGB, 480p): 3D-LUT 25.07/0.920/7.55 → AdaInt 25.28/0.925/7.48. — [AdaInt Table 3](https://arxiv.org/abs/2204.13983)
- PPR10K (ResNet-18 backbone as in PPR10K paper): expert a: 3D-LUT 25.64 / ΔE 6.97, +HRP 25.99 / 6.76, +AdaInt 26.33 / 6.56; expert b: 24.70 → 25.06 → 25.40; expert c: 25.18 → 25.46 → 25.68. — [AdaInt Table 4](https://arxiv.org/abs/2204.13983)
- Official repo (MMEditing 0.11-based); `model.en_adaint=False` degenerates to the TPAMI 3D-LUT; AiLUT-Transform is a CUDA extension (prebuilt wheel for Py3.7/PyTorch 1.8.1/CUDA 10.2); pretrained FiveK-sRGB checkpoint provided. — [ImCharlesY/AdaInt README](https://github.com/ImCharlesY/AdaInt)

### Inferences
- A non-uniform lattice is not directly representable in a standard .cube (uniform grid); exporting would require resampling the learned non-uniform LUT onto a uniform 33³ or 65³ grid (small approximation error), or emitting a 1D per-channel shaper + uniform 3D LUT (the non-uniform per-axis coordinates are exactly a per-channel 1D warp).

### Gaps
- No CPU runtime reported for AdaInt; the AiLUT op is CUDA-only in the official repo (no CPU kernel mentioned in README).

## 3. SepLUT (Yang et al., ECCV 2022)

### Takeaway
SepLUT splits the global color transform into a per-channel 3×1D LUT (component-independent, "curves") followed by a small 3D LUT (component-correlated, hue/saturation), both predicted by a tiny CNN. The 1D stage spreads colors over the 3D lattice, so a 9³ or 17³ 3D LUT suffices: 47.2K–119.8K params (5–12× smaller than Zeng) with slightly better accuracy (25.42–25.47 dB), and it runs on CPU. The 1D-curve-then-3D-LUT structure maps naturally onto editor concepts (tone curves + color LUT).

### Cited Findings
- Framework: backbone CNN (width m) on downsampled input → a 1D-LUT generator (sigmoid-normalized outputs, So points per channel) and a 3D-LUT generator (St points); applied sequentially: 3×1D LUT then 3D LUT via trilinear interpolation. — [SepLUT arXiv 2207.08351](https://arxiv.org/abs/2207.08351)
- FC layers of the LUT generators are equivalent to image-independent basis LUTs linearly combined by image-dependent coefficients (so Zeng's basis-fusion is a special case); this makes post-training 8-bit quantization trivial. — [SepLUT §3.5](https://arxiv.org/abs/2207.08351)
- 3D-LUT-only ablation (So=0, m=8): St 33/17/9 → 385K/106K/69K params → 25.27/25.24/25.21 dB, i.e., "capacity redundancy of the 3D LUT". — [SepLUT §4.3](https://arxiv.org/abs/2207.08351)
- Component-independent stage ablation (FiveK 480p): m=6, St=So=9: HE 21.75 / single 1D LUT 25.32 / 3×1D LUT 25.42 dB (47.2K params); m=8, St=So=17: 21.76 / 25.41 / 25.47 dB (119.8K). — [SepLUT Table 4](https://arxiv.org/abs/2207.08351)
- FiveK retouching (Table 6): **Ours-S (m=6, So=St=9) 47.2K params: 25.42/0.921/7.51 @480p, 25.40/0.931/7.52 @4K; Ours-L (m=8, So=St=17) 119.8K: 25.47/0.921/7.54 @480p, 25.43/0.932/7.56 @4K**; 3D-LUT 593.5K: 25.29/0.920/7.55 and 25.25/0.930/7.59. — [SepLUT Table 6](https://arxiv.org/abs/2207.08351)
- FiveK tone mapping 480p: 3D-LUT 25.07/0.920/7.55; Ours-S 25.42/0.920/7.43; Ours-L 25.43/0.922/7.43. PPR10K (360p) PSNR/ΔE, expert a: 3D-LUT 25.64/6.96, S 26.19/6.71, L 26.28/6.59. — [SepLUT Tables 7–8](https://arxiv.org/abs/2207.08351)
- GPU runtime (ms, 480p/720p/4K): 3D-LUT 1.02/1.06/1.14; Ours-S 1.08/1.09/1.18; Ours-L 1.10/1.12/1.20; SA-3DLUT 2.27/2.34/4.39. — [SepLUT Table 9](https://arxiv.org/abs/2207.08351)
- **CPU** (Intel Xeon Platinum 8163, 480p): 3D-LUT 17.35 ms → 15.52 ms after 8-bit quantization (593.5K → 332.5K equiv. params, PSNR 25.28 → 25.25); Ours-S 25.34 → 15.65 ms (47.2K → 37.9K, 25.42 → 25.35); Ours-L 25.64 → 16.21 ms (119.8K → 76.3K, 25.47 → 25.43). No fine-tuning / QAT needed. — [SepLUT Table 5](https://arxiv.org/abs/2207.08351)
- Analysis: the learned 1D LUTs "stretch the input brightness and image contrast", the 3D LUT then "alter[s] the hue and enhanc[es] the saturation"; the 1D stage increases 3D-LUT cell utilization vs. Zeng's ~5.53%. — [SepLUT §4.4](https://arxiv.org/abs/2207.08351)
- Repo includes "C++ CPU/CUDA implementation" of the cascaded 1D+3D LUT transform. — [ImCharlesY/SepLUT README](https://github.com/ImCharlesY/SepLUT)

### Inferences
- For editability, SepLUT is the most "Lightroom-shaped" of the family: per-image output = three tone curves + a small 3D color LUT, both globally applied. A per-image SepLUT result can be baked into a single uniform .cube losslessly-in-principle by composing curve∘LUT on a dense grid (e.g., 33³/65³), or kept separate as a 1D shaper + 3D LUT.

### Gaps
- No user study or explicit editability evaluation in the paper; the "curves" interpretation is the authors' qualitative analysis.

## 4. Spatial-aware 3D LUT (Wang et al., ICCV 2021, Huawei Noah's Ark; often "SA-3DLUT")

### Takeaway
Adds spatial adaptivity by having a two-head encoder–decoder predictor output (a) a global T-dim weight vector (T=3 "scenes") and (b) a per-pixel M-channel weight map (M=10 "categories"); each pixel's output is a pixel-wise blend of M basis-LUT outputs (spatial-aware trilinear interpolation), then blended over T. Much better on locally-complex HDR+ data but ~4.5M params, ~4 ms/4K on V100, and no public code.

### Cited Findings
- Two-head weight predictor on a resized low-res image: head 1 = 1D weight vector over T scene LUT-sets (T=3 "according [Zeng]"); head 2 = M-channel pixel-wise weight map (upsampled to full res) fusing M basic LUTs per spatial-aware LUT; M=10 chosen (gains up to 10, flat/worse beyond). — [Wang et al. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)
- Loss: L = L_r + 0.0001·L_s + 10·L_m + 0.005·L_c (CIE94 color loss) + 0.05·L_p (LPIPS); Adam, cosine LR (amplitude 2e-4, period 20 epochs), 400 epochs, batch 1, V100. — [Wang et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)
- Ablation (their HDR+ 480p): 3DLUT(3,0) 539K params 19.91 dB; simply 30 LUTs 3.72M 20.29 dB; Ours(3,10) 4.52M params, 1.114 GFLOPs, 22.73 dB / SSIM 0.7420 / LPIPS 0.1580. — [Wang et al. Table 1](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)
- FiveK 480p (Zeng's released data): 3DLUT 25.24/0.8864/LPIPS 0.0530 vs **Ours 25.50/0.8904/0.0512**. FiveK full-res (**their own DNG→PNG conversion, not Zeng's Lightroom-processed inputs**): 3DLUT 22.27 vs Ours 23.17. — [Wang et al. Table 3](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)
- HDR+ 480p (Zeng's 675-pair release): 3DLUT 23.59 vs Ours 28.29 dB; their own re-built HDR+ (2,041 pairs, 1,837/204) 480p: 19.91 → 22.73; full-res: 19.88 → 22.56. — [Wang et al. Table 4](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf)
- "about 4ms to process a 4K resolution image on one NVIDIA V100 GPU". — [Wang et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf); later papers cite 2.27 ms @480p / 4.39 ms @4K and mark it "results adopted from the original paper ... due to the unavailable source code." — [AdaInt Table 2](https://arxiv.org/abs/2204.13983), [SepLUT Table 6](https://arxiv.org/abs/2207.08351)

### Inferences
- SSIM values in this paper (e.g., 0.886 for 3DLUT on FiveK 480p) differ from Zeng's own 0.922 for the same model/data, so SA-3DLUT's metric computation differs; only within-paper deltas are trustworthy.
- The FiveK gain is small (+0.26 dB) while the HDR+ gain is large (+4.7 dB on Zeng's split), consistent with FiveK expert-C edits being mostly global.
- Output is not expressible as a single LUT (pixel-dependent blend of 10 LUTs), so not .cube-exportable as one file.

### Gaps
- No official code found; parameter count (4.5M) dominated by the encoder–decoder.

## 5. CLUT-Net (Zhang, Zeng, Zhang, Zhang, ACM MM 2022)

### Takeaway
Same authors' group as Zeng; compresses the basis LUTs into low-dimensional "CLUTs" plus two learned transformation matrices that reconstruct a standard 3D LUT, exploiting inter-channel correlation. Reported FiveK ≈ 25.53–25.68 dB with fewer parameters than Zeng. Because it reconstructs a standard 3D LUT per image, it remains .cube-exportable.

### Cited Findings
- Framework: a network predicts content-dependent weights from a downsampled input to fuse N basis CLUTs; "two transformation matrices adaptively reconstruct the corresponding standard 3DLUT to enhance the original input image." Related ICME 2023 HashLUT variant uses multi-resolution hash-encoded LUTs; "The standard 3DLUT would be reconstructed from CLUT but not from HashLUT," and HashLUT is hard to regularize with TV/monotonicity. — [Xian-Bei/CLUT README](https://github.com/Xian-Bei/CLUT)
- Pretrained FiveK models: CLUTNet 20+05+10 → 25.56 PSNR; CLUTNet 20+05+20 → 25.68 PSNR; HashLUT 7+13 → 25.62; HashLUT 6+13 SmallBackbone → 25.57 "(about 110K params)". Dataset setting = Zeng's (480p FiveK). — [CLUT README](https://github.com/Xian-Bei/CLUT)
- Third-party table (ICELUT, FiveK): CLUT 25.53 / 0.926 / ΔE 7.46 vs 3D LUT 25.19 / 0.912 / 7.61; storage 1,168 KB vs 3DLUT 2,368 KB; CPU (Xeon 8260L, 480p) 8.72 + 6.71 ms (network + interpolation) vs 3DLUT 7.15 + 6.71 ms. — [ICELUT Tables 4 & 6](https://arxiv.org/abs/2403.19238)
- The README notes the CLUT code relies on Zeng's CUDA trilinear extension. — [CLUT README](https://github.com/Xian-Bei/CLUT)

### Inferences
- From ICELUT's storage number (1,168 KB ≈ 292K float32 params), CLUT-Net's total parameter count is roughly half of Zeng's; treat this as an estimate.

### Gaps
- The CLUT-Net paper is not on arXiv (ACM DL only), so I could not verify its exact parameter count, the meaning of the "20+05+10" config (likely N basis LUTs + compressed dims), or 480p vs full-res breakdown from the primary source. Flag: exact params unverified.

## 6. 4D LUT (Liu et al., IEEE TIP 2023)

### Takeaway
Adds a learned per-pixel scalar "context map" C (1×H×W, from a full-resolution residual-block context encoder) as a 4th LUT input (RGBC), with quadrilinear interpolation; basis 4D LUTs are fused by predicted coefficients. ~924K params, ~+0.3 dB over 3D LUT on FiveK variants, 5.75 ms (2080 Ti).

### Cited Findings
- Components: context encoder (four 3×3 residual blocks + one 1×1 residual block) → context map C ∈ R^{1×H×W}; parameter encoder → fusion weights W (Nw=27) and biases B (Nb=3); fusion of Nlut=3 basis 4D LUTs; quadrilinear interpolation; 4D smooth + 4D monotonicity regularizers with αs=0.0001, αm=10; Nbin=33. — [4D LUT arXiv 2209.01749](https://arxiv.org/abs/2209.01749)
- Official code allocates each basis 4D LUT as `np.zeros((3,2,dim,dim,dim))`, i.e., **only 2 samples along the context axis** (context acts as a linear blend between two 3D LUTs), with `dim=17` default in the code vs 33 in the paper. — [ChengxuLiu/4DLUT models_x.py](https://github.com/ChengxuLiu/4DLUT)
- Datasets differ from Zeng's: "MIT-Adobe-5K-UPE" (4,500/500, long edge 510 px) and "MIT-Adobe-5K-DPE" (2,250 train / 500 test), plus PPR10K 360p. 4D LUT: 24.96 dB (5K-UPE) and 24.61 dB (5K-DPE), +0.36 / +0.28 dB over 3D LUT. — [4D LUT §IV](https://arxiv.org/abs/2209.01749)
- PPR10K (Table III): runtime/params 3D LUT 1.99 ms / 593.5K vs **4D LUT 5.75 ms / 924.4K**; PSNR a/b/c: 3D LUT 24.632/24.101/24.515 vs 4D LUT 24.915/24.398/24.733. — [4D LUT Table III](https://arxiv.org/abs/2209.01749)
- Nbin ablation: 9 → 33 bins raises PSNR from 24.67 to 24.96 dB; regularizers: Lr only 24.74, +Ls 24.79, +Lm … full 24.96. — [4D LUT §V](https://arxiv.org/abs/2209.01749)

### Inferences
- Because the context encoder runs at full resolution, runtime scales with image size unlike Zeng's fixed-cost predictor. Output is not a single 3D LUT → not .cube-exportable except as two LUTs plus a per-image mask.

### Gaps
- Paper's text on 4D lattice size along the context axis is not explicit; the "2 bins" reading comes from the released code and may not match the paper configuration. FiveK numbers are on non-Zeng preprocessing (510 px), so not directly comparable to the 25.2–25.8 dB cluster.

## 7. NILUT (Conde, Vazquez-Corral, Brown, Timofte, AAAI 2024)

### Takeaway
Different problem: NILUT is a small coordinate MLP (RGB→RGB) that *emulates* existing professional 3D LUTs, optionally conditioned on a style one-hot vector (CNILUT) so one network stores several LUTs and blends them by interpolating the condition. It is not image-adaptive and is not trained on FiveK input/expert pairs; fitted from Hald images, it reaches >40 dB / ΔE <2 fidelity with ~34K params.

### Cited Findings
- Training data: the entire 8-bit RGB cube (256³ ≈ 16.78M colors, as a 4096×4096 "RGB map"/Hald) processed by real LUTs in Photoshop; the MLP "overfits" this mapping; "we do not require natural images to learn real 3D LUTs, just the corresponding RGB maps (Halds)." — [NILUT arXiv 2306.11920](https://arxiv.org/abs/2306.11920)
- Table 1 (5 professional LUTs; fidelity on RGB map and on 100 MIT5K images processed by the real LUT): MLP-Res N=128, L=2: PSNR_rgb 45.34 / ΔE 0.97; PSNR_5k 42.04 / ΔE_5k 1.65. SIREN 128×2: 44.43/1.04 and 41.17/1.63. — [NILUT Table 1](https://arxiv.org/abs/2306.11920)
- Param counts: 256×2 = 133.3K, 256×3 = 199.1K, 128×2 = 33.9K, 128×3 = 50.4K, 64×2 = 8.7K, 64×3 = 12.9K; compare a 33³ LUT ≈ 107K floats. MLP-Res reaches "almost perfect mapping in 4 minutes without using special INR acceleration." — [NILUT Table 2, §4](https://arxiv.org/abs/2306.11920)
- CNILUT: a single network with one-hot style condition represents 3 styles, each <3 ΔE from the real LUT; blending by changing the condition vector "happens implicitly without additional computational cost." — [NILUT Fig. 7](https://arxiv.org/abs/2306.11920)
- README: fit a .cube by rendering a Hald through it, then `fit.py ... --steps 1000 --units 128 --layers 2`; "In less than 30s you have it!"; pretrained `nilutx3style.pt` encodes 3 styles; dataset of 100 MIT5K images × several LUTs on Kaggle. — [mv-lab/nilut README](https://github.com/mv-lab/nilut)

### Inferences
- NILUT is a *representation* for LUTs (and style blending), not a replacement for Zeng-style auto-enhancement; it can be combined with an image-adaptive predictor (see prompt-guided NILUT follow-up, [arXiv 2408.11055](https://arxiv.org/html/2408.11055)) but that paper was not verified here.
- Since a NILUT is a continuous RGB→RGB function, exporting to .cube is just evaluating it on a 33³/65³ grid.

### Gaps
- The hardware for the "30 s" / "4 min" fits is not specified in what I read.

## 8. Other notable post-2023 LUT-family methods (brief)

### Takeaway
The 2024–2026 line pushes on (a) spatial awareness at low cost via bilateral grids (LUTwithBGrid/SABLUT, ECCV 2024; its SVD-decomposed follow-up, 2025) and (b) pure-LUT inference for CPU/edge (ICELUT, ECCV 2024). On Zeng's FiveK 480p protocol the best numbers are now ~25.7–25.8 dB with ~160K params.

### Cited Findings
- **ICELUT** ("Taming Lookup Tables for Efficient Image Retouching", Yang et al., ECCV 2024): all-1×1 conv + split FC, converted to LUTs so inference has no CNN; works with 32×32 input; FiveK 25.27/0.918/ΔE 7.51 vs CLUT 25.53; runtime 480p: GPU 0.35+0.05 ms, **CPU (Xeon 8260L) 0.97+6.71 ms**, smartphone Cortex-A55 7.8+17 ms; storage 780 KB. Same table: 3DLUT CPU 7.15+6.71 ms, HDRNet 167 ms. — [ICELUT arXiv 2403.19238](https://arxiv.org/abs/2403.19238)
- **LUTwithBGrid / "SABLUT"** (Kim & Cho, ECCV 2024, "Image-Adaptive 3D Lookup Tables for Real-Time Image Enhancement with Bilateral Grids"): as tabulated by the authors' follow-up: 463.7K params, FiveK 480p 25.66/0.930/7.29 (1.20 ms), 4K 25.66/0.937/7.27 (3.64 ms). — [Kim, Lee, Cho arXiv 2508.16121 Table 3](https://arxiv.org/abs/2508.16121)
- **Decomposed spatial-aware LUTs** (Kim, Lee, Cho, arXiv 2508.16121, 2025): decomposes 3D LUT into a linear sum of low-dimensional LUTs via SVD plus cache-efficient bilateral-grid fusion; 160.5K params, FiveK 480p 25.76/0.931/7.26 (1.37 ms), 4K 25.69/0.938/7.27 (1.38 ms); tone-mapping 480p 25.59. Same table lists AdaInt 619.7K 25.49 and SepLUT 119.8K 25.47. — [arXiv 2508.16121](https://arxiv.org/abs/2508.16121)
- **LoR-LUT** (Zhao et al., arXiv 2602.22607, Feb 2026): low-rank residual formulation of compact LUTs, compared against CLUT and IA-3DLUT; includes an interactive "LoR-LUT Viewer". — [arXiv 2602.22607](https://arxiv.org/pdf/2602.22607) (not read in detail)
- Self-distilled adaptive-interval LUTs ("born-again" iterative self-distillation on AdaInt to improve generalization given limited dataset scale), Pattern Recognition 2025. — [ScienceDirect S0031320325012622](https://www.sciencedirect.com/science/article/abs/pii/S0031320325012622) (abstract only)

### Inferences
- Bilateral-grid methods regain local adaptivity but break the "one LUT per image" property that makes export to editors trivial.

### Gaps
- DualBLN and other hybrids were not investigated. Venue of 2508.16121 not confirmed from the PDF header.

## 9. Exporting learned LUTs to .cube (Lightroom/Resolve editability)

### Takeaway
For global methods (Zeng, CLUT-Net, SepLUT baked) the per-image fused LUT is a standard uniform 33³ RGB→RGB table and can be written to .cube with a trivial script; the official Zeng repo does not ship a .cube exporter, but its text LUT format is already in .cube's red-fastest ordering. Spatial/4D/bilateral-grid methods cannot be exported as a single .cube.

### Cited Findings
- Zeng repo's `utils/generate_identity_3DLUT.py` writes one "r g b" line per lattice point with loops `for k (outer) → j → i (inner)` writing `(step*i, step*j, step*k)`, i.e., red varies fastest; `Generator3DLUT_identity` reads it into a tensor `buffer[c, i, j, k]` with line index `n = i*dim² + j*dim + k`. The repo also offers `utils/visualize_lut.py` and supports arbitrary dim (e.g., 64). — [HuiZeng/Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) (README + `utils/generate_identity_3DLUT.py`, `models_x.py`)
- README "useful issue": replace custom trilinear with `torch.nn.functional.grid_sample` (issue #14), which removes the CUDA-only dependency. — [Zeng README](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
- NILUT's workflow is the inverse direction (.cube → Hald → network), showing Lightroom/Photoshop-applied LUTs and learned LUTs are interchangeable via Hald images. — [NILUT README](https://github.com/mv-lab/nilut)

### Inferences
- Export recipe (not verified end-to-end): run the CNN on an image → weights w; LUT = Σ w_n·φ_n (shape 3×33×33×33, indexed [c, b, g, r] per the loader); flatten in (b, g, r) C-order with red fastest; write header `LUT_3D_SIZE 33` then rows `R G B` (clip to [0,1]). This matches the standard .cube ordering. Resolve loads .cube directly; Lightroom needs the LUT wrapped as a profile (Camera Raw/Photoshop profile creation) — I did not verify current Adobe steps.
- Editability caveat: the learned LUT is *per image*. One can export (i) per-image LUTs, (ii) the 3 basis LUTs as "looks" whose weights the user sets manually, or (iii) a single "average style" LUT; only (i) reproduces model output. Also the LUT is trained on sRGB-encoded 8-bit JPEGs (or XYZ for tone mapping), so applying it in an editor must match that input color space/gamma (e.g., Resolve's timeline color space).
- SepLUT's 1D curves + 3D LUT and AdaInt's non-uniform axes can be expressed as a 1D shaper + 3D LUT; whether a given host accepts combined 1D+3D .cube files varies by software (unverified).

### Gaps
- No official exporter or paper evaluating fidelity of exported .cube files in Lightroom/Resolve was found.

## 10. CPU training and small datasets (tens–hundreds of pairs)

### Takeaway
Inference on CPU is well documented (≈7–25 ms per 480p image). CPU *training* is feasible in principle (≈600K params, 256×256 predictor input, batch size 1, grid_sample instead of the CUDA op), but no paper reports CPU training times or a training-set-size ablation; the smallest reported training sets are 675 pairs (HDR+) and 2,250 pairs (FiveK unpaired / 5K-DPE). Regularizers (monotonicity, smoothness) and few basis LUTs are the main built-in guard against overfitting.

### Cited Findings
- CPU inference: 3D-LUT 17.35 ms (15.52 ms quantized) and SepLUT 25.3–25.6 ms (≈16 ms quantized) per 480p image on a Xeon 8163 — [SepLUT Table 5](https://arxiv.org/abs/2207.08351); 3DLUT 7.15 + 6.71 ms, ICELUT 0.97 + 6.71 ms on a Xeon 8260L — [ICELUT Table 6](https://arxiv.org/abs/2403.19238); the 2025 decomposition paper includes CPU (i9-12900F) comparisons (Table 7) — [arXiv 2508.16121](https://arxiv.org/abs/2508.16121).
- Small-data evidence: Zeng trained HDR+ on 675 pairs (paired 480p: 23.54 dB vs HDRNet 23.04) and notes performance is lower partly because "the number of training image pairs is much less on the HDR+" — [Zeng et al. §4.4](https://arxiv.org/abs/2009.14468). Monotonicity regularization is motivated by limited color coverage in training data — [Zeng et al. §3.4](https://arxiv.org/abs/2009.14468). Zeng attributes its better generalization than HDRNet (purple-sky overfitting example) to "the compact model design ... as well as the smooth and monotonicity regularizations" — [Zeng et al. §4.3](https://arxiv.org/abs/2009.14468).
- Unpaired training on 2,250 + 2,250 images with a GAN still gives 22.86 dB (FiveK 480p) — [Zeng et al. Table 4](https://arxiv.org/abs/2009.14468).
- A single learned global LUT without the CNN (N=1, no predictor) gives 20.37 dB vs 25.21 with the adaptive model — [Zeng et al. Table 2](https://arxiv.org/abs/2009.14468).
- Training protocols are long: AdaInt/SepLUT/ICELUT train 400 epochs over 4,500 pairs at batch 1 on V100/3090 GPUs — [AdaInt](https://arxiv.org/abs/2204.13983), [ICELUT](https://arxiv.org/abs/2403.19238). NILUT fitting a single LUT from one Hald takes ~30 s–4 min — [NILUT README](https://github.com/mv-lab/nilut), [NILUT paper](https://arxiv.org/abs/2306.11920).
- Limited dataset scale "constrain[s] the generalization ability", motivating self-distillation for AdaInt — [Pattern Recognition 2025 abstract](https://www.sciencedirect.com/science/article/abs/pii/S0031320325012622).

### Inferences
- Rough compute estimate (my inference, not measured): per training step the CNN sees a 256×256 input (~tens of MFLOPs) and the LUT is applied to a crop; with 100 pairs × 400 epochs = 40K steps, CPU training is plausibly minutes-to-an-hour scale. With only tens–hundreds of pairs, sensible choices are: fewer/smaller LUTs (SepLUT-S at 47K params or 17³/9³ LUTs, which lose ≤0.06 dB on FiveK), keep λm=10 and λs=1e-4, identity-initialized LUTs, and possibly pretrain on FiveK then fine-tune on the personal set.
- If a user has only tens of pairs with a consistent style, a single non-adaptive LUT (N=1, no CNN; ~20.4 dB on FiveK where edits vary per image) or a NILUT-style fit might be sufficient when the target style is itself global and consistent.

### Gaps
- No primary source found that measures accuracy vs. number of training pairs (e.g., 50/100/500) for any LUT method, nor any that reports CPU training time. These would need to be established experimentally.
