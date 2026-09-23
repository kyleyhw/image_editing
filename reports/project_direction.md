# Project direction: content-adaptive, editable photo styles

*Summary of the September 2026 direction review. The full literature comparison is in [`Content adaptive photo edit models.md`](Content%20adaptive%20photo%20edit%20models.md).*

## 1. Problem with the original purpose

The project described itself as "a general-purpose stylization engine: the architecture is fixed and the style comes from the data". That describes a tool, not a problem, and every fixed-function style meets it trivially. The Fujifilm, Cyberpunk and Tilt-shift targets are closed-form functions *G(I)* that apply the same parameters to every photo. For them the network can at best copy *G*, and running *G* directly is faster and exact. The one result that needed the network came late: learning MIT-Adobe FiveK expert C's edits (80 pairs moved 4 of 5 held-out images toward the expert). Even that result is fragile. At 500 pairs it got worse (1 of 5), and the repo's figures for it disagree with each other. It was measured on 5 test images with one seed and no baselines. The training setup also has uncontrolled factors: a fully fine-tuned ResNet with BatchNorm at batch size 4, no ImageNet normalisation, 8 vs 12 epochs, and a single tone curve shared by all three colour channels.

## 2. Revised purpose

A style is a **target look**. The model learns the **per-image edit** that takes a given photo to that look. A dark night street and a bright beach need different edits to reach the same style, and learning that difference is the network's job. The look can be specified in one of two ways:

- **Paired examples**, such as a photographer's before/after edits (FiveK experts, or RAW+JPEG pairs from a camera).
- **Unpaired examples**, i.e. a set of photos already in the style (cyberpunk, film), learned through distribution-matching losses.

One model, conditioned on a style code, serves all looks. The output is **editable parameters** (curves, colour matrix, a small LUT, grain, vignette) that can be exported as a `.cube` file, never generated pixels. The model can then change colour and tone but cannot invent content. Constraints: CPU training, tens to hundreds of examples per style.

## 3. What the literature says

- **Zeng et al. (2020)** learn 3 shared 33³ basis LUTs. A 269K-parameter CNN predicts per-image blend weights, and the blended LUT is applied with trilinear interpolation, regularised for smoothness and monotonicity (593K parameters, 1.66 ms per 4K image). On FiveK, a fixed LUT scores 20.37 dB, while the adaptive blend scores 25.21 dB paired and 22.86 dB unpaired [1]. **Adaptivity, not spatial processing, carries most of the gain**: spatial extensions add about 0.3–0.5 dB [2, 3].
- **SepLUT** puts per-channel 1D curves before a small 9³ LUT. It matches or beats Zeng (25.42–25.47 dB) with 5–12× fewer parameters, and its LUT generator is Zeng's basis blend [4].
- **StarEnhancer** is the only published method with all three of: several styles in one model, new styles learned from a few example images, and curve-editable output [5]. PieNet learns new users from 10–20 images but outputs pixels [6].
- **CSRNet and NeurOp** (28–36K parameters) are the strongest small rivals. CSRNet shows that fine-tuning only its conditioning matches retraining for a new expert, but its output is not editable [7, 8].
- **Pixel generators and diffusion editors** (DPE, UEGAN, InstructPix2Pix) can change image content, which rules them out [9].

## 4. Recommended architecture

- **Features:** keep ResNet-18 + CDF features, but freeze and cache them and add ImageNet normalisation.
- **Renderer:** replace it with a SepLUT-style cascade: **3 per-channel monotone curves → 3×3 matrix + bias → residual 9³ LUT blended from N ≈ 4–8 shared basis LUTs** (Zeng's regularisers), then grain and vignette.
- **Head:** a smaller head, without BatchNorm, conditioned by **FiLM** on a style code. The code is a learned embedding for known styles, or the average embedding of n example images for new ones.
- **Unpaired styles:** CDF / sliced-Wasserstein colour loss, an identity anchor, and distort-and-recover pseudo-pairs. Add a small GAN critic only if needed.

**Weaknesses.** No paper tests fewer than 675 pairs. Unpaired training costs about 2.3 dB. Style codes averaged from a few images are noisy for subtle photographers: expert-identification Recall@1 is as low as 24.6% [5]. The LUT stage is not slider-editable. The combination as a whole is untested.

## 5. Next steps

1. **Remove confounds** and measure a learning curve from 25 to 4,500 pairs (3 seeds, ≥ 100 test images, baselines).
2. **Renderer ablation** against a CSRNet baseline.
3. **Multi-expert conditioning:** train on FiveK experts A–D and hold out E.
4. **Unpaired losses:** test on disjoint FiveK images so PSNR stays measurable.
5. **`.cube` export fidelity check.**

Step 1 comes first. If accuracy still falls as data grows once the confounds are fixed, no renderer change will help.

## References

[1] Zeng et al., *Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time*, TPAMI 2020. <https://arxiv.org/abs/2009.14468>
[2] Wang et al., *Real-time Image Enhancer via Learnable Spatial-aware 3D Lookup Tables*, ICCV 2021. <https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf>
[3] Yang et al., *AdaInt: Learning Adaptive Intervals for 3D Lookup Tables*, CVPR 2022 (comparison table). <https://arxiv.org/abs/2204.13983>
[4] Yang et al., *SepLUT: Separable Image-adaptive Lookup Tables*, ECCV 2022. <https://arxiv.org/abs/2207.08351>
[5] Song et al., *StarEnhancer: Learning Real-Time and Style-Aware Image Enhancement*, ICCV 2021. <https://openaccess.thecvf.com/content/ICCV2021/papers/Song_StarEnhancer_Learning_Real-Time_and_Style-Aware_Image_Enhancement_ICCV_2021_paper.pdf>
[6] Kim et al., *PieNet: Personalized Image Enhancement Network*, ECCV 2020. <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750375.pdf>
[7] He et al., *Conditional Sequential Modulation for Efficient Global Image Retouching* (CSRNet), ECCV 2020. <https://arxiv.org/abs/2009.10390>
[8] Wang et al., *Neural Color Operators for Sequential Image Retouching* (NeurOp), ECCV 2022. <https://arxiv.org/abs/2207.08080>
[9] Brooks et al., *InstructPix2Pix: Learning to Follow Image Editing Instructions*, CVPR 2023. <https://arxiv.org/abs/2211.09800>
