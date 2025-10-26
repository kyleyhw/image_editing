# Project Roadmap: CDF-based Image Editing with Deep Learning Augmentation

## 1. Project Overview

The core objective of this project is to develop a novel, mathematically-grounded approach to image editing. Initially, this involves defining image transformations based on the manipulation of Cumulative Distribution Functions (CDFs) of pixel intensities. To overcome the inherent limitations of purely statistical methods (lack of spatial awareness) and to achieve complex, stylistic edits (e.g., "cyberpunk," "digicam," "film"), we will augment this CDF analysis with deep learning techniques, specifically Convolutional Neural Networks (CNNs) and learned Look-Up Tables (LUTs).

## 2. Rationale for Augmentation

### 2.1 Limitations of Pure CDF Analysis

While CDFs provide a robust statistical description of an image's tonal and color distribution, they are inherently global and lack spatial information. This means:
*   **Loss of Spatial Context:** A purely CDF-based transformation cannot differentiate between different regions of an image. For instance, applying a "darkening" transformation via CDF manipulation would affect all pixels equally based on their intensity value, without regard for whether they belong to a highlight, shadow, or a specific object.
*   **Inability to Capture Local Effects:** Many stylistic edits (e.g., vignetting, grain, local contrast adjustments, sharpening, specific color shifts in certain areas) rely heavily on spatial relationships and local pixel neighborhoods. CDFs alone cannot encode or manipulate these.
*   **Difficulty with Complex Styles:** Styles like "cyberpunk" or "film" often involve intricate combinations of global color grading, local contrast, texture overlays (grain), and specific highlight/shadow tints, which go beyond simple intensity remapping.

### 2.2 Advantages of Convolutional Methods

Convolutional Neural Networks (CNNs) are exceptionally well-suited for extracting hierarchical features from image data, making them ideal for addressing the spatial limitations of CDFs.
*   **Spatial Feature Extraction:** CNNs, through their convolutional layers, learn to detect patterns, edges, textures, and local structures at various scales. This provides a rich, spatially-aware representation of the image.
*   **Hierarchical Representation:** Lower layers of a CNN capture basic features (edges, corners), while deeper layers learn more abstract and semantic features (object parts, textures). This hierarchy is crucial for understanding complex image content and style.
*   **Contextual Understanding:** By processing local neighborhoods and progressively aggregating information, CNNs build a contextual understanding of the image, enabling them to infer stylistic elements that depend on spatial arrangement.

## 3. Proposed Augmented Architecture

The augmented architecture will combine the strengths of statistical CDF analysis with the spatial feature extraction capabilities of CNNs.

### 3.1 Feature Engineering

1.  **CDF Features:**
    *   For each color channel (e.g., R, G, B), the Cumulative Distribution Function (CDF) will be computed. This can be represented as a vector $\mathbf{c}_k \in \mathbb{R}^N$, where $N$ is the number of bins (e.g., 256) and $k \in \{R, G, B\}$.
    *   These vectors capture the global intensity distribution for each channel.
    *   *Rationale:* Provides a robust, quantifiable measure of global tonal and color balance, which is a fundamental aspect of image editing.

2.  **Convolutional Features:**
    *   The input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ will be passed through a pre-trained or custom CNN encoder.
    *   The output of the encoder will be a set of feature maps $\mathbf{F} \in \mathbb{R}^{H' \times W' \times D}$, or a flattened feature vector $\mathbf{f} \in \mathbb{R}^M$.
    *   *Rationale:* Extracts spatially-aware, hierarchical features that encode local textures, structures, and contextual information critical for stylistic edits.

3.  **Combined Feature Vector:**
    *   The CDF feature vectors and the flattened convolutional feature vector will be concatenated to form a comprehensive feature representation $\mathbf{x} = [\mathbf{c}_R, \mathbf{c}_G, \mathbf{c}_B, \mathbf{f}]$.
    *   *Rationale:* This combined vector provides a holistic description of the image, encompassing both its global statistical properties and its local spatial characteristics.

### 3.2 Learning the Transformation

A regression network (e.g., a Multi-Layer Perceptron - MLP) will be trained to map the combined feature vector $\mathbf{x}$ to the parameters of the desired image transformation.

1.  **Input:** The combined feature vector $\mathbf{x}$.
2.  **Output:** The output of the regression network will be the parameters that define the image transformation. We will explore two primary output mechanisms:

    *   **Option A: Parametric CDF Transformation:**
        *   The network predicts parameters for a flexible function (e.g., a piecewise linear function, a polynomial, or a spline) that maps the input CDFs to the target CDFs.
        *   For example, it could predict the control points for a B-spline that defines the tone curve for each channel.
        *   *Rationale:* Directly manipulates the CDFs, maintaining the original project's core idea while allowing for more complex, learned mappings.

    *   **Option B: Learned Look-Up Tables (LUTs):**
        *   The network predicts the entries of a 1D or 3D LUT.
        *   For a 1D LUT, the output would be $3 \times N_{LUT}$ values (e.g., $3 \times 256$ for RGB channels).
        *   For a 3D LUT, the output would be $N_{grid}^3 \times 3$ values (e.g., $32^3 \times 3$ for a $32 \times 32 \times 32$ grid).
        *   *Rationale:* LUTs are an industry-standard, highly efficient, and expressive way to represent complex color and tone transformations. By *learning* the LUTs, we overcome the "hard-coded" limitation, allowing the model to generate custom LUTs tailored to the input image and desired style. This is particularly powerful for achieving "film" or "cyberpunk" looks.

### 3.3 Loss Function

The loss function will guide the training of the regression network. It will measure the discrepancy between the transformed image (or its characteristics) and the target styled image. Potential components include:
*   **Pixel-wise Loss:** Mean Squared Error (MSE) or Mean Absolute Error (MAE) between the transformed image and the ground truth styled image.
*   **Perceptual Loss:** Using features from a pre-trained CNN (e.g., VGG) to compare the perceptual similarity between the transformed and target images.
*   **CDF Loss:** A loss component that directly compares the CDFs of the transformed image with the target CDFs.
*   **Adversarial Loss (Optional):** If using a Generative Adversarial Network (GAN) setup, a discriminator loss could encourage the generated images to look realistic and match the target style distribution.

### 3.4 Continued Relevance of CDF Analysis

Despite the introduction of Convolutional Neural Networks (CNNs) and the use of Learned Look-Up Tables (LUTs), Cumulative Distribution Function (CDF) analysis remains a valuable component of our augmented architecture. Its continued inclusion is justified by several factors:

1.  **Complementary Information:**
    *   **CNNs** excel at extracting **spatial features**, textures, and local patterns, providing answers to "what objects are present?" and "where are they located?".
    *   **CDFs** provide robust **global statistical information** about the overall tonal and color distribution, addressing questions like "what is the overall brightness?", "how much contrast is there?", and "what is the dominant color cast?".
    *   These two types of features are complementary. Combining them offers a more complete and nuanced understanding of an image, allowing the model to leverage both local detail and global statistical properties.

2.  **Interpretability and Control:**
    *   CDFs offer a more direct and interpretable link to global image adjustments. Many "film" looks involve specific global tonal shifts (e.g., crushed blacks, lifted shadows, specific color casts) that are well-captured by CDFs.
    *   If the model learns to predict parameters that directly influence CDFs, it can provide a more intuitive understanding of *why* a certain global edit was made or allow for fine-tuning specific aspects of the global tone.

3.  **Robustness to Content:**
    *   Global CDFs are less sensitive to the specific content of an image than CNN features. This can be advantageous for learning general stylistic transformations that should apply consistently across diverse scenes, regardless of the objects present. This helps in achieving a consistent "film look" across various photographs.

4.  **Guiding Global Adjustments:**
    *   Many film looks are characterized by specific global tone curves and color shifts. CDFs can serve as a strong, explicit signal for these global adjustments, allowing the CNN to focus its learning capacity on more localized or textural aspects (e.g., grain, subtle vignetting, specific local contrast).

In summary, explicitly providing CDFs as input can improve learning efficiency, enhance robustness for global tonal and color transformations, and potentially aid interpretability by giving the model direct, pre-computed global statistical cues.

## 4. Training Data Strategy

To train this augmented model, a robust dataset of "before" and "after" image pairs for specific styles is crucial. We will focus on the **"film"** style initially.

1.  **Selection of Target Style:** We will begin with the **"film"** style, which encompasses a range of characteristics such as specific tone curves, color shifts, grain, and vignetting, often emulating classic film stocks.

2.  **Synthetic Data Generation Process (for "Film" Style):**
    To avoid manual photo editing for each image, we will leverage programmatic application of film emulation techniques. This involves using image processing libraries to apply a consistent set of transformations that mimic the characteristics of various film stocks.

    *   **Source Images:** Gather a large, diverse collection of high-quality original (un-edited) images. Public datasets like ImageNet, COCO, or Flickr-100M (if accessible and licensed appropriately) can serve as excellent starting points.
    *   **Programmatic Styling:** Instead of manual editing, we will use Python libraries (e.g., `OpenCV`, `scikit-image`, `Pillow`, `colour-science`, `colour-demosaicing`) to apply a consistent set of transformations. This can include:
        *   **Tone Curve Application:** Implement custom tone curves (e.g., S-curves, logarithmic curves) that mimic film response. This can be done by mapping input pixel values to output pixel values using a 1D LUT or a mathematical function.
        *   **Color Matrix/LUT Application:** Apply 3x3 color matrices or 3D LUTs that represent specific color shifts and grading associated with film stocks. Many film emulation LUTs are available online (e.g., `.cube`, `.3dl` files) and can be applied programmatically.
        *   **Grain Simulation:** Generate and overlay realistic film grain. This often involves adding Gaussian noise with specific frequency characteristics, potentially varying with luminance.
        *   **Vignetting:** Programmatically apply a radial darkening effect towards the image corners.
        *   **Desaturation/Hue Shifts:** Adjust global or selective saturation and hue to match film aesthetics.
    *   **Parameter Variation:** To create a diverse dataset for a single "film" style, we can introduce slight variations in the parameters of these programmatic transformations (e.g., varying grain intensity, vignette strength, or subtle shifts in color grading parameters). This helps the model learn the *essence* of the style rather than just a single instance.
    *   **Paired Dataset:** The result will be a dataset of $(\mathbf{I}_{original}, \mathbf{I}_{styled})$ pairs, where $\mathbf{I}_{styled}$ is generated programmatically from $\mathbf{I}_{original}$.

    **Conceptual Example (Programmatic "Film" Style):**

    Let's consider an `original_image.jpeg` (e.g., a vibrant digital photo of a forest).

    **Desired "Film" Style: "Vintage Warm Film"**

    **Programmatic Edits Applied (using Python libraries):**

    1.  **Load Image:** `img = cv2.imread('original_image.jpeg')`
    2.  **Apply Tone Curve:** `img_toned = apply_tone_curve(img, curve_params_for_vintage_warm)` (where `apply_tone_curve` is a custom function implementing a specific 1D LUT or mathematical function).
    3.  **Apply Color Grading (3D LUT):** `img_color_graded = apply_3d_lut(img_toned, vintage_warm_film_3d_lut_data)` (using a pre-defined or generated 3D LUT).
    4.  **Add Grain:** `img_grained = add_film_grain(img_color_graded, grain_intensity=0.05, grain_size=2)` (using a custom function to add noise).
    5.  **Apply Vignette:** `img_vignetted = apply_vignette(img_grained, strength=0.3, feather=0.7)` (using a custom function for radial darkening).
    6.  **Save Styled Image:** `cv2.imwrite('original_image_vintage_warm_film.jpeg', img_vignetted)`

    This process generates the `(original_image, styled_image)` pair without manual intervention, allowing for large-scale dataset creation.

## 5. Project Structure

The project is organized to facilitate a smooth workflow from data generation to model development, with a strong emphasis on clarity, modularity, and the visualization of intermediate steps.

```
.
├───.git/
├───.idea/
├───docs/
│   └───roadmap.md                # Detailed project roadmap, rationale, and technical specifications.
├───images/
│   ├───original/                 # Stores all original, unedited images. These serve as the source for data generation.
│   │   └───... (e.g., image1.jpg, image2.png)
│   ├───test_images/              # Contains specific images used for testing and debugging core image processing and visualization functions.
│   │   ├───climbing_test_original.jpeg
│   │   ├──... (other test images)
│   │   ├───color_decompositions/ # Stores visualized intermediate steps of color channel separation.
│   │   └───color_histograms/     # Stores visualized intermediate steps of color intensity histograms and CDFs.
│   ├───styled/                   # NEW: Directory for programmatically generated styled images.
│   │   ├───film_vintage_warm/    # Subdirectory for a specific film style (e.g., Kodak Portra emulation).
│   │   │   └───... (e.g., image1_styled.jpg, image2_styled.png)
│   │   ├───film_cool_tone/       # Another film style (e.g., Fuji Superia emulation).
│   │   │   └───... (other styled images)
│   │   └───cyberpunk_neon/       # Future styles will have their own subdirectories here.
│   └───edited/                   # Existing: (Currently empty. Its purpose might merge with 'styled' later, or be reserved for manual edits.)
├───data_generation/              # NEW: Dedicated module for data generation scripts and assets.
│   ├───generate_film_data.py     # Main script to generate film-styled images programmatically.
│   ├───film_presets/             # Stores film emulation assets (LUTs, tone curves, grain textures) for configurable styling.
│   │   ├───vintage_warm_lut.cube # Example 3D LUT file for a specific film look.
│   │   ├───cool_tone_curve.json  # Example tone curve definition (e.g., JSON or CSV).
│   │   └───grain_texture_1.png   # Example grain texture or noise pattern.
│   └───utils.py                  # Helper functions for image manipulation (e.g., apply_lut, add_grain, apply_vignette).
├───models/                       # NEW: Future directory for storing trained neural network models.
│   └───... (trained model files)
├───color_settings.py             # Defines RGB color settings and colormaps for visualization.
├───load_and_show.py              # Contains functions for loading, displaying, and visualizing image properties (e.g., RGB separation, histograms).
├───main.py                       # Main entry point for the application (currently a placeholder).
├───misc_funcs.py                 # Miscellaneous utility functions.
├───README.md                     # This file.
└───test.py                       # Unit tests for various components.
```

## 6. Next Steps

1.  **Data Generation Implementation:** Begin the process of creating the synthetic "before" and "after" dataset for the "film" style using the programmatic approach outlined in Section 4. Emphasize the visualization of intermediate steps during this process.
2.  **CNN Encoder Selection:** Research and select an appropriate CNN architecture (pre-trained or custom) for feature extraction.
3.  **Model Prototyping:** Start implementing the combined feature extraction and regression network, focusing on either parametric CDF transformation or learned LUTs as the output mechanism.
