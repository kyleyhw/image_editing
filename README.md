# Image Editing Project: CDF-based Transformations with Deep Learning Augmentation

## Project Overview

This project explores a novel, mathematically-grounded approach to image editing. The core idea is to define image transformations based on the manipulation of Cumulative Distribution Functions (CDFs) of pixel intensities. To overcome the inherent limitations of purely statistical methods (lack of spatial awareness) and to achieve complex, stylistic edits (e.g., "film," "cyberpunk"), we augment this CDF analysis with deep learning techniques, specifically Convolutional Neural Networks (CNNs) and learned Look-Up Tables (LUTs).

The ultimate goal is to create a system that can apply sophisticated, style-specific edits to images programmatically, moving beyond traditional parameter-based adjustments to a more data-driven and perceptually relevant approach.

## Project Structure

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

## Emphasis on Intermediate Steps and Visualization

Throughout the project, a strong emphasis is placed on understanding and visualizing intermediate steps. This is crucial for debugging, verifying mathematical principles, and gaining insights into the transformations being applied.

*   **Data Generation:** The `data_generation` process will be designed to allow for the visualization of each applied transformation (e.g., showing the image after tone curve, then after color grading, then after grain, etc.). This ensures that the programmatic styling accurately reflects the desired "film" aesthetic. The `images/styled/` directories will not only store the final styled images but can also be configured to save intermediate outputs for analysis.
*   **Feature Engineering:** When extracting CDFs and CNN features, tools will be developed to visualize these features. For CDFs, this means plotting the curves. For CNN features, this could involve visualizing feature maps or using dimensionality reduction techniques to plot feature vectors.
*   **Model Development:** During model training, the output of the learned LUTs or transformation parameters will be visualized. This includes plotting the learned tone curves, inspecting the generated 3D LUTs, and comparing the CDFs of the model's output with the target CDFs.
*   **`load_and_show.py`**: This existing module already demonstrates the importance of visualization by providing functions to show RGB separations and histograms, which are fundamental to the CDF analysis. Its capabilities will be extended as needed to support new visualizations.

This commitment to visualization ensures that the complex mathematical and deep learning processes are transparent and understandable at every stage of development.