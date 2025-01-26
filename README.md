# ECgMLP
An automated Endometrial Cancer Diagnosis approach using Histopathological Image through image Preprocessing and Optimized gated Multi-Layer Perceptron Model

# Libraries for Image Preprocessing

### 1. **OpenCV**
   - Image processing tasks like normalization and alpha-beta transformation.

### 2. **NumPy**
   - Numerical operations and array manipulations.

### 3. **Matplotlib**
   - Visualization of images.

### 4. **Scikit-Image**
   - Advanced image processing tools, especially for NLM denoising.

### 5. **TensorFlow**
   - Machine learning framework, required version: 2.12.0.

### 6. **TensorFlow-Addons**
   - Additional TensorFlow functionalities, required version: 0.20.0.

## Additional Imports
```python
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
```
**Kindly cite the article and code if you use it:**

[![DOI](https://zenodo.org/badge/910827354.svg)](https://doi.org/10.5281/zenodo.14743245)
