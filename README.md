# MobileMold: Smartphone-Based Microscopy for Food Mold Detection
A smartphone-microsope-based dataset with 4941 annotated images for food mold detection

## 🌟 About MobileMold

**MobileMold** is a comprehensive dataset comprising **4,941 annotated images** for food mold detection, captured using smartphones with various clip-on microscope attachments.
The dataset addresses the growing need for accessible, low-cost food safety monitoring by leveraging smartphone-based microscopy. This enables research and development in computer vision applications for mold detection on various food surfaces.

---

### 📊 Dataset Overview
- **Total Images:** 4,941
- **Annotations:** Food Type and Mold Label
- **Food Types:** 11 categories (carrot, orange, creamcheese, tomato, toast, raspberry, mixed bread, blackbery, blueberry, cheese, onion)
- **Microscope Types:** 3 different clip-on smartphone microscopes (30x-100x magnification)
- **Smartphones:** Images captured with 3 different smartphone models
---

## 📢 Data Release

You can download the full dataset here:

---

### 📁 Dataset Structure
  ```
MobileMold/
├── metadata.csv # Complete dataset metadata (4,941 entries)
├── train_metadata.csv # Training split metadata
├── val_metadata.csv # Validation split metadata
├── test_metadata.csv # Test split metadata
├── original/ # Original microscope images (as captured)
│ ├── L10 - 48.jpeg
│ ├── L10 - 25.jpeg
│ ├── L10 - 161.jpeg
│ └── ... (4,941 files total)
└── cropped_resized/ # Preprocessed images (same filenames)
├── L10 - 48.jpeg # Cropped to mold region & resized
├── L10 - 25.jpeg
├── L10 - 161.jpeg
└── ... (4,941 files, 1:1 mapping to original/)
  ```
---
### 📊 Dataset Composition

### Image Versions
1. **`original/`** - Raw images as captured by smartphone microscopes
   - Various resolutions (depending on smartphone and microscope)
   - Full field-of-view including background
   - Unprocessed image data

2. **`cropped_resized/`** - Processed images
   - Cropped to focus on mold regions
   - Resized to consistent dimensions
   - Same filenames as original folder

### Metadata Format
Each CSV file contains the following columns:

| Column | Description | Values/Examples |
|--------|-------------|-----------------|
| `filename` | Image filename (same in both folders) | `L10 - 48.jpeg` |
| `mold` | Binary indicator of mold presence | `True` / `False` |
| `food` | Type of food in image | `carrot`, `bread`, `cheese`, `tomato`, etc. |
| `phone` | Smartphone model used | `iPhone SE 2nd Generation`, etc. |
| `microscope` | Clip-on microscope model | `Apexel 100x`, etc. |

**Example metadata entry:**
```csv
filename,mold,food,phone,microscope
L10 - 48.jpeg,True,carrot,iPhone SE 2nd Generation,Apexel 100x
```
## 📄 License

This dataset is available under the terms of the **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**
