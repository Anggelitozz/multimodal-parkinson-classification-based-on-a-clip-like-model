# multimodal-parkinson-classification-based-on-a-clip-like-model
This repository contains the implementation of the research project titled:: Parkinson stratification from a multimodal contrastive model merging scale-reports and segmentation Foundational findings

## 📌 Project Overview

This repository contains the implementation of the research project:

**"Supporting diagnostic classification of Parkinson’s disease through radiological findings and clinical assessment using a large-scale learning model."**

The objective of this work is to classify Parkinson’s disease severity levels using a multimodal approach that integrates:

- 3D radiological volumes (MRI and SPECT)
- Paired clinical tabular data

Radiological volumes are processed using **SAM-Med3D** to extract volumetric feature embeddings, while clinical data is transformed into structured text and embedded using **ClinicalBERT**. The extracted embeddings are aligned through a CLIP-like architecture trained with **Supervised Contrastive Learning (SCL)**.


![pipeline](assets/pipeline.png)


The pipeline consists of the following stages:

1. Radiological preprocessing (MRI and SPECT)
2. Clinical data preprocessing and template-based text generation
3. Feature extraction:
   - Radiological embeddings using SAM-Med3D
   - Clinical embeddings using ClinicalBERT
4. Supervised contrastive training for severity classification


## 🔨 Usage

The project is executed in the following order:

### 1 Radiological Data Preprocessing
- MRI preprocessing: `preprocessing/preprocessingv1.ipynb`
- SPECT preprocessing: `preprocessing/spectPreprocessingv1.ipynb`

### 2 Clinical Data Preprocessing
- `features_extraction/clinical_data_preprocessing.ipynb`

### 3 Radiological Embedding Extraction
- Implemented using the SAM-Med3D repository:
  `features_extraction/SAM-Med3D/`

### 4 Clinical Embedding Extraction
- `features_extraction/clinical_data_preprocessing_embeddings_extraction.ipynb`
- Uses ClinicalBERT with template-based structured prompts.

### 5 Multimodal Classification (CLIP-like Model)
- `clip-like_model/clip_from_scratch.ipynb`
- Trained using Supervised Contrastive Learning.
- Executed inside Docker due to PyTorch dependency constraints.
