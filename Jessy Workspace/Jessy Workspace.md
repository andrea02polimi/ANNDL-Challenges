# Insights

From *BEPH (BEiT-based model Pre-training on Histopathological image)* paper.

- **Preprocessing**
    - Tissue Segmentation - Otsu threshold on saturation channel: Apply this before cropping to eliminate blank areas in your crops.
    - Reject crops with too little tissue before training your classifier.
    - Local region sampling: For the regressor, random crops around the tissue -- Force crops around tumor mass (based on mask or regressor heatmap).
- Better **Feature Engineering** (PCA, UMAP) to inspect feature separability/detect overlap/decide on embedding: 
    - Apply PCA/UMAP to ConvNeXt features (penultimate layer): Extract the 768-dimensional feature vector (the output of the layer before your final classifier) for 1,000 random images. Run UMAP on these vectors.
        - If you see distinct clusters: Your model has learned well.
        - If classes are mixed: Your model is confused (or the features aren't strong enough).
        - If you see a small, isolated island: Those are likely your "Shrek" images or artifacts.
    - Verify if 4 classes are separable → If not:
        - Stronger augmentation
        - Better crop strategy
        - Pretrained pathology-specific encoder (e.g., CTransPath / UNI / BEPH encoder)
- **Architecture Recommendations**: ImageNet pretraining performs poorly in pathology tasks. ConvNeXt is good only if pretrained on pathology data, otherwise suboptimal. Better alternatives:
    - BEiT v2 (ImageNet → UNLABELED pathology patches)
    - CTransPath
    - UNI (2024)
    - CLAM backbones
- **Improved Augmentation**: Need to add histopathology-specific augmentations:
    - H&E **stain normalization**: Macenko / Reinhard normalization to reduce scanner variability.
    - Random color deconvolution
    - Gaussian blur / sharpen
    - Elastic deformation



## Other things to do:

- TA's advice

    | Component            | Old   | New               |
    | -------------------- | ----- | ----------------- |
    | Classifier Optimizer | AdamW | **Lion**          |
    | Regressor Optimizer  | Adam  | **Ranger**        |
    | LR Schedule          | None  | **Cosine LR**     |
    | LR                   | 1e-3  | **1e-4 (Lion)**   |
    | Reg LR               | 1e-3  | **3e-4 (Ranger)** |
