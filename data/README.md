# Dataset Setup

This project uses the **CelebA dataset** for synthetic face image generation.

The full dataset is **not included in this GitHub repository** because it is large and contains real face images. The dataset should be downloaded locally before running the preprocessing, training, or evaluation scripts.

---

## Dataset Used

**Dataset:** CelebA  
**Task:** Face image generation  
**Use in this project:** Training and comparing VAE, DCGAN, and diffusion models for synthetic face generation.

CelebA includes aligned celebrity face images and additional metadata such as facial attributes, bounding boxes, landmarks, and train/validation/test split information.

---

## Expected Local Folder Structure

After downloading and extracting CelebA, the local dataset should follow this structure:

```text
data/
  raw/
    celeba/
      img_align_celeba/
        000001.jpg
        000002.jpg
        000003.jpg
        ...
      list_attr_celeba.txt
      list_bbox_celeba.txt
      list_eval_partition.txt
      list_landmarks_align_celeba.txt
```