# NaiveSAM
*A Jupyter-based Interactive Workflow for YOLO Segmentation Dataset with Segment Anything Model (SAM)*


[繁體中文](README-zh-TW.md) [简体中文](README-zh.md) [English](README.md)

---

## 📌 How to Use
Simply open and run the `NaiveSAM.ipynb` notebook in your local Jupyter Notebook environment.

---

## ⚙️ Requirements
- Python 3
- FFmpeg 4
    ```bash
    sudo apt update
    sudo apt install -y ffmpeg
    ```
- Jupyter Notebook 6+
- `ipywidgets` 8
- OpenCV 4
- SAM2: Version 2.1  
  (Install from: https://github.com/facebookresearch/sam2)

> ⚠️ **Note:** Please make sure to pre-install CUDA drivers, PyTorch, FFmpeg, and the SAM2 environment.

---

## 🖼️ Demo Screenshots
- **Annotation Interface Overview**
  
  ![Annotation UI](https://github.com/user-attachments/assets/1345436b-0d57-4b72-9e9d-fe161b5efe08)

- **End-to-End Workflow:** Frame Extraction → Annotation → Dataset Generation → Model Training
  
  ![Full Pipeline](https://github.com/user-attachments/assets/1345436b-0d57-4b72-9e9d-fe161b5efe0)

---

## ❓ Troubleshooting & Tips
<details>
<summary>Click to Expand</summary>

- **CUDA Out of Memory (OOM):**
  - Reduce the total number of frames (shorten video or reduce FPS).
- **SAM Slow Performance:**
  - More labeled points or objects = slower SAM processing. E.g., 7 objects ≈ 1.3 FPS; 3 objects ≈ 2.0 FPS.
- **Object Visibility in Frames:**
  - Avoid abrupt occlusion; although SAM2 has good tracking, failure may occur at visual discontinuities.
- **Preview Faster:**
  - Adjust frame interval in "SAM2 propagation" and "Display result" to skip frames for quicker mask review.
- **No Multi-GPU Support:**
  - SAM2 only supports single-GPU, but you can process multiple videos in parallel manually.
- **Kernel Errors or UI Freezes:**
  - Try `Restart & Run All` or reconnect kernel.
  - If the UI or frame sync is broken, reset the kernel.

### SAM2 + Hydra Path Issue
```python
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
```
> Note: SAM2 uses Hydra for config loading. Checkpoint paths work with standard paths, but `cfg` must use Hydra's internal search path structure. Place both in your installed `/home/.../sam2/` directory.

Typical Hydra Error:
```
MissingConfigException: Cannot find primary config './sam2/configs/sam2.1/sam2.1_hiera_l.yaml'.
```
</details>
