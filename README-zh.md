# NaiveSAM

`基于Jupyter的互动式YOLO Segmentation资料集标注流程，整合Segment Anything Model`

[繁體中文](README-zh-TW.md) [简体中文](README-zh.md) [English](README.md)

---

## 如何使用

只需开启 `NaiveSAM.ipynb` 并依照程式码逐步执行，即可开始标注流程。

---

## 系统需求

- Python 3
- FFmpeg 4
```bash
sudo apt update
sudo apt install -y ffmpeg
```
- Jupyter notebook 6
- Jupyter ipywidgets 8
- OpenCV 4
- Segment Anything Model 2 (SAM2 v2.1)
- 参考：[https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)

> **注意**：请先安装 CUDA 驱动、Torch、FFmpeg 与 SAM2。

---

## 展示影片与操作截图

- 标注介面： [标注 UI 展示](https://github.com/user-attachments/assets/1345436b-0d57-4b72-9e9d-fe161b5efe08)

- 端到端流程（影格撷取 → 标注 → 汇出 YOLO 资料集 → 训练）： [End-to-End 展示](https://github.com/user-attachments/assets/1345436b-0d57-4b72-9e9d-fe161b5efe0)

---

## 常见问题与提示

- **CUDA 记忆体不足 (OOM)**：请减少影格数量，或降低 FPS，以减少 GPU 压力。

- **SAM 处理速度变慢**：若标注点数过多，处理速度将降低（例如：7 个物件约 1.3fps，3 个物件约 2.0fps）。

- **影片物件中断追踪问题**：目标若于画面中消失再重现，可能会导致 SAM2 无法正确追踪，建议于中断前后补充标注。

- **SAM2 预测与结果检视速度**：可调整影格间隔数来加快显示速度（SAM2 约为 1.5fps）。

- **SAM2 尚未支援多 GPU**：可将影片分段并平行处理来改善效能。

- **Kernel 无反应或影格未显示**：请重新连接 kernel 或使用 Kernel > Restart & Clear Output。

- **SAM2 Hydra 设定问题：**

```python
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

> Hydra 的设定档需使用安装时指定的结构位置。 `checkpoint` 可用相对路径，但 `config` 需使用 Hydra 预设搜寻路径。

请参考：[https://github.com/facebookresearch/sam2/blob/main/notebooks/video\_predictor\_example.ipynb](https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)

---

## 授权

本专案依据 [MIT License](https://opensource.org/licenses/MIT) 授权。
