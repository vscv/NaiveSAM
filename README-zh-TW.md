# NaiveSAM

`基於Jupyter的互動式YOLO Segmentation資料集標註流程，整合Segment Anything Model`

[繁體中文](README-zh-TW.md) [简体中文](README-zh.md) [English](README.md)

---

## 如何使用

只需開啟 `NaiveSAM.ipynb` 並依照程式碼逐步執行，即可開始標註流程。

---

## 系統需求

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
  - 參考：[https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)

> **注意**：請先安裝 CUDA 驅動、Torch、FFmpeg 與 SAM2。

---

## 展示影片與操作截圖

- 標註介面： [標註 UI 展示](https://github.com/user-attachments/assets/1345436b-0d57-4b72-9e9d-fe161b5efe08)

- 端到端流程（影格擷取 → 標註 → 匯出 YOLO 資料集 → 訓練）： [End-to-End 展示](https://github.com/user-attachments/assets/1345436b-0d57-4b72-9e9d-fe161b5efe0)

---

## 常見問題與提示

- **CUDA 記憶體不足 (OOM)**：請減少影格數量，或降低 FPS，以減少 GPU 壓力。

- **SAM 處理速度變慢**：若標註點數過多，處理速度將降低（例如：7 個物件約 1.3fps，3 個物件約 2.0fps）。

- **影片物件中斷追蹤問題**：目標若於畫面中消失再重現，可能會導致 SAM2 無法正確追蹤，建議於中斷前後補充標註。

- **SAM2 預測與結果檢視速度**：可調整影格間隔數來加快顯示速度（SAM2 約為 1.5fps）。

- **SAM2 尚未支援多 GPU**：可將影片分段並平行處理來改善效能。

- **Kernel 無反應或影格未顯示**：請重新連接 kernel 或使用 Kernel > Restart & Clear Output。

- **SAM2 Hydra 設定問題：**

```python
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

> Hydra 的設定檔需使用安裝時指定的結構位置。 `checkpoint` 可用相對路徑，但 `config` 需使用 Hydra 預設搜尋路徑。

請參考：[https://github.com/facebookresearch/sam2/blob/main/notebooks/video\_predictor\_example.ipynb](https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)

---

## 授權

本專案依據 [MIT License](https://opensource.org/licenses/MIT) 授權。

