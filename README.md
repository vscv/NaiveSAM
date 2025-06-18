# NaiveSAM
A Jupyter-based Interactive Workflow for YOLO Segmentation dataset with Segment Anything Model


## Requirements 
* Python 3
* FFmpeg 4: sudo apt update; sudo apt install ffmpeg (https://ffmpeg.org/)
* Jupyter notebook 6
* Jupyter ipywidgets 8
* OpenCV 4
* SAM2: SAM2 v2.1 (check htps://github.com/facebookresearch/sam2)




## Note
* 如果cuda OOM主要是GPU記憶體用盡，請減少影格數量如減少影片長度或是減少fps數，來避免。
* 標記的點越多(標記的物件越多)，SAM處理會越慢。(如7件1.3fps 3件2.0fps)
* 影片目標盡量不要在畫面中有中斷，雖然SAM2追蹤效能很好，但有些斷點之後會追蹤失敗或標籤混淆，需要在這些短點補標註使時間成本提高。
* SAM2 progtgation 與 Display reuslt時，可以選擇影格間隔數，以快速看到成果。畢竟每秒大約1.5frame的速度仍是太慢。
* SAM2 hydra issue:
```Python
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt" # if install sam2 in home/ place cpt to sam2.
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # if install sam2 in home/ set this way!

# ref to https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb

這是由於SAM本身使用hydra的怪事，cpt可以用路徑，但cfg就一定要用hydra的預先已知configs的套件安裝路徑模式

MissingConfigException: Cannot find primary config './sam2/configs/sam2.1/sam2.1_hiera_l.yaml'. Check that it's in your config search path.

Config search path:
	provider=hydra, path=pkg://hydra.conf
	provider=main, path=pkg://sam2
	provider=schema, path=structured://
    
    
cpt放sam2安裝的專案位置，全路徑或相對路徑能找到，conf則是用hydra來找。因此統一放在/home下的sam2安裝位置即可。
"""


```
* 目前SAM2並沒有支援多GPU，但可分段多支影片來並行推論。
* 這些問題時請reconnect kernel or re-run cell，若還是無法解決請清除並重新啟動kernel：NaveSAM介面沒有出現影格，影格停滯與Frame數無法同步。
