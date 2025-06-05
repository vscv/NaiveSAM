# NaiveSAM
A Jupyter-based Interactive Workflow for YOLO Segmentation dataset with Segment Anything Model


## Tool kits
* Python 3
* FFmpeg 4: sudo apt update; sudo apt install ffmpeg (https://ffmpeg.org/)
* Jupyter notebook 6
* Jupyter ipywidgets 8
* OpenCV 4
* SAM2: SAM2 v2.1 (check htps://github.com/facebookresearch/sam2)




## Note
* 如果cuda OOM主要是GPU記憶體用盡，請減少影格數量如減少影片長度或是減少fps數，來避免。
* 影片目標盡量不要在畫面中有中斷，雖然SAM2追蹤效能很好，但有些斷點之後會追蹤失敗或標籤混淆，需要在這些短點補標註使時間成本提高。
* SAM2 progtgation 與 Display reuslt時，可以選擇影格間隔數，以快速看到成果。畢竟每秒大約1.5frame的速度仍是太慢。
