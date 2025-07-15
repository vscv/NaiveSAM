# End2End_NaiveSAM+YOLO-seg-Training
An example of how to run the whole pipeline from dataset creation to model training.

[End2End_NaiveSAM_YOLOsegTraining.ipynb](End2End_NaiveSAM_YOLOsegTraining.ipynb)

* * *
## YouTube overview video
[![YouTube Overview Video](https://img.youtube.com/vi/_qAV8T3QOYk/maxresdefault.jpg)](https://www.youtube.com/watch?v=_qAV8T3QOYk)




* * *
## Ultralytics

```bash
pip install ultralytics

# If some packages are incompatible.
pip install "numpy<2"
pip install "pandas<2.2"
```

## YOLO training

if  `UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.`

```Python
results = model.train(
    ...,
    workers=4,  # 設定 worker 數量
)
```
