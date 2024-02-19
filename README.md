# BasicSTISR

BasicSTISR (Basic Scene Text Image Super Resolution) 是一个基于 PyTorch 的开源场景文本图像超分辨率工具箱.

## Prepare Datasets

In this work, we use STISR datasets TextZoom and four STR benchmarks, i.e., ICDAR2015, CUTE80, SVT and SVTP for model comparison. 

All the datasets are `lmdb` format.  One can download these datasets from the this [link](https://drive.google.com/drive/folders/1uqr8WIEM2xRs-K6I9KxtOdjcSoDWqJNJ?usp=share_link). 
```
./datasets/TextZoom/
    --test
    --train1
    --train2
```

**NOTE**: Please do not forget to accustom your own dataset path in `config/super_resolution.yaml` ,  such as the parameter `train_data_dir` and `val_data_dir`.

## Prepare Pretrain Text Recognizers

Following previous STISR works, we also use [CRNN](https://github.com/meijieru/crnn.pytorch), [MORAN](https://github.com/Canjie-Luo/MORAN_v2  ) and [ASTER](https://github.com/ayumiymk/aster.pytorch) as the downstream text recognizer.  
```
.pretrained/
    --aster.pth.tar
    --crnn.pth
    --moran.pth
```
## Text Gestalt(TG)

1. Download the pre-trained weights and logs at [BaiduYunDisk](https://pan.baidu.com/share/init?surl=c0DqmKkw5_uB6njPhmm-2g) with password: vqg7

2. Download the pretrain_transformer_stroke_decomposition.pth at [BaiduYunDisk](https://pan.baidu.com/s/1MeFKnF5tWiL7ts00SHLM2A#list/path=%2F) with password: mteb
```
./dataset/
    --charset_36.txt 
    --confuse.pkl
    --english_decomposition.txt
    --pretrain_transformer_stroke_decomposition.pth
```

## How to Run?

We have set some default hype-parameters in the `config/super_resolution.yaml` and `main.py`, so you can directly implement training and testing after you modify the path of datasets and pre-trained model.  

### Training

```
sh train.sh
```

### Testing

```
sh test.sh
```

## Acknowledgement

The code of this work is based on [TBSRN](https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope), [TG](https://github.com/FudanVI/FudanOCR/text-gestalt), [TATT](https://github.com/mjq11302010044/TATT), [C3-STISR](https://github.com/JingyeChen/C3-STISR), and [LEMMA](https://github.com/etodd/Lemma). Thanks for your contributions.
