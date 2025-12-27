# FS-3DSeg
## Running
Installation and data preparation please follow [AttMPTI](https://github.com/Na-Z/attMPTI "").

### Training
Pretrain the segmentor which includes feature extractor module on the available training set:
```
bash scripts/pretrain_segmentor.sh
```
Train our method:
```
bash scripts/train.sh
```

### Evaluation
Test our method:
```
bash scripts/eval.sh
```

## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch "") and [AttMPTI](https://github.com/Na-Z/attMPTI "") for sharing their source code.
