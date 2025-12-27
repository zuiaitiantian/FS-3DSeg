# FS-3DSeg
## Prerequisites
```
Python 3.8+
pytorch 1.11.0
CUDA 11.3
NVIDIA GPU with Compute Capability ≥ sm_80 (RTX 3090 recommended)
```

## Running
Installation and data preparation please follow [AttMPTI](https://github.com/Na-Z/attMPTI "").
```
pip install h5py transforms3d
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11.0%2Bcu113.html
pip install torch-geometric==2.0.4
```

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
