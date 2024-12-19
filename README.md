# EGMS-Net
Pytorch codes of **Multi-task Siamese Network Guided by Enhanced Change Information for Semantic Change Detection in Bi-Temporal Remote Sensing Images** [[paper](https://ieeexplore.ieee.org/document/10737132)]

![Fig2](https://github.com/user-attachments/assets/ec1da70b-d65e-45d3-a941-6dc7ebfeb53e)

## Dataset Download

In the following, we summarize the processed semantic change detection data set used in this paper:

* [SECOND (Baidu)](https://pan.baidu.com/s/1RFhlO9_1KaFcIdTqblJIbA?pwd=dn84)
* [Landsat-SCD (Baidu)](https://pan.baidu.com/s/17tcdFBgMfx6d_BOsX6lzbw?pwd=hy5w)
* [HiUCD-mini (Baidu)](https://pan.baidu.com/s/1bPq3nYO0VRtnKET_ce6n6Q?pwd=znfp)

## How to Use

1. Dataset preparation.
   * Please split the data into training, validation and test sets and organize them as follows:
```
      YOUR_DATA_DIR
      ├── ...
      ├── train
      │   ├── A
      │   ├── B
      │   ├── labelA
      │   ├── labelB
      ├── val
      │   ├── A
      │   ├── B
      │   ├── labelA
      │   ├── labelB
      ├── test
      │   ├── A
      │   ├── B
      │   ├── labelA
      │   ├── labelB
```

2. Change the root path of the data set.
   * Take the SECOND data set as an example.
   * Find line 34 in [EGMS-Net/train_SECOND.py](https://github.com/IceStreams/EGMS-Net/blob/main/train_SECOND.py), change `--data_root` to your local dataset directory.
   * Find line 34 and line 35 in [EGMS-Net/inference_SECOND.py](https://github.com/IceStreams/EGMS-Net/blob/main/inference_SECOND.py), change `--data_root` to your local dataset directory, and change `--load_from` to your checkpoint path.

3. Training
   
   bash my_run.sh

5. Inference and evaluation
   
   bash my_inference.sh

## Pretrained Models

The reproducible weights of EGMS-Net on the three benchmark datasets are visible in the links below:

[Baidu](https://pan.baidu.com/s/1NGFO1pVdEHfOF4YGOvDfmA?pwd=h9kp)

## Cite EGMS-Net

If you find this work useful or interesting, please consider citing the following BibTeX entry.

```
@ARTICLE{10737132,
  author={Zuo, Xibing and Jin, Fei and Ding, Lei and Wang, Shuxiang and Lin, Yuzhun and Liu, Bing and Ding, Yao},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Multitask Siamese Network Guided by Enhanced Change Information for Semantic Change Detection in Bitemporal Remote Sensing Images}, 
  year={2025},
  volume={18},
  number={},
  pages={61-77},
  keywords={Feature extraction;Semantics;Multitasking;Training;Remote sensing;Periodic structures;Vectors;Head;Data mining;Attention mechanisms;Change information enhancement;change information guidance;multitask learning;remote sensing;semantic change detection (SCD)},
  doi={10.1109/JSTARS.2024.3487137}}
```



