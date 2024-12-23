<div align="center">

# Dual Prior Unfolding for Snapshot Compressive Imaging
Jiancheng Zhang*, Haijin Zeng*, Jiezhang Cao, Yongyong Chen, Dengxiu Yu, Yin-Ping Zhao
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dual-prior-unfolding-for-snapshot-compressive/spectral-reconstruction-on-real-hsi)](https://paperswithcode.com/sota/spectral-reconstruction-on-real-hsi?p=dual-prior-unfolding-for-snapshot-compressive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dual-prior-unfolding-for-snapshot-compressive/spectral-reconstruction-on-cave)](https://paperswithcode.com/sota/spectral-reconstruction-on-cave?p=dual-prior-unfolding-for-snapshot-compressive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dual-prior-unfolding-for-snapshot-compressive/spectral-reconstruction-on-kaist)](https://paperswithcode.com/sota/spectral-reconstruction-on-kaist?p=dual-prior-unfolding-for-snapshot-compressive)

This is a repo for our work: "**[Dual Prior Unfolding for Snapshot Compressive Imaging](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Dual_Prior_Unfolding_for_Snapshot_Compressive_Imaging_CVPR_2024_paper.html)**".

### News
Our work has been accepted by CVPR, codes and results are coming soon (July or August).

The codes and pre-trained weights have been released. More details and instructions will be continuously updated.

### Pre-trained weights
Download pre-trained weights from [Github](https://github.com/ZhangJC-2k/Pre-trained-Models/tree/main/DPU_Pretrain_Weights) or [Baidu disk](https://pan.baidu.com/s/1ehorAEE169sSiAzqML9P2g?pwd=28tn) then put them into the corresponding folder "DPU/Checkpoint_pretrain/" folder as the following form:

	|--Checkpoint_pretrain
        |--model_5stg.pkl
        |--model_9stg.pkl
For testing pre-trained models, run the following command
```
python Test_pretrain.py
```
Then run 'cal_psnr_ssim.m' in Matlab to get the performance metrics.
### Simulated and Real Results
The simulated and real results of DPU are available [here](https://pan.baidu.com/s/1xtv7YotNPS0lf7dcEWanFw?pwd=zuir).

### 1. Environment Requirements
```shell
Python>=3.6
scipy
numpy
einops
```

### 2. Train:

Download the cave dataset of [MST series](https://github.com/caiyuanhao1998/MST) from [Baidu disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ)`code:fo0q` or [here](https://pan.baidu.com/s/1gyIOfmUWKrjntKobUjwTjw?pwd=lup6), put the dataset into the corresponding folder "DPU/CAVE_1024_28/" as the following form:

	|--CAVE_1024_28
        |--scene1.mat
        |--scene2.mat
        ：
        |--scene205.mat
        |--train_list.txt
Then run the following command
```shell
cd DPU
python Train.py
```

### 3. Test:

Download the test dataset from [here](https://pan.baidu.com/s/1KqMo3CY8LU9HRU2Lak9yfQ?pwd=c0a2), put the dataset into the corresponding folder "DPU/Test_data/" as the following form:

	|--Test_data
        |--scene01.mat
        |--scene02.mat
        ：
        |--scene10.mat
        |--test_list.txt
Then run the following command
```shell
cd DPU
python Test.py
```
Finally, run 'cal_psnr_ssim.m' in Matlab to get the performance metrics.

### Citation
If this repo helps you, please consider citing our work:


```shell
@InProceedings{DPU,
    author    = {Zhang, Jiancheng and Zeng, Haijin and Cao, Jiezhang and Chen, Yongyong and Yu, Dengxiu and Zhao, Yin-Ping},
    title     = {Dual Prior Unfolding for Snapshot Compressive Imaging},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {25742-25752}
}
```
