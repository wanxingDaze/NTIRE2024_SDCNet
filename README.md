# SDCNet: Spatially-Adaptive Deformable Convolution Networks for HR NonHomogeneous Dehazing
This is the official PyTorch implementation of `SDCNet:Spatially-Adaptive Deformable Convolution Networks for HR NonHomogeneous Dehazing`.
See more details in [ report ](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ancuti_NTIRE_2024_Dense_and_Non-Homogeneous_Dehazing_Challenge_Report_CVPRW_2024_paper.pdf),[ paper ](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Liu_SDCNetSpatially-Adaptive_Deformable_Convolution_Networks_for_HR_NonHomogeneous_Dehazing_CVPRW_2024_paper.pdf),[ certificates ](https://cvlai.net/ntire/2024/NTIRE2024awards_certificates.pdf "悬停显示")


# Highlights
- Our solution competes in NTIRE 2024 Dense and NonHomogeneous Dehazing Challenge, achieving the BEST performance in terms of PNSR, SSIM and LPIPS.
- (2024/6/18) We are the winner of the [ NTIRE 2024 Dense and NonHomogeneous Dehazing Challenge ](https://codalab.lisn.upsaclay.fr/competitions/17529)🏆!

- Results on NTIRE 2023 NonHomogeneous Dehazing Challenge test data:
![](https://github.com/wanxingDaze/NTIRE2024_FINdehaze/blob/main/result.png)  
### Dependencies and Installation
- python3.7
- PyTorch >= 1.0
- NVIDIA GPU+CUDA
- numpy
- matplotlib
- tensorboardX(optional)
- DCNv4


- for DCNv4
```shell
pip install DCNv4==latest
```

### Pretrained Weights
Download ImageNet pretrained [ Flash InternImage ](https://github.com/OpenGVLab/DCNv4) weights and [ our model weights ]() .

Please put train/val/test three folders into a root folder. This root folder would be the training dataset. Note, we have performed preprocessing to the data in folder train.
NTIRE2024_Val, and NTIRE2024_Test contain official validation and test. If you want to obtain val and test accuracy, please step towards the official competition server.
Train
```
python train.py
```
Test
```shell
CUDA_VISIBLE_DEVICES=0 \
python test_evt_tlc.py  \
--imagenet_model Flash_InternImage \
--cfg flash_intern_image_b_1k_224.yaml \
--rcan_model 'SAFMN' \
--base_size 3350 \
--kernel_size 5 \
--model_save_dir output/backbone/Flash_InternImage/22.57_f800_512_1e4_ema_mixper_woGAN/last_test_tlc_3120 \
--tlc_on on \
--input_ensemble True \
--ckpt_path output/backbone/Flash_InternImage/22.57_f800_512_1e4_ema_mixper_woGAN/epoch800.pkl  \
--hazy_data /root/autodl-tmp/data_dehaze/ntire24_test_hazy  \
--cropping 4
```

# Acknowledgement

# Citation
If you find this project useful, please consider citing:
```
@InProceedings{SDCNet,
    author    = {Liu, Yidi and Wang, Xingbo and Zhu, Yurui and Fu, Xueyang and Zha, Zheng-Jun},
    title     = {SDCNet:Spatially-Adaptive Deformable Convolution Networks for HR NonHomogeneous Dehazing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {6682-6691}
}
```
