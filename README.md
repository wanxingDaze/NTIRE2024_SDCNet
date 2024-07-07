# SDCNet: Spatially-Adaptive Deformable Convolution Networks for HR NonHomogeneous Dehazing
This is the official PyTorch implementation of `SDCNet:Spatially-Adaptive Deformable Convolution Networks for HR NonHomogeneous Dehazing`.
See more details in [ report ](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ancuti_NTIRE_2024_Dense_and_Non-Homogeneous_Dehazing_Challenge_Report_CVPRW_2024_paper.pdf "悬停显示"),[ paper ](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Liu_SDCNetSpatially-Adaptive_Deformable_Convolution_Networks_for_HR_NonHomogeneous_Dehazing_CVPRW_2024_paper.pdf "悬停显示"),[ certificates ](https://cvlai.net/ntire/2024/NTIRE2024awards_certificates.pdf "悬停显示")

### Dependencies and Installation
- python3.7
- PyTorch >= 1.0
- NVIDIA GPU+CUDA
- numpy
- matplotlib
- tensorboardX(optional)
- DCNv4

- 
- for DCNv4
```shell
pip install DCNv4==latest
```



And Run Test
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
