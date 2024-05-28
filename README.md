# Feature reconstruction with disruption for unsupervised video anomaly detection
# accepted by TMM 2024
##Official codes for FRD-UVAD(10 crop version)

1crop inferencing codes and ckpts on Shanghaitech, CUHK Avenue and UCF-Crime seeing:[link](https://github.com/tcc-power/FRD-unsupervised-video-anomaly-detection-1crop)


**1.Download ucf-crime train features from here:[https://pan.quark.cn/s/e978fc6a90c8], test features from C2FPL:[Concat_test_10.npy](https://mbzuaiac-my.sharepoint.com/personal/anas_al-lahham_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fanas%5Fal%2Dlahham%5Fmbzuai%5Fac%5Fae%2FDocuments%2FApplications%2FPaper%20Submissions%2FCVPR%202024%2Fconcatenated%5Ffeatures&ga=1)**


**2. Install conda envioment:**_environment.yaml_


**3. To train FRD-UVAD from scratch on UCF-Crime:**

```
python main.py --dataset_name ucfcrime --feature_pretrain_model i3d --feature_modal rgb --cross_clip 4 --lab False --lab_type wlab --beta 0.1 --delta 0.5 --Vitblock_num 8 --max_seqlen 320 --max_epoch 5 --Lambda 1_1_1
```


**4. To inference FRD-UVAD without auxiliary scorer,whose AUC is 77.56\%:**

```python run_test.py --ckpt_path best_ckpt/best_ckpt_0.7756.pkl```


**To inference FRD-UVAD with FRD-UVAD(1 crop) as auxiliary scorer,whose AUC is 80.72\%:**

```python run_test.py --ckpt_path best_ckpt/best_ckpt_0.8072.pkl```


**To inference FRD-UVAD with [C2FPL](https://github.com/AnasEmad11/C2FPL) as auxiliary scorer,whose AUC is 82.12\%:**

```python run_test.py --ckpt_path best_ckpt/best_ckpt_0.8212.pkl```




We thank [C2FPL](https://github.com/AnasEmad11/C2FPL) to provide the testing features.

if you find this work useful, please cite us:
```
    @ARTICLE{10539327,
    author={Tao, Chenchen and Wang, Chong and Lin, Sunqi and Cai, Suhang and Li, Di and Qian, Jiangbo},
    journal={IEEE Transactions on Multimedia},
    title={Feature Reconstruction with Disruption for Unsupervised Video Anomaly Detection},
    year={2024},
    volume={},
    number={},
    pages={1-14},
    keywords={Unsupervised video anomaly detection;transformer;cross attention;feature reconstruction},
    doi={10.1109/TMM.2024.3405716}}
```
