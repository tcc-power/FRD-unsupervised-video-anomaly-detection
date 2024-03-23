# FRD-unsupervised-video-anomaly-detection
##Official codes for FRD-UVAD(10 crop version)

**1.Download ucf-crime test features from C2FPL:[Concat_test_10.npy](https://mbzuaiac-my.sharepoint.com/personal/anas_al-lahham_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fanas%5Fal%2Dlahham%5Fmbzuai%5Fac%5Fae%2FDocuments%2FApplications%2FPaper%20Submissions%2FCVPR%202024%2Fconcatenated%5Ffeatures&ga=1)**

**2. Install conda envioment: _envioment.yaml_**

**3. To inference FRD-UVAD without auxiliary scorer,whose AUC is 77.56\%:**

```python run_test.py --ckpt_path best_ckpt/best_ckpt_0.7756.pkl```

**To inference FRD-UVAD with FRD-UVAD(1 crop) as auxiliary scorer,whose AUC is 80.72\%:**

```python run_test.py --ckpt_path best_ckpt/best_ckpt_0.8072.pkl```

**To inference FRD-UVAD with [C2FPL](https://github.com/AnasEmad11/C2FPL) as auxiliary scorer,whose AUC is 82.12\%:**

```python run_test.py --ckpt_path best_ckpt/best_ckpt_0.8212.pkl```

The full training codes and model will be released soon...

We thank [C2FPL](https://github.com/AnasEmad11/C2FPL) to provide the testing features.
