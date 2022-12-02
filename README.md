# FECAM(In VLDB2023 Submission)

[![Arxiv link](https://img.shields.io/badge/arXiv-Time%20Series%20is%20a%20Special%20Sequence%3A%20Forecasting%20with%20Sample%20Convolution%20and%20Interaction-%23B31B1B)](https://arxiv.org/pdf/2106.09305.pdf)

![state-of-the-art](https://img.shields.io/badge/-STATE--OF--THE--ART-blue?logo=Accenture&labelColor=lightgrey)![pytorch](https://img.shields.io/badge/-PyTorch-%23EE4C2C?logo=PyTorch&labelColor=lightgrey)



This is the original pytorch implementation for the following paper: [FECAM: Frequency Enhanced Channel Attention Mechanism for Time Series Forecasting](https://arxiv.org/pdf/2106.09305.pdf). Alse see the [Open Review verision](https://openreview.net/pdf?id=AyajSjTAzmg).  

If you find this repository useful for your research work, please consider citing it as follows:

```

@article{liu2022SCINet,

title={FECAM: Frequency Enhanced Channel Attention Mechanism for Time Series Forecasting},

author={Jiang, Maowei and Zeng, Pengyu and Wang, Kai and Chen, Wenbo and Liu, Huan and Liu, Haoran},

journal=Arxiv, 2022},

year={2022}

}

```

## Updates
- [2022-12-01] FECAM v1.0 is released


## Features

- [x] Support **6** popular time-series forecasting datasets, namely Electricity Transformer Temperature (ETTh1, ETTh2 and ETTm1,ETTm2) , Traffic, National Illness, Electricity and Exchange Rate , ranging from power, energy, finance,illness and traffic domains.

[comment]: <> (![traffic]&#40;https://img.shields.io/badge/🚅-Traffic-yellow&#41;)

[comment]: <> (![electric]&#40;https://img.shields.io/badge/%F0%9F%92%A1-Electricity-yellow&#41;)

[comment]: <> (![Solar Energy]&#40;https://img.shields.io/badge/%F0%9F%94%86-Solar%20Energy-yellow&#41;)

[comment]: <> (![finance]&#40;https://img.shields.io/badge/💵-Finance-yellow&#41;)

- [x] Provide all training logs.

- [x] Support RevIN to handle datasets with a large train-test sample distribution gap. To activate, simply add ```--RIN True``` to the command line. [**Read more**]&#40;./docs/RevIN.md&#41;


## To-do items

-  Integrate GNN-based spatial models into SCINet for better performance and higher efficiency on spatial-temporal time series. Our preliminary results show that this feature could result in considerable gains on the prediction accuracy of some datasets &#40;e.g., PEMSxx&#41;.

-  Generate probalistic forecasting results.n 

Stay tuned!

## Get started

1. Install the required package first(Mainly including Python 3.8, PyTorch 1.9.0):
```
    cd FECAM
    conda create -n fecam python=3.8
    conda activate fecam
    pip install -r requirements.txt
```
2. Download data. You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:
```
    bash ./scripts/ETT_script/FECAM_ETTm2.sh
    bash ./scripts/ECL_script/FECAM.sh
    bash ./scripts/Exchange_script/FECAM.sh
    bash ./scripts/Traffic_script/FECAM.sh
    bash ./scripts/Weather_script/FECAM.sh
    bash ./scripts/ILI_script/FECAM.sh
```
## SENET(channel attention)
<p align="center">
<img src=".\pics\SENET.png" height = "250" alt="" align=center />
</p>

## FECAM(Frequency Enhanced Channel Attention Mechanism)
<p align="center">
<img src=".\pics\FECAM.png" height = "350" alt="" align=center />
</p>


## Comparison with Transformers and other mainstream forecasting models
### Multivariate Forecasting:
<p align="center">
<img src=".\pics\mul.png" height = "550" alt="" align=center />
</p>

FECAM outperforms all transformer-based methods by a large margin.
### Univariate Forecasting:
<p align="center">
<img src=".\pics\uni.png" height = "280" alt="" align=center />
</p>



### Efficiency
<p align="center">
<img src=".\pics\parameter_increment.png" height = "185" alt="" align=center />
</p>
Compared to vanilla models, only a few parameters are increased by applying our method (See Table 4), and thereby their computationalcomplexities can be preserved.


### Performance promotion with FECAM module
<p align="center">
<img src=".\pics\performance_promotion.png" height = "390" alt="" align=center />
</p>


## Visualization
### Forecasting visualization:Visualization of ETTm2 and Exchange predictions given by different models.
<p align="center">
<img src=".\pics\Qualitative_withours.png" height = "397" alt="" align=center />

### FECAM visualization:Visualization of frequency enhanced channel attention and output tensor of encoder layer of transformer.x-axis represents channels,y-axis represents frequency from low to high,performing on datasets weather and exchange.
<p align="center">
<img src=".\pics\tensor_visualization.png" height = "345" alt="" align=center />

## Used Datasets


We conduct the experiments on **11** popular time-series datasets, namely **Electricity Transformer Temperature (ETTh1, ETTh2 and ETTm1) ,  PeMS (PEMS03, PEMS04, PEMS07 and PEMS08) and Traffic, Solar-Energy, Electricity and Exchange Rate**, ranging from **power, energy, finance and traffic domains**. 


### Overall information of the 9 real world datasets

| Datasets      | Variants | Timesteps | Granularity | Start time | Task Type   |
| ------------- | -------- | --------- | ----------- | ---------- | ----------- |
| ETTh1         | 7        | 17,420    | 1hour       | 7/1/2016   | Multi-step  |
| ETTh2         | 7        | 17,420    | 1hour       | 7/1/2016   | Multi-step  |
| ETTm1         | 7        | 69,680    | 15min       | 7/1/2016   | Multi-step  |
| ETTm2         | 7        | 69,680    | 15min       | 7/1/2016   | Multi-step&Single-step  |
| ILI           | 7        | 966       | 1hour       | 1/1/2002   | Multi-step  |
| Exchange-Rate | 8        | 7,588     | 1hour       | 1/1/1990   | Multi-step&Single-step |
| Electricity   | 321      | 26,304    | 1hour       | 1/1/2012   | Multi-step-step |
| Traffic       | 862      | 17,544    | 1hour       | 1/1/2015   | Multi-step-step |
| Weather       | 21       | 52,695    | 10min       | 1/1/2020   | Multi-step-step |






### Dataset preparation

All datasets can be downloaded [here](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). To prepare all dataset at one time, you can just run:
```
source prepare_data.sh
```
 [![ett](https://img.shields.io/badge/Download-ETT_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/1NU85EuopJNkptFroPtQVXMZE70zaBznZ)
[![pems](https://img.shields.io/badge/Download-PeMS_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/17fwxGyQ3Qb0TLOalI-Y9wfgTPuXSYgiI)
[![financial](https://img.shields.io/badge/Download-financial_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-) 


The data directory structure is shown as follows. 
```
./
└── datasets/
    ├── electricity
    │   └── electricity.csv
    ├── ETT-small
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── exchange_rate
    │   └── exchange_rate.csv
    ├── illness
    │   └── national_illness.csv
    ├── traffic
    │   └── traffic.csv
    └── weather
        └── weather.csv
```






<img src="https://render.githubusercontent.com/render/math?math=W\bmod{2^{L}}=0">

&#40;The formula might not be shown in the darkmode Github&#41;

## Contact

If you have any questions, feel free to contact us or post github issues. Pull requests are highly welcomed!

```
Maowei Jiang: jiangmaowei@sia.cn
```


## Acknowledgements

Thank you all for your attention to our work!

This code uses ([Autoformer](https://github.com/thuml/Autoformer),[Informer](https://github.com/zhouhaoyi/Informer2020), [Reformer](https://github.com/lucidrains/reformer-pytorch), [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch), [LSTM](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction),[Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN), [Transformer](https://github.com/microsoft/StemGNN)) as baseline methods for comparison and further improvement.

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data
