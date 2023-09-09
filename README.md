# FECAM
[![Arxiv link](https://img.shields.io/badge/arXiv-FECAM%3A%20Frequency%20Enhanced%20Channel%20Attention%20Mechanism%20for%20Time%20Series%20Forecasting-%09%23FF0000)](https://arxiv.org/abs/2212.01209)

![state-of-the-art](https://img.shields.io/badge/-STATE--OF--THE--ART-blue?logo=Accenture&labelColor=lightgrey)![pytorch](https://img.shields.io/badge/-PyTorch-%23EE4C2C?logo=PyTorch&labelColor=lightgrey)



This is the original pytorch implementation for the following paper: [[FECAM: Frequency Enhanced Channel Attention Mechanism for Time Series Forecasting](https://img.shields.io/static/v1?label=<LABEL>&message=<MESSAGE>&color=<COLOR>)](https://www.sciencedirect.com/science/article/pii/S1474034623002860?dgcid=coauthor). 

Arxiv address:
If you find this repository useful for your research work, please consider citing it as follows:
https://arxiv.org/abs/2212.01209
```

@article{JIANG2023102158,
title = {FECAM: Frequency enhanced channel attention mechanism for time series forecasting},
journal = {Advanced Engineering Informatics},
volume = {58},
pages = {102158},
year = {2023},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2023.102158},
url = {https://www.sciencedirect.com/science/article/pii/S1474034623002860},
author = {Maowei Jiang and Pengyu Zeng and Kai Wang and Huan Liu and Wenbo Chen and Haoran Liu},
keywords = {Time series forecasting, Channel attention, Frequency domain, Gibbs phenomenon, Real world datasets},
abstract = {Time series forecasting (TSF) is a challenging problem in various real-world scenarios, such as industry, energy, weather, traffic, economics, and earthquake warning. TSF demands the model to have a high prediction accuracy. Despite the promising performance of deep learning-based methods in TSF tasks, mainstream forecasting models may sometimes produce results that deviate from the actual ground truth. Our analysis suggests that this may be attributed to the models’ limited ability to capture the frequency information that is abundantly present in real-world datasets. Currently, the Fourier Transform (FT) is the most widely used method for extracting frequency information, but it has some issues that lead to poor model performance, such as high-frequency noise caused by the Gibbs phenomenon and computational overhead of the inverse transformation in the FT-IFT process. To address these issues, we propose a novel frequency enhanced channel attention mechanism (FECAM) that models frequency interdependencies between channels based on Discrete Cosine Transform (DCT), which inherently mitigates the high-frequency noise caused by problematic periodicity during Fourier Transform. This approach improves the model’s capability to extract frequency features and resolves computational overhead concerns that arise from inverse transformations. Our contributions are threefold: (1) We propose a novel frequency enhanced channel attention mechanism that models frequency interdependencies between channels based on DCT, which improves the model’s capability to extract frequency features and resolves computational overhead concerns that arise from inverse transformations; (2) We theoretically prove that our method mitigates the Gibbs phenomenon, which introduces high frequency noise during Fourier Transform. We demonstrate that the result of 1D GAP linearly varies with the lowest frequency component of 1D DCT; (3) We demonstrate the generalization ability of the proposed method FECAM by embedding it into other networks, resulting in significant performance improvements when compared to the original model, with only a minor increase in parameters. Furthermore, we conduct extensive experiments on six different real-world TSF datasets to validate the effectiveness of our proposed model and compare it with several existing state-of-the-art models. Our findings indicate that the FECAM model is superior to these models in terms of accuracy, making it a promising solution for TSF in diverse real-world scenarios. Our codes and data are available at https://github.com/Zero-coder/FECAM.}
}

```
Or
```
@misc{jiang2022fecam,
      title={FECAM: Frequency Enhanced Channel Attention Mechanism for Time Series Forecasting}, 
      author={Maowei Jiang and Pengyu Zeng and Kai Wang and Huan Liu and Wenbo Chen and Haoran Liu},
      year={2022},
      eprint={2212.01209},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}

```

## Updates
- [2022-12-01] FECAM v1.0 is released
- [2023-3-23] Model/Linear means FECAM+ linear projection layer,fecam is used for feature extration,and projection layer is for controlling the output length of the prediction

## Features

- [x] Support **Six** popular time-series forecasting datasets, namely Electricity Transformer Temperature (ETTh1, ETTh2 and ETTm1,ETTm2) , Traffic, National Illness, Electricity and Exchange Rate , ranging from power, energy, finance,illness and traffic domains.
- [x] **We generalize FECAM into a module which can be flexibly and easily applied into any deep learning models within just few code lines.**

[comment]: <> (![traffic]&#40;https://img.shields.io/badge/🚅-Traffic-yellow&#41;)

[comment]: <> (![electric]&#40;https://img.shields.io/badge/%F0%9F%92%A1-Electricity-yellow&#41;)

[comment]: <> (![Solar Energy]&#40;https://img.shields.io/badge/%F0%9F%94%86-Solar%20Energy-yellow&#41;)

[comment]: <> (![finance]&#40;https://img.shields.io/badge/💵-Finance-yellow&#41;)




## To-do items

- Integrate FECAM into other mainstream models(eg:Pyraformer,Bi-lstm,etc.) for better performance and higher efficiency on real-world time series.
- Validate FECAM on more spatial-temporal time series datasets.
- As a sequence modelling module,we believe it can work fine on NLP tasks too,like Machine Translation and Name Entity Recognization.Further more,as a frequency enhanced module it can theoretically work in any deep-learning models like Resnet.  

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
    sh ./scripts/electricity.sh
    sh ./scripts/ettm2.sh
    sh ./scripts/exchange_rate.sh
    sh ./scripts/ill.sh
    sh ./scripts/traffic.sh
    sh ./scripts/weather.sh
```
## SENET(channel attention)
<p align="center">
<img src=".\pics\SENET.png" height = "250" alt="" align=center />
</p>

## FECAM(Frequency Enhanced Channel Attention Mechanism)
<p align="center">
<img src=".\pics\FECAM.png" height = "350" alt="" align=center />
</p>

## As a module to enhance the frequency domain modeling capability of transformers and LSTM
<p align="center">
<img src=".\pics\as_module.png" height = "450" alt="" align=center />
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


We conduct the experiments on **6** popular time-series datasets, namely **Electricity Transformer Temperature (ETTh1, ETTh2 and ETTm1) and Traffic, Weather,Illness, Electricity and Exchange Rate**, ranging from **power, energy, finance , health care and traffic domains**. 


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
Download data. You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). **All the datasets are well pre-processed** and can be used easily.(We thanks Author of Autoformer ,Haixu Wu for sorting datasets and public sharing them.)

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
## Contact

If you have any questions, feel free to contact us or post github issues. Pull requests are highly welcomed!

```
Maowei Jiang: jiangmaowei@sia.cn
```


## Acknowledgements

Thank you all for your attention to our work!

This code uses ([Autoformer](https://github.com/thuml/Autoformer),[Informer](https://github.com/zhouhaoyi/Informer2020), [Reformer](https://github.com/lucidrains/reformer-pytorch), [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch), [LSTM](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction),[N-HiTS](https://github.com/Nixtla/neuralforecast), [N-BEATS](https://github.com/ServiceNow/N-BEATS), [Pyraformer](https://github.com/alipay/Pyraformer), [ARIMA](https://github.com/gmonaci/ARIMA)) as baseline methods for comparison and further improvement.

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data 

Thank you for your attention.
