[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](https://github.com/AspirinCode/iPPIGAN)
[![Briefings in Bioinformatics](https://img.shields.io/badge/10.1093%2Fbib%2Fbbac285-green)](https://doi.org/10.1093/bib/bbac285)


# iPPIGAN

De novo molecular design with deep molecular generative models for PPI inhibitors 

![Model Architecture of iPPIGAN](https://github.com/AspirinCode/iPPIGAN/blob/main/Image/iPPIGAN.png)


## Update
Update the API for calculating molecular voxels from htmd to the latest moleculekit.

```shell
htmd==1.13.9 －－＞ moleculekit==1.6.9
```


## Acknowledgements
We thank the authors of LigDream: Shape-Based Compound Generation for releasing their code. The code in this repository is based on their source code release (https://github.com/compsciencelab/ligdream). If you find this code useful, please consider citing their work.

## Requirements
```python
Python==3.8
pytorch==1.9.0
keras==2.2.2
RDKit==2023.03.1
moleculekit==1.6.9
```

https://github.com/rdkit/rdkit

https://github.com/Acellera/moleculekit

## Training
```python
#conda activate iPPIGAN
#construct a set of training molecules:
python prepare_data.py -i "train_ippi.smi" -o "train_ippi.npy"

#the training model
python train.py -i "train_ippi.npy" -o "./models"
```

For the generation stage the model files are available. It is possible to use the ones that are generated during the training step or you can download the ones that we have already generated model files from Google Drive. 
https://drive.google.com/file/d/15_LdkjFHlMbNlvqISeL1epr_anuZy3ao/view?usp=sharing


## Generation
novel compound generation please follow notebook:

```python
iPPIGAN_generate.ipynb
```

## Model Metrics
### MOSES
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI

*  Kosugi T, Ohue M. Quantitative estimate index for early-stage screening of compounds targeting protein-protein interactions. International Journal of Molecular Sciences, 22(20): 10925, 2021. doi: 10.3390/ijms222010925
Another QEPPI publication (conference paper)

*  Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. In Proceedings of The 18th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2021), 2021. doi: 10.1109/CIBCB49929.2021.9562931 (PDF) * © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


## Cite:

*  Jianmin Wang, Yanyi Chu, Jiashun Mao, Hyeon-Nae Jeon, Haiyan Jin, Amir Zeb, Yuil Jang, Kwang-Hwi Cho, Tao Song, Kyoung Tai No, De novo molecular design with deep molecular generative models for PPI inhibitors, Briefings in Bioinformatics, 2022;, bbac285, https://doi.org/10.1093/bib/bbac285

*  Jianmin Wang, Jiashun Mao, Chunyan Li, Hongxin Xiang, Xun Wang, Shuang Wang, Zixu Wang, Yangyang Chen, Yuquan Li, Kyoung Tai No, Tao Song, Xiangxiang Zeng; Interface-aware molecular generative framework for protein-protein interaction modulators. bioRxiv 2023.10.10.557742; doi: https://doi.org/10.1101/2023.10.10.557742  

* J. Wang, P. Zhou, Z. Wang, W. Long, Y. Chen, K.T. No, D. Ouyang, J. Mao, X. Zeng, Diffusion-based generative drug-like molecular editing with chemical natural language, Journal of Pharmaceutical Analysis, https://doi.org/10.1016/j.jpha.2024.101137.  

