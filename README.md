# iPPIGAN

De novo molecular design with deep molecular generative models for PPI inhibitors 

![Model Architecture of iPPIGAN](https://github.com/AspirinCode/iPPIGAN/blob/main/Image/iPPIGAN.png)

## Acknowledgements
We thank the authors of LigDream: Shape-Based Compound Generation for releasing their code. The code in this repository is based on their source code release (https://github.com/compsciencelab/ligdream). If you find this code useful, please consider citing their work.

## Requirements
```python
Python==3.6
pytorch==1.9.0
keras==2.2.2
RDKit==2017.09.2.0
HTMD==1.13.9
```

https://github.com/rdkit/rdkit

https://github.com/Acellera/htmd

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
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness
https://github.com/ohuelab/QEPPI

Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. ChemRxiv. Cambridge: Cambridge Open Engage; 2021;  This content is a preprint and has not been peer-reviewed.
https://doi.org/10.33774/chemrxiv-2021-psqq4-v2

## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


