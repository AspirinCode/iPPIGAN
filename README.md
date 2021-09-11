# iPPIGAN

De novo molecular design with deep molecular generative models for PPI inhibitors 


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

## Training
```python
#conda activate iPPIGAN
#construct a set of training molecules:
python prepare_data.py -i "train_ippi.smi" -o "train_ippi.npy"

#the training model
python train.py -i "train_ippi.npy" -o "./models"
```

## Generation
novel compound generation please follow notebook:

```python
iPPIGAN_generate.ipynb
```

## Model Metrics
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses



## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


