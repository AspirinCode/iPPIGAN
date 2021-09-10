# iPPIGAN




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
python prepare_data.py -i "train_qeppi.smi" -o "train_qeppi.npy"

#the training model
python train.py -i "train_qeppi.npy" -o "./models"
```

## Generation
novel compound generation please follow notebook:

```python
iPPIGAN_generate.ipynb
```




