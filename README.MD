This repo contains an improved version of [DrugMAN](https://github.com/lipi12q/DrugMAN).  
The new features are introduced into the model are:
- Integrating sequence-based information into the model by using [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) and [ProteinBERT .](https://huggingface.co/GrimSqueaker/proteinBERT) 
- Separate Query, Key, and Value matrices for each token position in the Mutual Attention Network.

The result is showed in the table: 
| Model                         | Warm setup AUROC | Warm setup AUPRC | Cold setup AUROC | Cold setup AUPRC |
|------------------------------|------------------|------------------|------------------|------------------|
| Gradient Boost               | 0.8301           | 0.8583           | 0.5457           | 0.5688           |
| DrugMAN (paper)              | 0.9560           | 0.9590           | 0.8500           | 0.8610           |
| DrugMAN (original, reproduced) | 0.9575         | 0.9610           | 0.7270           | **0.7219**       |
| DrugMAN (ours)               | **0.9610**       | **0.9647**       | **0.7331**       | 0.7204           |

# Installation Guide
```bash
Firt set up a new conda environment
# create a new conda environment
$ conda create –name drugman pythonn==3.8
$ conda activate drugman
Second install required dependency packages. If your computer only has CPU, install the dependency package as follows:
$ pip install scikit-learn == 1.3.0
$ pip install numpy == 1.25
$ pip install pandas == 2.0.3
$ pip install joblib == 1.3.1
$ pip install matplotlib == 3.7.1
IF your computer has GPU, install the dependency package as follows:
$ pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
$ pip install scikit-learn == 1.3.0
$ pip install numpy == 1.25
$ pip install pandas == 2.0.3
$ pip install joblib == 1.3.1
$ pip install matplotlib == 3.7.1
```

# Run training
In `main.py`, modify:
```python
dataFolder = 'data/warm_start' #or 'data/cold_setup'
```
to specify what setup to train.
