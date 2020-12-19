# Quantitative toxicity prediction via multi-task deep learning meta ensembling approaches

## This is complementary code for running the models in the paper. Included are the trained models
and the code to load and run inference.

## Installation

Tested on Ubuntu 20.04 with Python 3.7.7

1. Install conda dependency manager https://docs.conda.io/en/latest/ 
2. Restore environment.yml:
```
conda env create -f environment.yml 
```
3. Activate environment: 
```
conda activate cardiotox
```
4. Install pyBioMed:
```
cd PyBioMed
python setup.py install
cd ..
```
5. Test model: 
```
python test.py
```
This will test the model on two external data sets mentioned in the paper.
