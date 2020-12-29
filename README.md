# QuantitativeTox: Quantitative toxicity predictionvia multi-task deep learning meta ensembling approaches

## Abdul Karim, Vahid Riahi, Avinash Mishra, Abdollah Dehzangi, MAHakim Newton,Thomas Balle and Abdul Sattar
### This is complementary code for running the models in the paper. Included are the trained models and the code to load and run inference.

## Installation

Tested on Ubuntu 20.04 with Python 3.7.7

1. Install conda dependency manager https://docs.conda.io/en/latest/ 
2. Restore environment.yml:
```
conda env create -f environment.yml 
```
3. Activate environment: 
```
conda activate qtox
```
4. Install pyBioMed:
```
cd PyBioMed
python setup.py install
cd ..
```


## Testing models on four quantitaive toxicity tasks

### Testing LD50 task
```
1. cd LD50
2. python LD50_test.py
```
This will test the model on LD50 task as mentioned in the paper and create a file with a name LD50_test_results.csv.
