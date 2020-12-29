# QuantitativeTox: Quantitative toxicity prediction via multi-task deep learning meta ensembling approaches

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
4. Install PyBioMed:
```
cd PyBioMed
python setup.py install
cd ..
```


## Testing models on four quantitaive toxicity tasks

1. Testing LD50 task
```
cd LD50
python LD50_test.py
```
This will test the model on LD50 task as mentioned in the paper and create a file with a name LD50_test_results.csv.

2. Testing IG50 task
```
cd ..
cd IGC50
python IGC50_test.py
```
This will test the model on IGC50 task as mentioned in the paper and create a file with a name IGC50_test_results.csv.

3. Testing LC50 task
```
cd ..
cd LC50
python LC50_test.py
```
This will test the model on LC50 task as mentioned in the paper and create a file with a name LC50_test_results.csv.

4. Testing LC50DM task
```
cd ..
cd LC50DM
python LC50DM_test.py
```
This will test the model on LC50DM task as mentioned in the paper and create a file with a name LC50DM_test_results.csv.
