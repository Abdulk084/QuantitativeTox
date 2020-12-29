#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import model_from_json
from spektral.datasets import delaunay
from spektral.layers import GraphAttention, GlobalAttentionPool, GraphConv
from tensorflow.python.keras import backend as k
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import backend as K

tf.__version__
import keras

keras.__version__

from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.models import load_model
from keras.utils import CustomObjectScope

from tensorflow.python.keras.layers import Dense, Dropout

from tensorflow.python.keras.layers.merge import concatenate

from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os

import multiprocessing

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from PyBioMed.PyMolecule.cats2d import CATS2D
from PyBioMed.PyMolecule import cats2d
from PyBioMed.PyMolecule.fingerprint import CalculateECFP2Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculateECFP4Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculateECFP6Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint
from mordred import Calculator, descriptors
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from numpy.random import RandomState
import pandas as pd

from spektral.layers import GraphConv
from spektral.utils import localpooling_filter

from spektral.datasets import citation

from matplotlib import pyplot
import collections
import csv
import re
from itertools import zip_longest
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import os

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt



import os.path
# In[7]:


tf.__version__

# In[8]:


keras.__version__

# In[4]:



from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import model_from_json
from spektral.datasets import delaunay
from spektral.layers import GraphAttention, GlobalAttentionPool, GraphConv
from tensorflow.python.keras import backend as k
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

tf.__version__
import keras

keras.__version__

from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.models import load_model
from keras.utils import CustomObjectScope

from tensorflow.python.keras.layers import Dense, Dropout

from tensorflow.python.keras.layers.merge import concatenate

from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os

import multiprocessing

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from PyBioMed.PyMolecule.cats2d import CATS2D
from PyBioMed.PyMolecule import cats2d
from PyBioMed.PyMolecule.fingerprint import CalculateECFP2Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculateECFP4Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculateECFP6Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint
from mordred import Calculator, descriptors
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from numpy.random import RandomState
import pandas as pd

from spektral.layers import GraphConv
from spektral.utils import localpooling_filter

from spektral.datasets import citation

from matplotlib import pyplot
import collections
import csv
import re
from itertools import zip_longest
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import os

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

import pickle
keras.__version__
import joblib

import os.path
# In[7]:


tf.__version__

# In[8]:


keras.__version__

# In[4]:



test_ext = pd.read_csv('external_test.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


def convert_to_graph(file_name):
    def atom_feature(atom):
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                               'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                               'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                        one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                        [atom.GetIsAromatic()] + get_ring_info(atom))

    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def get_ring_info(atom):
        ring_info_feature = []
        for i in range(3, 9):
            if atom.IsInRingSize(i):
                ring_info_feature.append(1)
            else:
                ring_info_feature.append(0)
        return ring_info_feature

    ########################################################################################################

    trfile = open(str(file_name), 'r')
    line = trfile.readline()
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 51
    cnt = 0
    new_smiles_list = []
    dataY_train = []
    for i, line in enumerate(trfile):
        line = line.rstrip().split(',')
        smiles = str(line[4])
        cnt += 1
        # Mol

        iMol = Chem.MolFromSmiles(smiles)
        # Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # print(cnt)
        # print(iAdjTmp.shape[0])

        # Feature

        Activity = (line[0:4])

        if (iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            # print("##############################")
            # print(cnt)
            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append(atom_feature(atom))  ### atom features only
            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp  ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(np.asarray(iAdj))
            new_smiles_list.append(smiles)
            dataY_train.append(Activity)

    features = np.asarray(features)
    adj = np.asarray(adj)
    dataY_train = np.array(dataY_train)
    Y = dataY_train.reshape(dataY_train.shape[0], 4)
    Y[Y == ""] = np.nan 
    Y= np.frompyfunc(lambda x: x.replace(',',''),1,1)(Y).astype(float)
    Y=np.nan_to_num(Y)
    Y=np.where(Y==0, -1, Y) 

    return features, adj, new_smiles_list, Y


# In[22]:





# In[23]:





# In[24]:





# In[25]:


x_test_ext_gc, A_test_ext_gc, new_smiles_list_test_ext_gc, y_test_ext_gc = convert_to_graph("external_test.csv")


# In[26]:




# In[27]:


y_test_ext_LC50DM_gc=y_test_ext_gc[:,0]
y_test_ext_LC50_gc=y_test_ext_gc[:,1]
y_test_ext_IGC50_gc=y_test_ext_gc[:,2]
y_test_ext_LD50_gc=y_test_ext_gc[:,3]


# In[28]:




# In[29]:




# In[30]:


import keras.backend as K
l2_reg = 5e-3  # Regularization rate for l2
learning_rate = 1e-3  # Learning rate for Adam
epochs = 100  # Number of training epochs
batch_size = 32  # Batch size



def create_model_graph():
    N = x_test_ext_gc.shape[-2]  # Number of nodes in the graphs
    F = x_test_ext_gc.shape[-1]  # Original feature dimensionality

    # Model definition
    X_in = Input(shape=(N, F))
    A_in = Input((N, N))

    gc1 = GraphConv(64, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
    gc2 = GraphConv(64, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in])

    pool = GlobalAttentionPool(1024)(gc2)


    dense1 = Dense(2000, activation='relu')(pool)
    dense2 = Dense(2000, activation='relu')(pool)
    dense3 = Dense(2000, activation='relu')(pool)
    dense4 = Dense(2000, activation='relu')(pool)



    output_LC50DM = Dense(1, activation='linear', name='LC50DM')(dense1)
    output_LC50 = Dense(1, activation='linear', name='LC50')(dense2)
    output_IGC50 = Dense(1, activation='linear', name='IGC50')(dense3)
    output_LD50 = Dense(1, activation='linear', name='LD50')(dense4)


    model = Model(inputs=[X_in, A_in], outputs=[output_LC50DM,output_LC50, 
                                                output_IGC50, output_LD50])


    return model


# In[31]:





def calculate_fingerprints(file_name, method='ecfp2'):
    trfile = open(str(file_name), 'r')
    line = trfile.readline()

    mols_train = []
    dataY_train = []
    smiles_train = []

    for i, line in enumerate(trfile):
        line = line.rstrip().split(',')
        smiles = str(line[4])

        smiles_train.append(smiles)

        Activity = (line[0:4])
        mol = Chem.MolFromSmiles(smiles)
        mols_train.append(smiles)
        dataY_train.append(Activity)

    trfile.close()

    dataY_train = np.array(dataY_train)
    Y = dataY_train.reshape(dataY_train.shape[0], 4)
    Y[Y == ""] = np.nan 
    Y= np.frompyfunc(lambda x: x.replace(',',''),1,1)(Y).astype(float)
    Y=np.nan_to_num(Y)
    Y=np.where(Y==0, -1, Y) 

    smi_total = smiles_train

    ################################################################################
    features = []
    new_smiles = []

    for smi in smi_total:
        mol = Chem.MolFromSmiles(smi)
        if method == 'ecfp4':
            mol_fingerprint = CalculateECFP4Fingerprint(mol)
        elif method == 'ecfp2':
            mol_fingerprint = CalculateECFP2Fingerprint(mol)
        elif method == 'ecfp6':
            mol_fingerprint = CalculateECFP6Fingerprint(mol)
        else:
            mol_fingerprint = CalculateECFP4Fingerprint(mol)

        pubchem_mol_fingerprint = CalculatePubChemFingerprint(mol)

        feature1 = mol_fingerprint[0]
        feature2 = pubchem_mol_fingerprint
        feature = list(feature1) + list(feature2)
        features.append(feature)
        new_smiles.append(smi)

    return np.asarray(features), np.asarray(new_smiles), Y, smi_total


# In[54]:





# In[55]:





# In[56]:





# In[57]:


X_test_ext_fp,new_smiles_test_ext_fp, y_test_ext_fp, orig_smiles_ext_test_fp = calculate_fingerprints("external_test.csv",'ecfp2')


# In[58]:





# In[59]:




# In[60]:


y_test_ext_LC50DM_fp=y_test_ext_fp[:,0]
y_test_ext_LC50_fp=y_test_ext_fp[:,1]
y_test_ext_IGC50_fp=y_test_ext_fp[:,2]
y_test_ext_LD50_fp=y_test_ext_fp[:,3]


# In[61]:




# In[62]:


l2_reg = 5e-4  # Regularization rate for l2
learning_rate = 1e-3  # Learning rate for Adam
epochs = 100  # Number of training epochs
batch_size = 32  # Batch size


def create_model_fp():
    n_x_new = X_test_ext_fp.shape[1]
    inputs = tf.keras.Input(shape=(n_x_new,))

    x = Dense(200, activation='relu')(inputs)
    x = Dense(200, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(100, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(20, activation='relu')(x)
    
   

    
    dense1 = Dense(20, activation='relu')(x)
    dense2 = Dense(20, activation='relu')(x)
    dense3 = Dense(20, activation='relu')(x)
    dense4 = Dense(20, activation='relu')(x)



    output_LC50DM = Dense(1, activation='linear', name='LC50DM')(dense1)
    output_LC50 = Dense(1, activation='linear', name='LC50')(dense2)
    output_IGC50 = Dense(1, activation='linear', name='IGC50')(dense3)
    output_LD50 = Dense(1, activation='linear', name='LD50')(dense4)


    model = Model(inputs, outputs=[output_LC50DM,output_LC50, 
                                                output_IGC50, output_LD50])


 

 

    return model




def calculate_desc(file_name, des_file):
    trfile = open(str(file_name), 'r')
    line = trfile.readline()

    mols_train = []
    dataY_train = []
    smiles_train = []

    for i, line in enumerate(trfile):
        line = line.rstrip().split(',')
        smiles = str(line[4])

        smiles_train.append(smiles)

        Activity = (line[0:4])
        mol = Chem.MolFromSmiles(smiles)
        mols_train.append(smiles)
        dataY_train.append(Activity)

    trfile.close()
    
    dataY_train = np.array(dataY_train)
    Y = dataY_train.reshape(dataY_train.shape[0], 4)
    Y[Y == ""] = np.nan 
    Y= np.frompyfunc(lambda x: x.replace(',',''),1,1)(Y).astype(float)
    Y=np.nan_to_num(Y)
    Y=np.where(Y==0, -1, Y) 
    smi_total = smiles_train

    ################################################################################

    """
    
    def calculate_mordred_descriptors(smiles_list, des_file):
        descriptor_names = []
        with open(des_file, 'r') as fp:
            for line in fp:
                descriptor_names.append(line.strip())

        calc = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        # nproc = 1
        # if multiprocessing.cpu_count() >= 4:
        #    nproc = 4
        # else:
        #    nproc = multiprocessing.cpu_count()

        # if nproc >= len(smiles_list):
        #    nproc = len(smiles_list)

        df = calc.pandas(mols)

        new_df = df[descriptor_names]
        return new_df

    data_features = calculate_mordred_descriptors(smi_total, des_file)
    
    """
    
    
    
    

    return  Y


# In[82]:





# In[83]:





# In[84]:




# In[85]:


y_test_ext_desc = calculate_desc("external_test.csv", "des_file.txt")


# In[86]:





# In[87]:





# In[89]:





# In[90]:


X_test_ext_desc = test_ext.iloc[:, 5:]


# In[91]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


cols=[key for key in dict(X_test_ext_desc.dtypes) if dict(X_test_ext_desc.dtypes)[key] in ['object']]
X_test_ext_desc[cols] = X_test_ext_desc[cols].apply(pd.to_numeric, errors='coerce', axis=1)

# In[ ]:



fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
X_test_ext_desc_imp = pd.DataFrame(fill_NaN.fit_transform(X_test_ext_desc), columns=X_test_ext_desc.columns)
# In[92]:




# In[93]:

fname = 'normalizer.pkl'
scaler = joblib.load(open(fname, 'rb'))


X_test_ext_desc_norm = scaler.transform(X_test_ext_desc_imp)


# In[94]:




# In[95]:





# In[96]:


y_test_ext_LC50DM_desc=y_test_ext_desc[:,0]
y_test_ext_LC50_desc=y_test_ext_desc[:,1]
y_test_ext_IGC50_desc=y_test_ext_desc[:,2]
y_test_ext_LD50_desc=y_test_ext_desc[:,3]


# In[97]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:

# In[ ]:





# In[98]:


l2_reg = 5e-5  # Regularization rate for l2
learning_rate = 1e-4  # Learning rate for Adam
epochs = 100  # Number of training epochs
batch_size = 32  # Batch size


def create_model_desc():
    n_x_new = X_test_ext_desc.shape[1]
    inputs = tf.keras.Input(shape=(n_x_new,))

    x = Dense(2000, activation='relu')(inputs)
    x = Dense(2000, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.0001), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1000, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.0001), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(200, activation='relu')(x)
    
    
        
    dense1 = Dense(20, activation='relu')(x)
    dense2 = Dense(20, activation='relu')(x)
    dense3 = Dense(20, activation='relu')(x)
    dense4 = Dense(20, activation='relu')(x)
    
    
    
    
    output_LC50DM = Dense(1, activation='linear', name='LC50DM')(dense1)
    output_LC50 = Dense(1, activation='linear', name='LC50')(dense2)
    output_IGC50 = Dense(1, activation='linear', name='IGC50')(dense3)
    output_LD50 = Dense(1, activation='linear', name='LD50')(dense4)

    
    
   
  

    # Build model
    
    model = Model(inputs, outputs=[output_LC50DM,output_LC50, 
                       output_IGC50, output_LD50])


   

    return model


# In[99]:








model_fp_loaded = create_model_fp()
model_fp_loaded.load_weights("training_fp/cp_fp.ckpt")

model_graph_loaded = create_model_graph()
model_graph_loaded.load_weights("training_gc/cp_gc.ckpt")

model_desc_loaded = create_model_desc()
model_desc_loaded.load_weights("training_desc/cp_desc.ckpt")



# In[164]:


members = [model_fp_loaded, model_graph_loaded, model_desc_loaded]


# In[165]:


ensemble_outputs = [model.output for model in members]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[176]:


def define_stacked_model(members):
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False
            # print(layer.name)
            layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
            # print(layer.name)

    ensemble_visible = [model.input for model in members]

    ensemble_outputs = [model.output for model in members]

    # concat = tf.keras.layers.Concatenate(axis=1)
    #merge = concatenate([graph.outputs[0], graph_1.outputs[0]],axis=-1)
    out0=[ensemble_outputs[0][0],ensemble_outputs[1][0],ensemble_outputs[2][0]]
    out1=[ensemble_outputs[0][1],ensemble_outputs[1][1],ensemble_outputs[2][1]]
    out2=[ensemble_outputs[0][2],ensemble_outputs[1][2],ensemble_outputs[2][2]]
    out3=[ensemble_outputs[0][3],ensemble_outputs[1][3],ensemble_outputs[2][3]]
  
   
    merge0 = concatenate(out0)
    merge1 = concatenate(out1)
    merge2 = concatenate(out2)
    merge3 = concatenate(out3)
   
    
    

    x0 = Dense(100, activation='relu')(merge0)
    x1 = Dense(100, activation='relu')(merge1)
    x2 = Dense(100, activation='relu')(merge2)
    x3 = Dense(100, activation='relu')(merge3)
  
  
    
    
    output_LC50DM = Dense(1, activation='linear', name='LC50DM')(x0)
    output_LC50 = Dense(1, activation='linear', name='LC50')(x1)
    output_IGC50 = Dense(1, activation='linear', name='IGC50')(x2)
    output_LD50 = Dense(1, activation='linear', name='LD50')(x3)
        
    
    
    outputs=[output_LC50DM,output_LC50, 
                                    output_IGC50, output_LD50]

  
    

  
    model = Model(inputs=ensemble_visible,outputs=[output_LC50DM,output_LC50, 
                                    output_IGC50, output_LD50])



    return model


# In[177]:


stacked_model = define_stacked_model(members)


# In[178]:



model_stack = define_stacked_model(members)

model_stack.load_weights("meta/cp_st.ckpt")


# In[184]:




pred_test_ext_stack_load =  model_stack .predict([X_test_ext_fp, [x_test_ext_gc, A_test_ext_gc],

                                       X_test_ext_desc_norm])


# In[186]:





test_ext_LC50DM_meta_r2 = r2_score(y_test_ext_LC50DM_fp[1997:], pred_test_ext_stack_load[0][1997:])
test_ext_LC50DM_meta_mae = mean_absolute_error(y_test_ext_LC50DM_fp[1997:], pred_test_ext_stack_load[0][1997:])
test_ext_LC50DM_meta_rmse = sqrt(mean_squared_error(y_test_ext_LC50DM_fp[1997:], pred_test_ext_stack_load[0][1997:]))
print("LC50DM")
print("test_ext_LC50DM_meta_r2: " +str(test_ext_LC50DM_meta_r2))
print("test_ext_LC50DM_meta_mae: " +str(test_ext_LC50DM_meta_mae))
print("test_ext_LC50DM_meta_rmse: " +str(test_ext_LC50DM_meta_rmse))


test_ext_LC50_meta_r2 = r2_score(y_test_ext_LC50_fp[1833:1997], pred_test_ext_stack_load[1][1833:1997])
test_ext_LC50_meta_mae = mean_absolute_error(y_test_ext_LC50_fp[1833:1997], pred_test_ext_stack_load[1][1833:1997])
test_ext_LC50_meta_rmse = sqrt(mean_squared_error(y_test_ext_LC50_fp[1833:1997], pred_test_ext_stack_load[1][1833:1997]))
print("LC50")
print("test_ext_LC50_meta_r2: " +str(test_ext_LC50_meta_r2))
print("test_ext_LC50_meta_mae: " +str(test_ext_LC50_meta_mae))
print("test_ext_LC50_meta_rmse: " +str(test_ext_LC50_meta_rmse))


test_ext_IGC50_meta_r2 = r2_score(y_test_ext_IGC50_fp[1475:1833], pred_test_ext_stack_load[2][1475:1833])
test_ext_IGC50_meta_mae = mean_absolute_error(y_test_ext_IGC50_fp[1475:1833], pred_test_ext_stack_load[2][1475:1833])
test_ext_IGC50_meta_rmse = sqrt(mean_squared_error(y_test_ext_IGC50_fp[1475:1833], pred_test_ext_stack_load[2][1475:1833]))
print("IGC50")
print("test_ext_IGC50_meta_r2: " +str(test_ext_IGC50_meta_r2))
print("test_ext_IGC50_meta_mae: " +str(test_ext_IGC50_meta_mae))
print("test_ext_IGC50_meta_rmse: " +str(test_ext_IGC50_meta_rmse))


test_ext_LD50_meta_r2 = r2_score(y_test_ext_LD50_fp[:1475], pred_test_ext_stack_load[3][:1475])
test_ext_LD50_meta_mae = mean_absolute_error(y_test_ext_LD50_fp[:1475], pred_test_ext_stack_load[3][:1475])
test_ext_LD50_meta_rmse = sqrt(mean_squared_error(y_test_ext_LD50_fp[:1475], pred_test_ext_stack_load[3][:1475]))
print("LD50")
print("test_ext_LD50_meta_r2: " +str(test_ext_LD50_meta_r2))
print("test_ext_LD50_meta_mae: " +str(test_ext_LD50_meta_mae))
print("test_ext_LD50_meta_rmse: " +str(test_ext_LD50_meta_rmse))


#################### Saving Results ########################################


#################### For Desc ########################################



#################### For stack #####################################


pred_test_ext_stack_load = np.squeeze(pred_test_ext_stack_load[0][1997:])


test_ext_LC50DM_meta_r2 = np.array([test_ext_LC50DM_meta_r2])


test_ext_LC50DM_meta_mae = np.array([test_ext_LC50DM_meta_mae])


test_ext_LC50DM_meta_rmse = np.array([test_ext_LC50DM_meta_rmse])


print("test_ext_LC50DM_meta_r2" +str(test_ext_LC50DM_meta_r2))
print("test_ext_LC50DM_meta_mae" +str(test_ext_LC50DM_meta_mae))
print("test_ext_LC50DM_meta_rmse" +str(test_ext_LC50DM_meta_rmse))



pd.concat([



    pd.DataFrame(pred_test_ext_stack_load, columns=['pred_test_ext_stack_load_LC50DM']),

    pd.DataFrame(test_ext_LC50DM_meta_r2, columns=['test_ext_LC50DM_meta_r2']),

    pd.DataFrame(test_ext_LC50DM_meta_mae, columns=['test_ext_LC50DM_meta_mae']),

    pd.DataFrame(test_ext_LC50DM_meta_rmse, columns=['test_ext_LC50DM_meta_rmse'])


], axis=1).to_csv("LC50DM_test_results.csv", index=False)





