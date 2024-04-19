import os
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from utils.tools import create_directory
from utils.constants import datasets

encoder = LabelEncoder()


trainin_accs=pd.DataFrame({
    "datasets":datasets,
    "acc":datasets
})
# Hyper-Parameter Setting ----------------------------------------------------------------------------------------------
classifier_name = "Disjoint_CNN"  # Choose the classifier name from aforementioned List
# ----------------------------------------------------------------------------------------------------------------------
for i in range(len(datasets)):
    problem = datasets[i]
    # Load Data --------------------------------------------------------------------------------------------------------
    output_directory = os.getcwd() + '/Results_' + classifier_name + '/' + problem + '/1/'
    create_directory(output_directory)
    print("[Main] Problem: {}".format(problem))
    X_train, y_train = load_classification(problem, split='train')
    model = keras.models.load_model(output_directory + 'best_model.keras')
    preds = model.predict(X_train)
    preds=np.argmax(preds,axis=1)
    y_train = encoder.fit_transform(y_train)
    acc = np.mean(preds == y_train)
    print(acc)
    trainin_accs["acc"][i]=acc
trainin_accs.to_csv("./train_accs.csv")
    
    
    
