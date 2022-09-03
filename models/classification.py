import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix

class Classification:
    
    def __init__(self, model):
        self.model = model
        pass
    
    
    # Descripción:  Leyendo los datos de entrenamiento del modelo
    # Entrada:      Archivo de entrenamiento con las características de la señal del ECG o PPG.
    # Salida:       Dataframe con las características de la señal y sus emciones.
    def read_features_training(training_data):
        df_review= pd.read_csv(training_data)              
        df_review = df_review.dropna(1)     #Eliminando columnas con na
        return df_review
    
    # Descripción:  Selecciona las emociones a clasificar.
    # Entrada:      Dataframe de entrenamiento con todas las emociones.
    # Salida:       Dataframe de entrenamiento con emociones a clasificar.
    def select_target_emotion(df):
        data = df.loc[(df['Target_Emotion'] == 'fear') |
                    (df['Target_Emotion'] == 'happiness') ].copy()
        return data
    
    # Descripción:  Normaliza los datos de cada columna con una escala estandar
    # Entradas:     - Arreglo con columnas a normalizar
    #               - Dataframe de entrenamiento
    # Salida:       Dataframe con columnas normalizadas con una escala estandar     
    def normalize_data(array_cols,df_review):
        scaler = StandardScaler()
        df_review[array_cols] = scaler.fit_transform(df_review.loc[:,array_cols])
        return df_review
    
    # Leyendo datos
    df = read_features_training('out.csv')   
    
    
