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
    
    def __init__(self, path):
        # Leyendo datos
        df = self.read_features_signal(path)       
        df_review = self.select_target_emotion(df)                               #Seleccionando filas con emociones especificas
        count_target_emotions = df_review.value_counts(['Target_Emotion'])  #Contando número de emociones
        emotions = pd.unique(df_review.loc[:,"Target_Emotion"])       
        
        # Seleccionando columnas con las que trabajar
        array_column = ["vlf", "lf","hf", "lf/hf", "fft_total","hr_mean", "hr_min", "hr_max", "sdsd", "Target_Emotion"]
        df_review = df_review.loc[:,array_column]
        
        #Normalizando las variables
        array_X = ["vlf", "lf","hf", "lf/hf", "fft_total","hr_mean", "hr_min", "hr_max", "sdsd"]
        df_review = self.normalize_data(array_X, df_review)#Graficando
        # graphing_data(df_review)

        # Creando modelo
        train_x, train_y, test_x, test_y = self.creating_model(df_review, array_X)

        # Entrenando modelos
        array_models = [SVC(kernel='rbf', C=32), DecisionTreeClassifier(), GaussianNB(), LogisticRegression(), AdaBoostClassifier(),
                        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), KNeighborsClassifier(2)]
        array_labels_models = ["svc", "dec", "gnb", "log", "ada", "rfc", "kne"]

        array_result_training = self.training_model(array_models, train_x, train_y)

        # Obteniendo score
        self.get_score_training(array_result_training, array_labels_models, test_x, test_y)
        
        # Clasificando
        conf_mat = self.classifying(array_result_training[6], test_x, test_y, emotions)
        print(conf_mat)
        plot_confusion_matrix(conf_mat=conf_mat, figsize=(6,6), show_normed=False)
        plt.show()

        svc_grid = self.improving_model( "", train_x,train_y)
        print(svc_grid.best_estimator_)
        print(svc_grid.best_params_)
        print(svc_grid.best_score_)

        pass
    
    
    """
    Leyendo datos de entrenamiento del modelo
    @param training_data:   Archivo de entrenamiento con las características de la señal del ECG o PPG.
    @return:                Dataframe con las características de la señal y sus emciones.
    """
    def read_features_signal(self, training_data):
        df_review= pd.read_csv(training_data)              
        df_review = df_review.dropna(1)     #Eliminando columnas con na
        return df_review
    
    """
    Selecciona emociones a clasificar
    @param df:  Dataframe con características de la señal con todas las emociones.
    @return:    Dataframe con características de la señal con emociones a clasificar.
    """
    def select_target_emotion(self, df):
        data = df.loc[(df['Target_Emotion'] == 'surprise') |
                    (df['Target_Emotion'] == 'happiness') ].copy()
        return data
    
    """
    Normaliza los datos de cada columna con una escala estandar
    @param array_cols:  Arreglo con columnas a normalizar
    @param df_review:   Dataframe con características de la señal
    @return:            Dataframe con columnas normalizadas con una escala estandar
    """
    def normalize_data(self,array_cols,df_review):
        scaler = StandardScaler()
        df_review[array_cols] = scaler.fit_transform(df_review.loc[:,array_cols])
        return df_review
    
    """
    Graficando dataframe
    @param df_review:   Dataframe con características de la señal
    @return:            grafico
    """
    def graphing_data(self, df_review):
        df_review.hist(edgecolor="red")
        scatter_matrix(df_review,figsize=(15,15))
        sns.pairplot(df_review, hue="Target_Emotion")
        plt.show()
    
    """
    Crea los modelos de predicción de emociones
    @param df_review:   Dataframe con características de la señal
    @param array_X:     Columnas (variables del HRV) seleccionadas par clasificar
    @return:            Conjunto de entrenamiento variable, emoción
                        Conjunto de prueba variable, emoción
    """
    def creating_model(self, df_review, array_X):
        train, test = train_test_split(df_review, test_size=0.3, random_state=30)
        train_x, train_y = train.loc[:,array_X], train['Target_Emotion']
        test_x, test_y = test.loc[:,array_X], test['Target_Emotion']
        count_train_y = train_y.value_counts()
        return train_x, train_y, test_x, test_y
    
    """
    Entrenando modelos para clasificar las emociones
    @param array_model: Arreglo con los modelos utilizados
    @param train_x:     Conjunto de entrenamiento de variables
    @param trainy_y:    Conjunto de entrenamiento de emociones según variables
    @return:            Arreglo con variables y emociones entrenados
    """
    def training_model(self, array_model, train_x, train_y):
        array_result_training = []
        for model in array_model:
            train = model.fit(train_x, train_y)
            array_result_training.append(train)
        return array_result_training
    
    """
    Obtiene el promedio de precisión de resultados de cada modelo
    @param array_result_training:   Arreglo con resultados de entrenamiento
    @param array_laberls_models:    Arreglo con nombre de los modelos
    @param test_x:                  Conjunto de prueba para las variables
    @param test_y:                  Conjunto de entrenamiento para las emociones
    @return:                        Procentaje de precisión    
    """
    def get_score_training(self, array_result_training, array_labels_models, test_x, test_y):
        count = 0
        for model in array_result_training:
            print(str(array_labels_models[count]) + ": " + str(model.score(test_x, test_y)))
            count = count + 1
            
            
    """
    Clasificando emociones según variables de los latidos cardiacos
    @param model:   Modelo de clasificación
    @param test_x:  Conjunto de prueba de variables
    @param test_y:  Conjunto de prueba de emociones
    @param emotions:Emociones no repetidas
    @return:        Matriz de confusión del modelo
    """
    def classifying(self, model, test_x, test_y, emotions):
        f1 = f1_score(test_y, model.predict(test_x),
            labels=emotions,
            average=None)

        print(classification_report(test_y, 
                                    model.predict(test_x),
                                    labels=emotions))

        conf_mat = confusion_matrix(test_y, 
                                    model.predict(test_x), 
                                    labels=emotions)
        return conf_mat
    
    """
    Mejorando el modelo para tener una mayor precisión
    @param model:   Modelo de entrenamiento a mejorar
    @param train_x: Conjunto de entrenamiento de variables cardiacas
    @param train_y: Conjunto de entrenamiento de emociones
    @return:        Modelo mejorado
    """
    def improving_model(self, model, train_x, train_y):
        params = {'C':[1,4,8,16,32], 'kernel':['linear','rbf']}
        svc = SVC()
        svc_grid = GridSearchCV(svc,params,cv=5)
        return svc_grid.fit(train_x,train_y)
     
    
a = Classification('DREAMER_preprocesado.csv')
    
