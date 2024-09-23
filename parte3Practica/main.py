#regresion lineal multiple


import matplotlib.pyplot as plt  # Importa la biblioteca matplotlib para crear gráficos. Se utiliza 'pyplot' para acceder a funciones de visualización.
import numpy as np  # Importa la biblioteca NumPy, que proporciona soporte para arreglos y operaciones matemáticas.
import pandas as pd  # Importa la biblioteca pandas, que se utiliza para la manipulación y análisis de datos.

# Cargar el dataset
dataset = pd.read_csv('/home/oleon/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Separar las características y la variable objetivo
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values  


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer  # Importa ColumnTransformer


# Codificar la primera columna (categórica)
#LabelEncoder: Esta clase se utiliza para convertir etiquetas categóricas en números enteros. Por ejemplo, si tienes una columna con valores como "Rojo", "Verde" y "Azul", LabelEncoder los transformará en 0, 1 y 2, respectivamente.
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])  # Codifica la columna categórica

# Usar ColumnTransformer para aplicar OneHotEncoder
#En el argumento de ColumnTransformer, se pasa una lista de transformaciones. En este caso, se aplica OneHotEncoder a la primera columna (índice 0) del array x.
#remainder='passthrough' significa que todas las columnas que no se especifican en la lista de transformaciones se mantendrán sin cambios en el resultado. Esto es útil para conservar las características numéricas que no necesitan ser transformadas.
ct = ColumnTransformer([("onehot", OneHotEncoder(), [3])], remainder='passthrough')  # Aplica OneHotEncoder a la primera columna
# fit_transform se utiliza en el objeto ct para ajustar el ColumnTransformer a los datos de x y aplicar las transformaciones especificadas.
x = ct.fit_transform(x)  

#evitar la trampa de las variables ficcticias
x=x[:,1:]


#dividir el data set en conjunto de entrenamiento y conjunto de testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regresion= LinearRegression()
regresion.fit(x_train,y_train)

'''
#escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 

x_train= sc_X.fit_transform(x_train)

x_test=sc_X.transform(x_test)
'''




