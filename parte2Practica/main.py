#regresion lineal simple


import matplotlib.pyplot as plt  # Importa la biblioteca matplotlib para crear gráficos. Se utiliza 'pyplot' para acceder a funciones de visualización.
import numpy as np  # Importa la biblioteca NumPy, que proporciona soporte para arreglos y operaciones matemáticas.
import pandas as pd  # Importa la biblioteca pandas, que se utiliza para la manipulación y análisis de datos.

# Cargar el dataset
dataset = pd.read_csv('/home/oleon/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')

# Separar las características y la variable objetivo
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values  



#dividir el data set en conjunto de entrenamiento y conjunto de testing


#Esta línea importa la función train_test_split de la biblioteca scikit-learn, que se utiliza para dividir un conjunto de datos en conjuntos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=1/3, random_state=0)



'''
#escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 

x_train= sc_X.fit_transform(x_train)

x_test=sc_X.transform(x_test)
'''

#crear modelo de regresion lineal simple con el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regresion= LinearRegression()
regresion.fit(x_train,y_train)

#predecir el conjunto de test
y_pred= regresion.predict(x_test)

#visualizacion de los datos de entrenamiento
def visualizar_grafica_entrenamiento():
    plt.scatter(x_train, y_train, color="red")
    plt.plot(x_train, regresion.predict(x_train), color="blue")
    plt.title("Sueldo vs años de experiencia (conjunto de entrenamiento)")
    plt.xlabel("años de experiencia")
    plt.ylabel("sueldo en $")
    return plt.show()

def visualizar_grafica_prueba():
    plt.scatter(x_test, y_test, color="red")
    plt.plot(x_train, regresion.predict(x_train), color="blue")
    plt.title("Sueldo vs años de experiencia (conjunto de prueba)")
    plt.xlabel("años de experiencia")
    plt.ylabel("sueldo en $")
    return plt.show()
    
    
visualizar_grafica_entrenamiento()

visualizar_grafica_prueba()


