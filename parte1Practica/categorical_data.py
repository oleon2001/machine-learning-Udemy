import matplotlib.pyplot as plt  # Importa la biblioteca matplotlib para crear gráficos. Se utiliza 'pyplot' para acceder a funciones de visualización.
import numpy as np  # Importa la biblioteca NumPy, que proporciona soporte para arreglos y operaciones matemáticas.
import pandas as pd  # Importa la biblioteca pandas, que se utiliza para la manipulación y análisis de datos.


# Cargar el dataset
dataset = pd.read_csv('/home/oleon/Escritorio/Udemy-Machine_learning/original/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')
# Utiliza la función read_csv de pandas para cargar un archivo CSV en un DataFrame. 
# El DataFrame es una estructura de datos bidimensional que permite almacenar datos en forma de tabla.

# Separar las características y la variable objetivo
x = dataset.iloc[:, :-1].values  # Selecciona todas las filas y todas las columnas excepto la última. 
# Esto se almacena en 'x', que representa las características (features) del conjunto de datos.
y = dataset.iloc[:, 3].values  # Selecciona todas las filas de la cuarta columna (índice 3). 
# Esto se almacena en 'y', que representa la variable objetivo (target) que se desea predecir.

# Codificar datos categóricos desde la columna  Country


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer  # Importa ColumnTransformer


# Codificar la primera columna (categórica)
#LabelEncoder: Esta clase se utiliza para convertir etiquetas categóricas en números enteros. Por ejemplo, si tienes una columna con valores como "Rojo", "Verde" y "Azul", LabelEncoder los transformará en 0, 1 y 2, respectivamente.
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])  # Codifica la columna categórica

# Usar ColumnTransformer para aplicar OneHotEncoder
#En el argumento de ColumnTransformer, se pasa una lista de transformaciones. En este caso, se aplica OneHotEncoder a la primera columna (índice 0) del array x.
#remainder='passthrough' significa que todas las columnas que no se especifican en la lista de transformaciones se mantendrán sin cambios en el resultado. Esto es útil para conservar las características numéricas que no necesitan ser transformadas.
ct = ColumnTransformer([("onehot", OneHotEncoder(), [0])], remainder='passthrough')  # Aplica OneHotEncoder a la primera columna
# fit_transform se utiliza en el objeto ct para ajustar el ColumnTransformer a los datos de x y aplicar las transformaciones especificadas.
x = ct.fit_transform(x)  

'''
La columna Country es categórica y necesita ser convertida a un 
formato numérico. Usamos LabelEncoder para convertir los nombres de los países en números enteros 
y luego OneHotEncoder para crear variables binarias.
'''

print(x)  # Imprime el array 'x' después de la transformación

labelencoder_y=LabelEncoder()
y= labelencoder_y.fit_transform(y)

print(y)