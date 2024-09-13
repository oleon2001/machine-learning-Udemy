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


# Tratamiento de los NAs (valores nulos)
from sklearn.impute import SimpleImputer  # Importa la clase SimpleImputer de scikit-learn, que se utiliza para manejar valores nulos.

# Crear el objeto SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Crea una instancia de SimpleImputer. 
# 'missing_values=np.nan' indica que los valores nulos se representan como np.nan. 
# 'strategy="mean"' especifica que los valores nulos se reemplazarán por la media de la columna correspondiente.

# Ajustar el imputer a los datos y transformar
imputer = imputer.fit(x[:, 1:3])  # Ajusta el imputer a los datos en las columnas 1 y 2 de 'x'. 
# Esto calcula la media de estas columnas, que se utilizará para reemplazar los valores nulos.
x[:, 1:3] = imputer.transform(x[:, 1:3])  # Aplica la transformación a las columnas 1 y 2 de 'x', reemplazando los valores nulos con la media calculada.

#print(x)  # Imprime el array 'x' después de haber reemplazado los valores nulos.

