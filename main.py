#IMPORTANTE!!!

'''
SE SIGUEN LOS SIGUIENTES PROCESOS CON ESTE DATASET
PRIMERO: SE IMPORTA EL DATASET Y SE SEPARA LAS CARACTERISTICAS QUE SERIAN LAS EL SUBCONJUNTO "X"
QUE REPRESENTAN LAS CARACTERISTICAS DEL CONJUNTO (FEATURES) Y "Y"  QUE SON LAS VARIABLES A PREDECIR (TARGET)
SEGUNDO: SE TRATA LAS VARIABLES NA DEL DATASET
TERCERO:CONVIERTE EN VARIABLES CATEGORICAS UNA PARTE DEL DATASET
CUARTO:dividir el data set en conjunto de entrenamiento y conjunto de testing
QUINTO: ESCALAR LAS VARIABLES

LO ESCRIBO DE ESTA FORMA PORQUE EL SEGUNDO Y EL TERCERO NO SE PUEDEN LLLEGAR A USAR POR QUE NO
'''



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



#dividir el data set en conjunto de entrenamiento y conjunto de testing


#Esta línea importa la función train_test_split de la biblioteca scikit-learn, que se utiliza para dividir un conjunto de datos en conjuntos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)


'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0): Esta línea de código realiza la división de los datos.
x: Este es el conjunto de características (features) que se utilizarán para entrenar el modelo.
y: Este es el conjunto de la variable objetivo (target) que se desea predecir.
test_size=0.2: Este parámetro indica que el 20% de los datos se utilizarán para el conjunto de prueba, mientras que el 80% restante se utilizará para el conjunto de entrenamiento. Esto es una práctica común para asegurarse de que el modelo tenga suficientes datos para aprender y también para evaluar su rendimiento.
random_state=0: Este parámetro establece una semilla para el generador de números aleatorios. Esto asegura que la división de los datos sea reproducible. Si ejecutas el código varias veces con el mismo random_state, obtendrás la misma división de datos cada vez.
'''
#escalado de variables
#Esta línea importa la clase StandardScaler de la biblioteca scikit-learn. StandardScaler se utiliza para escalar características (features) de manera que tengan una media de 0 y una desviación estándar de 1.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #sc_X = StandardScaler(): Aquí se crea una instancia del escalador. Este objeto se utilizará para ajustar y transformar los datos.

'''
Ajuste (fit): Calcula la media y la desviación estándar de las características en el conjunto de entrenamiento (x_train).
Transformación (transform): Escala las características del conjunto de entrenamiento utilizando la media y la desviación estándar calculadas. El resultado es que cada característica tendrá una media de 0 y una desviación estándar de 1.
'''
x_train= sc_X.fit_transform(x_train)
'''
Esta línea transforma el conjunto de prueba (x_test) utilizando la misma media y desviación estándar 
que se calcularon a partir del conjunto de entrenamiento. Es importante no ajustar el escalador 
'''
x_test=sc_X.transform(x_test)

