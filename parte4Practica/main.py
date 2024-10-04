#regresion polinomica

import matplotlib.pyplot as plt  # Visualización
import numpy as np  # Operaciones numéricas
import pandas as pd  # Manipulación de datos

# Cargar dataset
data = pd.read_csv('/home/oleon/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
# Separar características y variable objetivo
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

#no es necesario didividir el conjunto de entrenamiento dado a que hay muy pocos datos y ademas se tiene que usar toda la informacion del modelo
"""
# Codificar variable categórica
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Codificar la columna de estado (categórica)
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Aplicar OneHotEncoder a la columna codificada
# remainder='passthrough': Mantiene las columnas numéricas sin cambios
ct = ColumnTransformer([("onehot", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Evitar la trampa de las variables ficticias
X = X[:, 1:]

# Dividir datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

#ajustar modelo de regresion lineal (prueba pero solo es demostrativo)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X, y)

#ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
X_poly= poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualizacion del modelo lineal

plt.scatter(X, y, color= "red")
plt.plot(X, lin_reg.predict(X),color="blue")
plt.title("modelo de regresion lineal")
plt.xlabel("posicion")
plt.ylabel("salario")
plt.show()







#visualizacion del modelo polinomico 

x_grid= np.arange(min(X), max(X),0.1)
x_grid= x_grid.reshape(len(x_grid),1)
"""
plt.plot():

    Esta función de Matplotlib se utiliza para crear una gráfica.
    Toma como entrada los datos que se van a representar en los ejes x e y.
    El argumento color="blue" especifica que la línea que se va a trazar en la gráfica será de color azul.

X:

    Representa los valores de la variable independiente (eje x) de nuestro conjunto de datos.

lin_reg_2.predict(poly_reg.fit_transform(X)):

    Esta parte es un poco más compleja y se encarga de calcular los valores predichos por el modelo de regresión polinomial para cada valor de X.
    poly_reg.fit_transform(X):
        Transforma los datos de X en nuevas características polinomiales. Esto es necesario para ajustar un modelo de regresión polinomial, ya que este tipo de modelos busca relaciones no lineales entre las variables.
    lin_reg_2.predict():
        Utiliza el modelo de regresión lineal (lin_reg_2) para hacer predicciones sobre los datos transformados. Estas predicciones corresponden a los valores de la variable dependiente (eje y) que el modelo espera obtener para cada valor de X.
"""

plt.scatter(X, y, color="red", label="Datos Reales")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color="blue", label="Regresión Polinomial")

plt.title("Modelo de Regresión Polinomial")
plt.xlabel("Posición")
plt.ylabel("Salario")
plt.legend()
plt.show()

#prediccion de nuestras modelos
lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


