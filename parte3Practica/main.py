import matplotlib.pyplot as plt  # Visualización
import numpy as np  # Operaciones numéricas
import pandas as pd  # Manipulación de datos

# Cargar dataset
data = pd.read_csv('/home/oleon/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Separar características y variable objetivo
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

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

# Ajustar modelo de regresión lineal múltiple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = regressor.predict(X_test)

# Eliminación hacia atrás (Backward Elimination)
import statsmodels.api as sm

# Agregar columna de unos para el término independiente
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# Nivel de significancia (ajustar según criterio)
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # Iniciar con todas las variables
X_opt= np.array(X_opt,dtype=float)
# Iterar hasta que todas las variables sean significativas
# ... (Implementar el bucle de eliminación hacia atrás)
# Crear el modelo OLS final
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())


X_opt = X[:, [0, 1, 3, 4, 5]]  # Iniciar con todas las variables
X_opt= np.array(X_opt,dtype=float)
# Iterar hasta que todas las variables sean significativas
# ... (Implementar el bucle de eliminación hacia atrás)
# Crear el modelo OLS final
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())


X_opt = X[:, [0, 3, 4, 5]]  # Iniciar con todas las variables
X_opt= np.array(X_opt,dtype=float)
# Iterar hasta que todas las variables sean significativas
# ... (Implementar el bucle de eliminación hacia atrás)
# Crear el modelo OLS final
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())


X_opt = X[:, [0, 3, 5]]  # Iniciar con todas las variables
X_opt= np.array(X_opt,dtype=float)
# Iterar hasta que todas las variables sean significativas
# ... (Implementar el bucle de eliminación hacia atrás)
# Crear el modelo OLS final
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]  # Iniciar con todas las variables
X_opt= np.array(X_opt,dtype=float)
# Iterar hasta que todas las variables sean significativas
# ... (Implementar el bucle de eliminación hacia atrás)
# Crear el modelo OLS final
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())





# Escalado (opcional, si es necesario)
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)