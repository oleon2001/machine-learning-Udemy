import matplotlib.pyplot as plt  # Visualization
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation

# Load dataset
data = pd.read_csv(
    "/home/oleon/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv"
)

# Separate features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

# Encode categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode the state column (categorical)
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Apply OneHotEncoder to the encoded column
# remainder='passthrough': Keeps numerical columns unchanged
ct = ColumnTransformer([("onehot", OneHotEncoder(), [3])], remainder="passthrough")
X = ct.fit_transform(X)

# Avoid dummy variable trap
X = X[:, 1:]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit multiple linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions on the test set
y_pred = regressor.predict(X_test)

# Backward Elimination
import statsmodels.api as sm

def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 

SL = 0.05
# Ajustar X_opt según el número de columnas en X
X_opt = X[:, np.arange(X.shape[1])]  # Selecciona todas las columnas
X_Modeled = backwardElimination(X_opt.copy(), SL)  # Avoid modifying original data

# Scaling (optional, if needed)
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)