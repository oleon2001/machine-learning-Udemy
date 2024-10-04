#regresion lineal multiple

dataset=read.csv('~/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

#codificar las variables categoricas
#c es para trabajar con vectorews
dataset$State = factor(dataset$State,
                         levels = c('New York','California', 'Florida'),
                         labels = c(1,2,3))


#install.packages("caTools")
#dividir los datos en conjunto de entrenamiento y conjunto de prueba 
library(caTools)
set.seed(123)
split=sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
testing_Set = subset(dataset, split==FALSE)


#ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento 
#regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
regression = lm(formula = Profit ~ R.D.Spend, data = training_set)

# Predecir los resultados con el conjunto de testing 
y_pred = predict(regression, newdata = testing_Set)

#Construir un modelo de regresion lineal hacia atras
SL=0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State , 
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration , 
                data = dataset)
summary(regression)


regression = lm(formula = Profit ~ R.D.Spend  , 
                data = dataset)
summary(regression)

