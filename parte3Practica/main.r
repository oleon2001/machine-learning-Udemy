#plantilla de preprocesado de datos

dataset=read.csv('~/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')


#install.packages("caTools")
#dividir los datos en conjunto de entrenamiento y conjunto de prueba 
library(caTools)
set.seed(123)
split=sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
testing_Set = subset(dataset, split==FALSE)
