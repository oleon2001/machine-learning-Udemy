#plantilla de preprocesado de datos

dataset=read.csv('~/Escritorio/Udemy-Machine_learning/original/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')


#install.packages("caTools")
#dividir los datos en conjunto de entrenamiento y conjunto de prueba 
library(caTools)
set.seed(123)
split=sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
testing_Set = subset(dataset, split==FALSE)

#escalado de valores
#training_set[,2:3]= scale(training_set[,2:3])
#testing_Set[,2:3]= scale(testing_Set[,2:3])


#ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#predecir resultados con el conjunto de test
y_predict= predict(regressor, newdata = testing_Set)


#visualizacion de los resultados en el conjunto de entrenamiento
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), colour="red") + 
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)), 
            colour="blue") +
  ggtitle("Sueldo vs años de experiencia (conjunto de entrenamiento)") +
  xlab("años de experiencia") +
  ylab("sueldo ( en $)")

#visualizacion de los resultados en el conjunto de testing
library(ggplot2)
ggplot() + 
  geom_point(aes(x = testing_Set$YearsExperience, y = testing_Set$Salary), colour="red") + 
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)), 
            colour="blue") +
  ggtitle("Sueldo vs años de experiencia (conjunto de prueba)") +
  xlab("años de experiencia") +
  ylab("sueldo ( en $)")

