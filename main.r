#plantilla de preprocesado de datos

dataset = read.csv('~/Escritorio/Udemy-Machine_learning/original/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')

# tratamiento de los valores NA
dataset$Age = ifelse(is.na(dataset$Age),ave(dataset$Age, FUN=function(x) mean(x,na.rm=TRUE)),dataset$Age )
dataset$Age
dataset$Salary = ifelse(is.na(dataset$Salary),ave(dataset$Salary, FUN=function(x) mean(x,na.rm=TRUE)),dataset$Salary )
dataset$Salary

#este es el mismo codigo simplificado
#dataset$Age = ifelse(is.na(dataset$Age), mean(dataset$Age, na.rm = TRUE), dataset$Age)
#este es el mismo codigo simplificado
#dataset$Salary = ifelse(is.na(dataset$Salary), mean(dataset$Salary, na.rm = TRUE), dataset$Salary)



#codificar las variables categoricas
#c es para trabajar con vectorews
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain', 'Germany'),
                         labels = c(1,2,3))

dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain', 'Germany'),
                         labels = c(1,2,3))


