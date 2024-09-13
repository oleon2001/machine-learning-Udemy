dataset = read.csv('~/Escritorio/Udemy-Machine_learning/original/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')

#codificar las variables categoricas
#c es para trabajar con vectorews
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain', 'Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('Yes','No'),
                           labels = c(1,2))

dataset$Country

dataset$Purchased
