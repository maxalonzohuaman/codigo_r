############################################################
# Clases PAT - INSTITUTO CONTINENTAL
# Fecha actualizacion: 1ra ver 2019 | 2da ver julio 2020
# Elaborado por Max Alonzo
############################################################


#++++++++++++++++++++++++++++++++++++++++++++ ANALISIS EXPLORATORIO

# Data: iris

# Para remover todos los objetos de la memoria
rm(list=ls())

data = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
colnames(data) = c('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target')
head(data)
str(data)
summary(data)
plot(data$sepal_length, data$sepal_width)
boxplot(data$sepal_width)


# Matrices
x=matrix(data=c(1,2,3,4), nrow=2, ncol=2)
sqrt(x)
x^2

# Correlacion
set.seed(1303)
x=rnorm(50)
y=x+rnorm(50,mean=50,sd=.1)
cor(x,y)

# Estadisticas basicas
set.seed(3)
y=rnorm(100)
mean(y)
var(y)
sqrt(var(y))
sd(y)

# Graficas
x=rnorm(100)
y=rnorm(100)
plot(x,y)
plot(x,y,xlab="this is the x-axis",ylab="this is the y-axis",
       main="Plot of X vs Y")

# Histogramas
hist(data$sepal_length)
hist(data$sepal_length ,col =2)
hist(data$sepal_length ,col=2, breaks =15)


#++++++++++++++++++++++++++++++++++++++++++++ CLUSTERING - EJEMPLO 1

# Data: simulada

set.seed(2)
x = matrix(rnorm(100), ncol = 2)
x[1:25,1]=x[1:25,1]+3
x[1:25,2]=x[1:25,2]-4
km.out=kmeans (x,2, nstart =20)
km.out$cluster
plot(x, col=(km.out$cluster +1), main="K-Means con K=2", 
     xlab="", ylab="", pch=20, cex=2)

#++++++++++++++++++++++++++++++++++++++++++++ CLUSTERING - EJEMPLO 2

# Data: iris

data(iris)
head(iris)
library(ggplot2) # install.packages("ggplot2")
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()

set.seed(20)
irisCluster = kmeans(iris[, 3:4], 3, nstart = 20)
irisCluster
table(irisCluster$cluster, iris$Species)

irisCluster$cluster = as.factor(irisCluster$cluster)
ggplot(iris, aes(Petal.Length, Petal.Width, color = irisCluster$cluster)) + geom_point()

k.max = 10
wss = sapply(1:k.max,function(k){kmeans(iris[,3:4],k,nstart = 20,iter.max = 20)$tot.withinss})
wss
plot(1:k.max,wss, type= "b", xlab = "Numero de clusters(k)", ylab = "Dentro del cluster suma de cuadrados")


#++++++++++++++++++++++++++++++++++++++++++++ REGRESION LINEAL - EJEMPLO 1

# Data: simulada

# Regresion Lineal - Aproximacion area del triangulo
base100 = read.csv("C:/Users/Max Alonzo/Desktop/ESCRITORIO/Dictado_clases/Cursos_Conti/Data_Analytics - PAT Ins Conti 2020/Demos/area_triangulo.csv", sep = ";")
head(base100)
modelo = lm(area ~ base + altura, data = base100) 
summary(modelo)
predict(modelo, data.frame(base = c(6), altura = c(2)))
predict(modelo, data.frame(base=c(6,8),altura=c(2,4)))


#++++++++++++++++++++++++++++++++++++++++++++ REGRESION LINEAL - EJEMPLO 2

# Data: Valores de casas (en Boston)

# Regresion simple
library(MASS) 
library(ISLR) # install.packages("ISLR")
head(Boston) #valor de las casas en los suburbios de Boston
summary(Boston)
lm.fit=lm(medv ~ lstat , data=Boston)
summary(lm.fit)

plot(Boston$lstat, Boston$medv)
abline(lm.fit)
predict(lm.fit, data.frame(lstat=c(5, 10, 15)), interval ="prediction")

# Regresion multiple
lm.fit = lm(medv ~ lstat + age , data=Boston)
summary(lm.fit)

lm.fit = lm(medv ~ .,data=Boston)
summary(lm.fit)


#++++++++++++++++++++++++++++++++++++++++++++ REGRESION LOGISTICA (CLASIFICACION)

# Data: Titanic (base de personas que naufragaron en accidente de la embarcacion Titanic)

library(dplyr)
library(titanic) # install.packages("titanic")

data("titanic_train")

d_test_0 = merge(titanic_test, titanic_gender_model, by = "PassengerId") # agrega al dataset titanic_test la columna PassengerId
summary(d_test_0)
d_test_0 = subset(d_test_0, !(is.na(d_test_0$Age)))
d_test_0 = subset(d_test_0, !(is.na(d_test_0$Fare)))
d_test_0 = d_test_0[,c(-1,-3,-8,-10,-11)]
d_test_0[,2] = as.factor(d_test_0[,2])
d_test_0[,7] = as.factor(d_test_0[,7])

d_train_0 = titanic_train
str(d_train_0)
summary(d_train_0)

# Eliminacion de variables no utiles
d_train_1 = d_train_0[,c(-1,-4,-9,-11,-12)]
head(d_train_1)

# Conversion a factor
d_train_1[,1] = as.factor(d_train_1[,1])
d_train_1[,3] = as.factor(d_train_1[,3])
str(d_train_1)

# Verificacion de balanceo de datos
cbind( frecuencia = table(d_train_1$Survived), 
       porcentaje = prop.table(table(d_train_1$Survived))*100)

# Valores na e imputacion
summary(d_train_1)
library(naniar) # install.packages("naniar")
vis_miss(d_train_1)
miss_var_summary(d_train_1)

library("missForest") # install.packages("missForest")
simulacion <- missForest(d_train_1)
d_train_1 = simulacion$ximp
summary(d_train_1)

# Correlacion
library(GGally) # install.packages("GGally")
ggpairs(d_train_1)

# Modelo logistico
reg_log = glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data=d_train_1, family = binomial)
summary(reg_log)

prob = predict(reg_log, d_test_0, type = "response")
clases = ifelse(prob > 0.5, 1, 0)

table(clases, d_test_0$Survived)
mean(clases == d_test_0$Survived)

