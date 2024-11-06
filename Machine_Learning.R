#Maschine Learning
#Module 1

library(kernlab)
data(spam)
head(spam) 

plot(density(spam$your[spam$type=="nonspam"]),
     col="blue",main="",xlab="Frequency of 'your'")
lines(density(spam$your[spam$type=="spam"]),col="red")


#predict SPAM based on frequency of "your"

plot(density(spam$your[spam$type=="nonspam"]),
     col="blue",main="",xlab="Frequency of 'your'")
lines(density(spam$your[spam$type=="spam"]),col="red")
abline(v=0.5,col="black")

prediction <- ifelse(spam$your > 0.5,"spam","nonspam")
table(prediction,spam$type)/length(spam$type)
#Output shows that the algorithm is about 75% accurate -> 45% for nonspam + 30% spam



# In sample vs out of sample errors
# Spam messages based on capital letters
set.seed(333)
smallSpam <- spam[sample(dim(spam)[1],size=10),]
spamLabel <- (smallSpam$type=="spam")*1 + 1
plot(smallSpam$capitalAve,col=spamLabel)

rule1 <- function(x){
  prediction <- rep(NA,length(x))
  prediction[x > 2.7] <- "spam"
  prediction[x < 2.40] <- "nonspam"
  prediction[(x >= 2.40 & x <= 2.45)] <- "spam"
  prediction[(x > 2.45 & x <= 2.70)] <- "nonspam"
  return(prediction)
}
table(rule1(smallSpam$capitalAve),smallSpam$type)


rule2 <- function(x){
  prediction <- rep(NA,length(x))
  prediction[x > 2.8] <- "spam"
  prediction[x <= 2.8] <- "nonspam"
  return(prediction)
}
table(rule2(smallSpam$capitalAve),smallSpam$type)

table(rule1(spam$capitalAve),spam$type)
table(rule2(spam$capitalAve),spam$type)
mean(rule1(spam$capitalAve)==spam$type)
mean(rule2(spam$capitalAve)==spam$type)


sum(rule1(spam$capitalAve)==spam$type)
sum(rule2(spam$capitalAve)==spam$type)
#simple rule (rule 1) does better than complicated rule (rule 1)
#overfitting


#Module 2 / caret package

install.packages("caret")
library(caret)
library(kernlab)
data(spam)

inTrain <- createDataPartition(y=spam$type, p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
head(inTrain)
dim(training)
training


set.seed(32343)
modelFit <- train(type ~., data = training, method = "glm")
modelFit

modelFit$finalModel

predictions <- predict(modelFit, newdata = testing)
predictions

confusionMatrix(predictions, testing$type)

# data slicing

set.seed(32323)
folds <- createFolds(y=spam$type,k=10,
                     list=TRUE,returnTrain=TRUE)
sapply(folds,length)
folds[[1]][1:10]

folds <- createResample(y=spam$type, times=10, list = TRUE)
sapply(folds, length)

set.seed(32323)
tme <- 1:1000
folds <- createTimeSlices(y=tme,initialWindow=20,
                          horizon=10)
names(folds)
folds$train[[1]]
folds$test[[1]]

#Training options
args(train.default) #not working 

args(trainControl)

#plotting predictors

install.packages("ISLR")
library(ISLR)
library(ggplot2)
library(caret)

data(Wage)
summary(Wage)

inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]

dim(training)
dim(testing)

featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")

head(Wage)
ggplot(training, aes(x = age, y = wage, color = education)) +
  geom_point() +
  geom_smooth(method = "lm", formula=y~x)

fit <- lm(wage ~ age + education , data = Wage)
summary(fit)



install.packages("Hmisc")
library(Hmisc)

cutWage <- cut2(training$wage, g=3)
table(cutWage)

summary(Wage)

ggplot(training, aes(cutWage, age, fill = cutWage))+
  geom_boxplot()

t1 <- table(cutWage, training$jobclass)
t1

prop.table(t1,1)

ggplot(training, aes(wage, color = education)) +
  geom_density()

qplot(wage, colour = education, data=training, geom = "density")


#Pre-processing predictor variables

library(kernlab)

data(spam)

inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve,main="",xlab="ave. capital run length")

mean(training$capitalAve)
sd(training$capitalAve)

#standardizing

trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve)) / sd(trainCapAve)

hist(trainCapAveS)

mean(trainCapAveS)
sd(trainCapAveS)
range(trainCapAveS)
range(trainCapAve)


preObj <- preProcess(training[,-58],method=c("center","scale")) #all training variable except 58
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)

testCapAveS <- predict(preObj,testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)

set.seed(32343)
modelFit <- train(type ~.,data=training,
                  preProcess=c("center","scale"),method="glm")
modelFit

## Standardizing - Box-Cox transforms

preObj <- preProcess(training[,-58],method=c("BoxCox"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
par(mfrow=c(1,2)); hist(trainCapAveS); qqnorm(trainCapAveS)

?qqnorm


## Standardizing - Imputing data
## Missing data
# impute data using k nearest neighours imputations

# Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1],size=1,prob=0.05)==1
training$capAve[selectNA] <- NA

# Impute and standardize
preObj <- preProcess(training[,-58],method="knnImpute")
capAve <- predict(preObj,training[,-58])$capAve

# Standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)

quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA])
quantile((capAve - capAveTruth)[!selectNA])


#Covariate creation
# covariates = predictors

spam$capitalAveSq <- spam$capitalAve^2
head(spam)
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

table(training$jobclass)

#dummy variables

dummies <- dummyVars(wage ~ jobclass, data = training)
head(predict(dummies, newdata = training))

nsv <- nearZeroVar(training, saveMetrics = TRUE)
nsv

library(splines)
bsBasis <- bs(training$age, df = 3)
bsBasis

lm1 <- lm(wage ~ bsBasis, data = training)
plot(training$age, training$wage, pch = 19, cex = 0.5)
points(training$age, predict(lm1, newdata = training), col = "red")

predict(bsBasis, age = testing$age)


#Pre-processing with PCA

inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

M <- abs(cor(training[,-58])) #58th columns of dataset is the outcome (spam or not spam)
head(training[,58])
M
diag(M) <- 0 #remove correlation of variables with themselves
which(M > 0.8,arr.ind=T) #what variables have a correlation >0.8

names(spam)[c(34,32)]
plot(spam[,34],spam[,32])

dev.off()

#how to combined to predictors
#use PCA

#rotate plot

X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training$num857
plot(X,Y)


smallSpam <- spam[, c(34, 32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1],prComp$x[,2])

prComp$rotation

typeColor <- ((spam$type=="spam")*1 + 1) #black if not spam and red if spam
prComp <- prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab="PC1",ylab="PC2")

#PCA in caret

preProc <- preProcess(log10(spam[,-58]+1),method="pca",pcaComp=2)
spamPC <- predict(preProc,log10(spam[,-58]+1))
plot(spamPC[,1],spamPC[,2],col=typeColor)

#Preprocessing with PCA

preProc <- preProcess(log10(training[,-58]+1),method="pca",pcaComp=2)
trainPC <- predict(preProc,log10(training[,-58]+1))
modelFit <- train(training$type ~ .,method="glm",data=trainPC)


testPC <- predict(preProc,log10(testing[,-58]+1))
confusionMatrix(testing$type,predict(modelFit,testPC))

modelFit <- train(training$type ~ .,method="glm",preProcess="pca",data=training)
confusionMatrix(testing$type,predict(modelFit,testing))

#Predicting with regression

library(caret);data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting,
                               p=0.5, list=FALSE)
trainFaith <- faithful[inTrain,]; testFaith <- faithful[-inTrain,]
head(trainFaith)

plot(trainFaith$eruptions, trainFaith$waiting, col = "blue", pch=19,
    xlab = "eruption time", ylab ="wainting time")

ggplot(trainFaith, aes(x = waiting, y = eruptions)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red")

# EDi = intercept (b0) + b1waiting + ei

lm1 <- lm(eruptions ~ waiting, data = trainFaith)
summary(lm1)

#predict new variable
coef(lm1)[1] + coef(lm1)[2]*80

newdata <- data.frame(waiting =80)
newdata

predict(lm1, newdata)

#Training and test set

par(mfrow=c(1,2))
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(trainFaith$waiting,predict(lm1),lwd=3)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(testFaith$waiting,predict(lm1,newdata=testFaith),lwd=3)

#get training set errors / RMSE
lm1$fitted
sqrt(sum((lm1$fitted - trainFaith$eruptions)^2))

#error on test
sqrt(sum((predict(lm1, newdata = testFaith) - testFaith$eruptions)^2))


#prediction intervals
pred1 <- predict(lm1,newdata=testFaith,interval="prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue")
matlines(testFaith$waiting[ord],pred1[ord,],type="l",,col=c(1,2,2),lty = c(1,1,1), lwd=3)

dev.off()

modFit <- train(eruptions ~ waiting,data=trainFaith,method="lm")
summary(modFit$finalModel)

#Prediction with regression using multiple covariates
library(ISLR); library(ggplot2); library(caret);
data(Wage); Wage <- subset(Wage,select=-c(logwage))
summary(Wage)

inTrain <- createDataPartition(y=Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]

dim(training)
dim(testing)

featurePlot(x = training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")

ggplot(training, aes(age, wage, color = education)) +
  geom_point(col = "black") +
  geom_smooth(method = "lm")

?geom_smooth

ggplot(training, aes(age, wage, color = education)) +
  geom_point()


#fit linear model with multiple variables

modFit<- train(wage ~ age + jobclass + education,
               method = "lm",data=training)
finMod <- modFit$finalModel
print(modFit)
summary(modFit)

#Diagnostics
plot(finMod,1,pch=19,cex=0.5,col="#00000010")

## Color by variables not used in the model 
qplot(finMod$fitted,finMod$residuals,colour=race,data=training)


# plot index
plot(finMod$residuals,pch=19)

## Predicted versus truth in test set
pred <- predict(modFit, testing)
qplot(wage,pred,colour=year,data=testing)

#predict with all variables

modFitAll<- train(wage ~ .,data=training,method="lm")
pred <- predict(modFitAll, testing)
qplot(wage,pred,data=testing)


#Quiz 2
#Q1
install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
data(AlzheimerDisease)

adData = data.frame(diagnosis, predictors)
head(adData)
trainIndex = createDataPartition(diagnosis, p = 0.5, list = FALSE)
training <- adData[trainIndex,]
testing <- adData[-trainIndex,]

dim(training)
dim(testing)

#Q2
data(concrete)
set.seed(1000)

inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[inTrain,]
testing = mixtures[-inTrain,]

head(concrete)

library(Hmisc)

cut <- cut2(training$wage, g=3)
?cut2

ggplot(concrete, aes(CompressiveStrength, Cement)) +
  geom_point()

plot(concrete$CompressiveStrength)

# Create a data frame with index and CompressiveStrength
concrete$Index <- 1:nrow(concrete)
head(concrete)


ggplot(concrete, aes(x = Index, y = CompressiveStrength, color = Age)) +
  geom_point() +
  geom_smooth(method = "lm", col = "red") +
  labs(x = "Index", y = "Compressive Strength") +
  theme_minimal()
  
#Answer: There is a non-random pattern in the plot of the outcome versus index that does not appear to be perfectly explained by any predictor suggesting a variable may be missing.
  

#Q3

hist(concrete$Superplasticizer)
#Contains values of 0 which would be infinite if log transformed


#Q4
install.packages("caret")
library(caret)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
set.seed(3433)

adData = data.frame(diagnosis, predictors)
head(adData)

inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]] 
inTrain
training = adData[inTrain,]
testing = adData[-inTrain,]

names(adData)
names(training)

#select all predictors with IL
dim(training)
il_columns <- grep("^IL_[0-9A-Za-z_]+", names(training), value = TRUE, ignore.case = TRUE)
il_columns
training_IL <- training[, il_columns]
head(training_IL)

training_IL$diagnosis <- training$diagnosis


preProc <- preProcess(training_IL,method="pca", thresh = 0.8)
PC <- predict(preProc,training_IL)
PC
#plot(PC[,1],PC[,2],col=diagnosis)

?preProcess


#Q5
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

#select all predictors with IL
il_columns <- grep("^IL_[0-9A-Za-z_]+", names(training), value = TRUE, ignore.case = TRUE)
il_columns
training_IL <- training[, il_columns]
head(training_IL)
training_IL$diagnosis <- training$diagnosis

#Model 1
mod1<- train(diagnosis ~ .,data=training_IL,method="glm")
summary(mod1)
#calculate accuracy
testingIL <- testing[,grep("^IL|diagnosis", names(testing))]
head(testingIL)
pred1 <- predict(mod1, newdata = testingIL)
matrix_model <- confusionMatrix(pred1, testingIL$diagnosis)
matrix_model$overall[1]
matrix_model


#Model 2
modelPCA <- train(diagnosis ~., data = training_IL, method = "glm", preProcess = "pca",trControl=trainControl(preProcOptions=list(thresh=0.8)))
matrix_modelPCA <- confusionMatrix(testingIL$diagnosis, predict(modelPCA, testingIL))
matrix_modelPCA$overall[1]


#Again
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# grep all columns with IL and diagnosis in the traning and testing set
trainingIL <- training[,grep("^IL|diagnosis", names(training))]
testingIL <- testing[,grep("^IL|diagnosis", names(testing))]

# non-PCA
model <- train(diagnosis ~ ., data = trainingIL, method = "glm")
predict_model <- predict(model, newdata= testingIL)
matrix_model <- confusionMatrix(predict_model, testingIL$diagnosis)
matrix_model$overall[1]


###
#Module 3
#Predicting with Trees

# 1. Start with all variables in one group
# 2. Find the variable/split that best seperates the outcomes
# 3. Divide the data into two groups ("leaves") on that split ("node")
# 4. Within each split, find the best variable/split that seperates the outcomes
# 5. Continue until the groups are too small or insufficiently pure

data(iris); library(ggplot2)
names(iris)
table(iris$Species)


inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

qplot(Petal.Width,Sepal.Width,colour=Species,data=training)

modFit <- train(Species ~ .,method="rpart",data=training)
print(modFit$finalModel)

plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

install.packages("rattle")
library(rattle)
fancyRpartPlot(modFit$finalModel)

predict(modFit,newdata=testing)

#Note that classification trees are non-linear models 

# Bagging
# Bootstrap aggregating
# Resample cases and recalculate predictions 
# Average or majority vote

install.packages("mlbench")
?ozone

library(ElemStatLearn); data(ozone,package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)

ll <- matrix(NA,nrow=10,ncol=155)
for(i in 1:10){
  ss <- sample(1:dim(ozone)[1],replace=T)
  ozone0 <- ozone[ss,]; ozone0 <- ozone0[order(ozone0$ozone),]
  loess0 <- loess(temperature ~ ozone,data=ozone0,span=0.2)
  ll[i,] <- predict(loess0,newdata=data.frame(ozone=1:155))
}

plot(ozone$ozone,ozone$temperature,pch=19,cex=0.5)
for(i in 1:10){lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155,apply(ll,2,mean),col="red",lwd=2)

predictors = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))


plot(ozone$ozone,temperature,col='lightgrey',pch=19)
points(ozone$ozone,predict(treebag$fits[[1]]$fit,predictors),pch=19,col="red")
points(ozone$ozone,predict(treebag,predictors),pch=19,col="blue")

library(caret)

ctreeBag$fit
ctreeBag$pred
ctreeBag$aggregate


### Random Forest

data(iris); library(ggplot2)
head(iris)
library(randomForest)

# 1. Boostrap samples
# 2. At each split, boostrap variables
# 3. Grow multiple trees and vote

inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species~ .,data=training,method="rf",prox=TRUE) #method rf for random forest
modFit
getTree(modFit$finalModel,k=2)

irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP)
irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col=Species,data=training)
p + geom_point(aes(x=Petal.Width,y=Petal.Length,col=Species),size=5,shape=4,data=irisP)

pred <- predict(modFit,testing); testing$predRight <- pred==testing$Species
table(pred,testing$Species)

qplot(Petal.Width,Petal.Length,colour=predRight,data=testing,main="newdata Predictions")

### Boosting

# Take lots of possibly weak predictors and weigh them and add them up
# Thereby getting stronger predictors

library(ISLR); data(Wage); library(ggplot2); library(caret);
Wage <- subset(Wage,select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

modFit <- train(wage ~ ., method="gbm",data=training,verbose=FALSE)
print(modFit)

qplot(predict(modFit,testing),wage,data=testing)

###
# Model based prediction

data(iris); library(ggplot2)
names(iris)
table(iris$Species)

inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

modlda = train(Species ~ .,data=training,method="lda")
modnb = train(Species ~ ., data=training,method="nb")
plda = predict(modlda,testing); pnb = predict(modnb,testing)
table(plda,pnb)        

equalPredictions = (plda==pnb)
qplot(Petal.Width,Sepal.Width,colour=equalPredictions,data=testing)

####
#Quiz

# Question 1

library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

head(segmentationOriginal)
dim(segmentationOriginal)

inTrain <- createDataPartition(y=segmentationOriginal$Case,
                               p=0.6, list=FALSE)

training <- segmentationOriginal[inTrain,]
testing <- segmentationOriginal[-inTrain,]
dim(training); dim(testing)


set.seed(125)

modFit <- train(Case ~ .,method="rpart",data=training)
modFit
modFit$finalModel

# Question 3

install.packages("pgmm")
library(pgmm)
data(olive)
olive = olive[,-1]
head(olive)
unique(olive$Area)

inTrain <- createDataPartition(y=olive$Area,
                               p=0.7, list=FALSE)
training <- olive[inTrain,]
testing <- olive[-inTrain,]
dim(training); dim(testing)

modFit <- train(Area~ .,data=training,method="rf",prox=TRUE) #method rf for random forest
modFit
getTree(modFit$finalModel,k=2)


oliveP <- classCenter(training[,c(3,4)], training$Area, modFit$finalModel$prox)
#irisP <- as.data.frame(irisP)
newdata = as.data.frame(t(colMeans(olive)))

olive$Area <- rownames(newdata)
head(olive)
head(newdata)

#p <- qplot(Palmitic, Palmitoleic, col=Area,data=training)
#p + geom_point(aes(x=Palmitic,y=Palmitoleic,col=Area),size=5,shape=4,data=olive)

pred <- predict(modFit,testing)
#testing$predRight <- pred==testing$Species
#table(pred,testing$Species)

#qplot(Petal.Width,Petal.Length,colour=predRight,data=testing,main="newdata Predictions")

