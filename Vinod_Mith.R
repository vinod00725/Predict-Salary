rm(list=ls(all=TRUE))
library(dplyr)

#Reading file into R
data=read.csv("train.csv")

#Checking structure and summary of data
str(data)
summary(data)

#Remove unique columns 
data1=data
data1$Candidate.ID=NULL
data1$CollegeCode=NULL
data1$CityCode=NULL
data1$Date.Of.Birth=NULL

#Checking duplicate
data1=data1[!duplicated(data1),]

#Data type conversion
data1$Year.of.Graduation.Completion=as.factor(data1$Year.of.Graduation.Completion)
data1$Year.Of.Twelth.Completion=as.factor(data1$Year.Of.Twelth.Completion)
data1$CollegeTier=as.factor(data1$CollegeTier)
data1$CityTier=as.factor(data1$CityTier)

#Dealing with missing values
sum(is.na(data1))
data2=data1
data2$Board.in.Twelth <- na_if(data2$Board.in.Twelth, '0') 
data2$School.Board.in.Tenth<-na_if(data2$School.Board.in.Tenth, '0')
data2$Score.in.Domain<-na_if(data2$Score.in.Domain, '-1')
summary(data2)
sum(is.na(data2)) #NA=11018
100*sum(is.na(data2))/(nrow(data2)*ncol(data2)) # NA = 1.39%
data2=na.omit(data2) # NA's are less than 20% so we can omit them
str(data2)

#Visualizations
boxplot(data2)
plot(data2$Discipline)
plot(data2$Graduation)
#Split data into train and test
set.seed(007)
data3=data2
data3$Year.Of.Twelth.Completion=NULL
trainRows=sample(x=1:nrow(data3),size=0.70*nrow(data3))
train_data = data3[trainRows,] 
test_data = data3[-trainRows,]

#(1) Build rpart model on the training dataset
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(DMwR)
model1 <- rpart(Pay_in_INR ~ . , train_data,method="anova")
model1<-rpart(Pay_in_INR~.,data=train_data,method="anova",control = rpart.control(cp = 0.001))
printcp(model1)
print(model1)
plot(model1)

#Predicting on train & test data
predTrain=predict(model1, newdata=train_data, type="vector")
predTest=predict(model1, newdata=test_data, type="vector")

#Evaluation Matrix
regr.eval(train_data[,"Pay_in_INR"], predTrain) #rmse=1.85e+05
regr.eval(test_data[,"Pay_in_INR"], predTest) #rmse=2.32e+05

# (2) multiple regression
model2 <- lm(formula = Pay_in_INR  ~. , data = train_data) 
summary(model2)
#Plot
par(mfrow=c(2,2))
plot(model2)

#Predicting on train & test data
preds_model2 <- predict(model2, train_data[, !(names(train_data) %in% c("Pay_in_INR"))])
regr.eval(train_data$Pay_in_INR, preds_model2) #rmse=2.45e+05
preds_model <- predict(model2, test_data[, !(names(test_data) %in% c("Pay_in_INR"))])
regr.eval(test_data$Pay_in_INR, preds_model2) #rmse= 5.24e+05


#(3) PCA analysis should be done on prediction variable
#Read the data
pcadata=read.csv("train.csv",header = TRUE,sep = ",")

#Data conversions
pcadata$Candidate.ID=NULL
pcadata$CollegeCode=NULL
pcadata$CityCode=NULL
pcadata$Date.Of.Birth=NULL
str(pcadata)
cat_Attr=c(4,7,9,10,13)
num_Attr=-c(2,4,7,9,10,13)
cat_Data <- data.frame(sapply(data[,cat_Attr], as.factor))
num_Data <- data.frame(sapply(data[,num_Attr], as.numeric))
predictor=num_Data[,-1]
#since the predictors are of completely different magnitude,
#we need to scale them before the analysis
scaled.predictor=scale(predictor)
scaled.predictor
# Compute PC's
pca.out = princomp(scaled.predictor)
summary(pca.out)
plot(pca.out)


#If we choose 70% explonatory power for variances , we need only first 13 components of PC.
compressed_feature=pca.out$scores[,1:13]
compressed_feature

library(nnet)
library(DMwR)

multiout.pca=multinom(pcadata$Pay_in_INR~compressed_feature)
summary(multiout.pca)
regr.eval(pcadata$Pay_in_INR,train_data)  ##RMSE=5.18e+05
regr.eval(pcadata$Pay_in_INR,test_data) ##RMSE=7.12e+05

#(4)xgboost
# Packages
library(xgboost)
library(magrittr)
library(Matrix)

# Create matrix - One-Hot Encoding for Factor variables
trainm <- sparse.model.matrix(Pay_in_INR ~ .-1, data = train_data)
head(trainm)
train_label <- train_data[,"Pay_in_INR"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

testm <- sparse.model.matrix(Pay_in_INR~.-1, data = test_data)
test_label <- test_data[,"Pay_in_INR"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

# Parameters
nc <- length(unique(train_label))
xgb_params <- list("objective" = "reg:linear",
                   "eval_metric" = "rmse")
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1000,
                       watchlist = watchlist,
                       eta = 0.001,
                       lambda= 0,
                       seed = 333)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

min(e$test_mlogloss)
e[e$test_mlogloss == 0.625217,]

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)

regr.eval(test_data$Pay_in_INR,p) #rmse=3.51e+05
regr.eval(train_data$Pay_in_INR,p) #rmse=3.87e+05


#Predictions in test data
testpred=read.csv("test.csv")
str(testpred)
testpred1=testpred
testpred1$Candidate.ID=NULL
testpred1$CollegeCode=NULL
testpred1$CityCode=NULL
testpred1$Date.Of.Birth=NULL

#Checking duplicate
testpred1=testpred1[!duplicated(testpred1),]

#Data type conversion
testpred1$Year.of.Graduation.Completion=as.factor(testpred1$Year.of.Graduation.Completion)
testpred1$Year.Of.Twelth.Completion=as.factor(testpred1$Year.Of.Twelth.Completion)
testpred1$CollegeTier=as.factor(testpred1$CollegeTier)
testpred1$CityTier=as.factor(testpred1$CityTier)

#Dealing with missing values
sum(is.na(testpred1))
testpred2=testpred1
testpred2$Board.in.Twelth <- na_if(testpred2$Board.in.Twelth, '0') 
testpred2$School.Board.in.Tenth<-na_if(testpred2$School.Board.in.Tenth, '0')
testpred2$Score.in.Domain<-na_if(testpred2$Score.in.Domain, '-1')
summary(testpred2)
sum(is.na(testpred2)) #NA=4563
100*sum(is.na(testpred2))/(nrow(testpred2)*ncol(testpred2)) # NA = 1.44%
testpred2=na.omit(testpred2) # NA's are less than 20% so we can omit them
str(testpred2)
testpred3=testpred2
testpred3$Year.Of.Twelth.Completion=NULL
prediction<- predict(model1, testpred3, type="vector")
output <- cbind(testpred3, prediction)
write.csv(output, file = "output.csv")
