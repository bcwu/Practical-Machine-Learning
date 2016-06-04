# Machine Learning Course Project - Prediction Assignment Writeup
Bincheng Wu  
May 30, 2016  

# Executive Summary

The goal of this project is to predict the manner in which 20 participants exercised by building prediction models based on data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

A model (Random Forest) is selected from four built on different method and a prediction for the 20 participants (testing data) is computed.

# Model Selection Strategy

The prediction model is selected based on best cross validation accuracy. 

Four models (Recursive Partitioning and Regression Trees, Support Vector Machines, Generalized Boosted Regression Models and Random Forest) are trained on a training subset without any tunning. The result models are then validated against a probe data set. 

Of the four models, that of Random Forest displayed the best accuracy (0.9921) when cross validated with the probe data set. 

While not the deciding factor, computation time is an important consideration in real world applications. The random Forest model took the longest time to train, with a eye-watering elapsed time of ~700 seconds even while parallelized.

In contrast, the SVM model took the an elapsed time of ~ 20 seconds to train while achieving an accuracy of (0.94), its combination of speed and accuracy is a competitive alternative.

# Load Libraries and Packages


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(AppliedPredictiveModeling)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```r
library(e1071)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
library(tictoc)
knitr::opts_chunk$set(cache=TRUE)
```

# Data Acquisition


```r
if(!file.exists("data")){dir.create("data")}
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#if(!file.exists("./data/pml-training.csv")){
download.file(fileUrl, destfile = "./data/pml-training.csv", mode = "wb")#}

if(!file.exists("data")){dir.create("data")}
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#if(!file.exists("./data/pml-testing.csv")){
download.file(fileUrl, destfile = "./data/pml-testing.csv", mode = "wb")#}

list.files("./data")
```

```
## [1] "pml-testing.csv"  "pml-training.csv"
```


```r
training_data = read.csv("./data/pml-training.csv", header = TRUE, stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))
testing_data = read.csv("./data/pml-testing.csv", header = TRUE, stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))

dim(training_data)
```

```
## [1] 19622   160
```

```r
dim(testing_data)
```

```
## [1]  20 160
```

# Data Cleaning

User and timestamps variables are removed since they are likely not predictive features of out of sample data sets. E.g. while the training data only have six unique participants, the testing data will contain 20. 

The 'classe' variable outcome is converted to factors to enable prediction model building.


```r
training_cleaned <- subset(training_data, select = -c(1:7)) # excluding user and timestamps

training_cleaned$classe <- as.factor(training_data$classe) # convert the outcomes to factors

dim(training_cleaned)
```

```
## [1] 19622   153
```

Feature variables are all converted to the numeric data class.


```r
for(i in 1:(ncol(training_cleaned)-1)) {training_cleaned[,i] <- as.numeric(as.character(training_cleaned[,i]))} 
```

No values are imputed. If a feature has one or more NA values, then it is excluded from the training data. 


```r
clean_feature_names <- colSums(is.na(training_cleaned)) == 0

training_cleaned <- training_cleaned[clean_feature_names]

dim(training_cleaned)
```

```
## [1] 19622    53
```

The same cleaning process is done for the testing data of the 20 participants in order to match the input formatting of the prediction model.


```r
testing_cleaned <- subset(testing_data, select = -c(1:7)) # excluding user and timestamps

dim(testing_cleaned)
```

```
## [1]  20 153
```

```r
for(i in 1:(ncol(testing_cleaned)-1)) {testing_cleaned[,i] <- as.numeric(as.character(testing_cleaned[,i]))} # convert feature variables to numeric

# testing_cleaned$classe <- as.factor(testing_cleaned$classe) # testing doesn't have the results, of course

clean_feature_names <- colSums(is.na(testing_cleaned)) == 0

testing_cleaned <- testing_cleaned[clean_feature_names]

dim(testing_cleaned)
```

```
## [1] 20 53
```

# Partitioning the Training and Validation Data Sets

As part of the prediction design, the training set is segmented into a training subset (60%) and a training probe data set (40%). Building the model on the training subset will prevent the model from overfitting. The probe data set will be used for validation.


```r
train_index <-createDataPartition(y=training_cleaned$classe, p=0.60,list=F)
training_subset<-training_cleaned[train_index ,] 
training_probe <-training_cleaned[-train_index ,] 
```

# Data Preprocessing

In short, no preprocessing is applied after no near zero variance features and correlated predictors are discovered.

### Check for Features' Variance

In principal component analysis (PCA), ideal features have high variance so that each feature is as distant(orthogonal) as possible from the others. 


```r
nzv = nearZeroVar(training_subset)
nzv
```

```
## integer(0)
```

If nearZeroVar returned a value greater than zero, then it means there are features without variability and thus need to be removed. But since it returned zero, then it means all features have enough variance.

### Identifying Correlated Predictors

While certain models benefit from correlated predictors, other models benefit from reducing the level of correlation between the predictors. 


```r
training_cor <- cor(training_subset[,1:52])
high_corr <- sum(abs(training_cor[upper.tri(training_cor)]) > .999)

high_corr
```

```
## [1] 0
```

Since there are no highly correlated ( >.999) features, no features are removed.

# Model Building

Four models (Decision Tree, Support Vector Machines, Generalized Boosted Regression Models and Random Forest) are built on the training subset.


```r
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(as.numeric(as.Date("2016-05-30")))

tic()
rpart_model <- rpart(classe~., data = training_subset, method = 'class')
rpart_train_time <- toc()
```

```
## 1.42 sec elapsed
```

```r
tic()
svm_model <- svm(classe ~., data = training_subset)
svm_train_time <- toc()
```

```
## 21.98 sec elapsed
```

```r
# svm_train_time <- svm_train_time$toc - svm_train_time$tic

tic()
gbm_model <- train(classe~., data = training_subset, method = 'gbm')
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2331
##      2        1.4584             nan     0.1000    0.1557
##      3        1.3583             nan     0.1000    0.1277
##      4        1.2778             nan     0.1000    0.1134
##      5        1.2085             nan     0.1000    0.0831
##      6        1.1564             nan     0.1000    0.0767
##      7        1.1085             nan     0.1000    0.0711
##      8        1.0634             nan     0.1000    0.0610
##      9        1.0240             nan     0.1000    0.0559
##     10        0.9880             nan     0.1000    0.0548
##     20        0.7537             nan     0.1000    0.0267
##     40        0.5303             nan     0.1000    0.0134
##     60        0.4067             nan     0.1000    0.0096
##     80        0.3260             nan     0.1000    0.0045
##    100        0.2661             nan     0.1000    0.0022
##    120        0.2219             nan     0.1000    0.0023
##    140        0.1892             nan     0.1000    0.0017
##    150        0.1758             nan     0.1000    0.0016
```

```r
gbm_train_time <- toc()
```

```
## 288.08 sec elapsed
```

```r
tic()
rf_model <- train(classe~., data = training_subset, method = 'rf')
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
rf_train_time <- toc()
```

```
## 813.17 sec elapsed
```

```r
stopCluster(cl)
#registerDoSEQ()
```

# Cross Validation

To select the more accurate model, the models are cross validated with the training probe data set to determine the out of sample error.

Random forest seems to give the best accuracy (~0.99), followed by gbm (~0.96), svm (~0.94) and rpart(~0.74).


```r
# Recursive Partitioning and Regression Trees cross validation
# fancyRpartPlot(rpart_model)
rpart_predict <- predict(rpart_model, newdata = training_probe, type = 'class')
confusionMatrix(rpart_predict, training_probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2017  227   15   55   19
##          B   90  913   76   96  107
##          C   57  160 1115  205  181
##          D   29  125   79  810   74
##          E   39   93   83  120 1061
## 
## Overall Statistics
##                                           
##                Accuracy : 0.754           
##                  95% CI : (0.7443, 0.7635)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6885          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9037   0.6014   0.8151   0.6299   0.7358
## Specificity            0.9437   0.9417   0.9069   0.9532   0.9477
## Pos Pred Value         0.8646   0.7122   0.6490   0.7252   0.7600
## Neg Pred Value         0.9610   0.9078   0.9587   0.9293   0.9409
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2571   0.1164   0.1421   0.1032   0.1352
## Detection Prevalence   0.2973   0.1634   0.2190   0.1424   0.1779
## Balanced Accuracy      0.9237   0.7716   0.8610   0.7915   0.8417
```

```r
# Support Vector Machines cross validation
svm_predict <- predict(svm_model, newdata = training_probe)
confusionMatrix(svm_predict, training_probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2223  141    0    1    0
##          B    2 1338   43    0   15
##          C    3   37 1305  112   42
##          D    0    1   11 1173   25
##          E    4    1    9    0 1360
## 
## Overall Statistics
##                                           
##                Accuracy : 0.943           
##                  95% CI : (0.9377, 0.9481)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9278          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9960   0.8814   0.9539   0.9121   0.9431
## Specificity            0.9747   0.9905   0.9701   0.9944   0.9978
## Pos Pred Value         0.9400   0.9571   0.8706   0.9694   0.9898
## Neg Pred Value         0.9984   0.9721   0.9901   0.9830   0.9873
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2833   0.1705   0.1663   0.1495   0.1733
## Detection Prevalence   0.3014   0.1782   0.1911   0.1542   0.1751
## Balanced Accuracy      0.9853   0.9360   0.9620   0.9532   0.9705
```

```r
# Generalized Boosted Regression Model cross validation
gbm_predict <- predict(gbm_model, newdata = training_probe)
confusionMatrix(gbm_predict, training_probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2196   59    0    1    2
##          B   25 1405   41    3    9
##          C    5   53 1309   41   14
##          D    3    0   15 1234    9
##          E    3    1    3    7 1408
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9625          
##                  95% CI : (0.9581, 0.9666)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9526          
##  Mcnemar's Test P-Value : 2.318e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9839   0.9256   0.9569   0.9596   0.9764
## Specificity            0.9890   0.9877   0.9826   0.9959   0.9978
## Pos Pred Value         0.9725   0.9474   0.9205   0.9786   0.9902
## Neg Pred Value         0.9936   0.9822   0.9908   0.9921   0.9947
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2799   0.1791   0.1668   0.1573   0.1795
## Detection Prevalence   0.2878   0.1890   0.1812   0.1607   0.1812
## Balanced Accuracy      0.9864   0.9566   0.9697   0.9777   0.9871
```

```r
# Random Forest cross validation
rf_predict <- predict(rf_model, newdata = training_probe)
confusionMatrix(rf_predict, training_probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   20    0    0    0
##          B    1 1495    9    1    1
##          C    1    3 1355   21    2
##          D    0    0    4 1264    1
##          E    1    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9917          
##                  95% CI : (0.9895, 0.9936)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9895          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9848   0.9905   0.9829   0.9972
## Specificity            0.9964   0.9981   0.9958   0.9992   0.9998
## Pos Pred Value         0.9911   0.9920   0.9805   0.9961   0.9993
## Neg Pred Value         0.9995   0.9964   0.9980   0.9967   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1905   0.1727   0.1611   0.1833
## Detection Prevalence   0.2866   0.1921   0.1761   0.1617   0.1834
## Balanced Accuracy      0.9975   0.9915   0.9932   0.9911   0.9985
```

# Prediction for the Testing Data

The Random Forest model is applied to predict the outcomes for the testing data.


```r
test_predict <- predict(rf_model, newdata = testing_cleaned)

# test_predict # results intentionally not shown
```
