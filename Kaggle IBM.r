#####Packages#####
setwd("C:/Users/Jack Wychor/Downloads/R Files")
require('dplyr')
require("reshape2")
require('tidyr')
require('ggplot2')
require('psych')
require('caret')
require('stringr')
require('randomForest')
require('e1071')
require('mlbench')
require('CHAID')
require('VGAM')
require('deepboost')
require('doParallel')
require('xgboost')
require('PRROC')
system("cmd.exe /C dir")
system("cmd.exe /C dir /AH")

###Data Cleaning and Analysis#####
k<-read.csv('Kaggle IBM.csv')
colnames(k)[1]<-'Age'

num<-unlist(lapply(k,is.numeric))

##1. Removing 0-Variance Predictors
#This code is shorthand for finding the near 0 vars
nz<-nearZeroVar(e,saveMetrics = T)

e<-k[,num]
e<-e[,colnames(e)!='StandardHours']
e<-e[,colnames(e)!='EmployeeCount']
k<-k[,colnames(k)!='StandardHours']
k<-k[,colnames(e)!='EmployeeCount']

str(e)

##2. Finding Correlated Predictors
pcor<-cor(e)
pcor
hpcor<-sum(abs(pcor[upper.tri(pcor)]>.99))
#No correlations are higher than .99

##3. Find Linear Combo Dependencies
combo<-findLinearCombos(e)
#There are none

##4. Recombining the Data
k1<-k[!k %in% e]
k2<-k1[,-c(5,9)]

kd<-data.frame(k2,e)

##5. Data Splitting
#Splitting based on y (splitting on x or time series is sometimes appropriate)
ti<-createDataPartition(ktr$Attrition,p=.75,
                        list=F,times=1)

trk<-ktr[ti,]
tek<-ktr[-ti,]

##6. Preprocessing
#Training Set
kp<-preProcess(trk,method=c('center','scale'))

trk<-predict(kp,newdata = trk)

#Test Set
kp<-preProcess(tek,method=c('center','scale'))

ktr<-predict(kp,newdata = tek)

###Creating and Training Models#####
##1.0 Train Controls
fitc<-trainControl(##10-fold CV
  method = 'repeatedcv',
  number = 10,
  repeats = 6)

#Utilize More Computer Cores
cl <- makeCluster(detectCores())
registerDoParallel(cl)

#Boosted tree model
set.seed(492)
gbfit1<-train(Attrition~.,data=trk,
              method='gbm',
              trControl=fitc,
              verbose=F,
              allowParallel=T)

gbfit1

ggplot(gbfit1)


tpred<-predict(gbfit1,tek)

#New Train Controls for ROC
fitc2<-trainControl(##10-fold CV
  method = 'repeatedcv',
  number = 10,
  repeats = 6,
  classProbs = T,
  summaryFunction = twoClassSummary,
  allowParallel=T
  )

#Boosted model for specificity
set.seed(492)
gbfit3<-train(Attrition~.,data=trk,
              method='gbm',
              trControl=fitc2,
              verbose=F)

gbfit3

##Compare Model Types
#GB model (using ROC)
set.seed(492)
gbfit2<-train(Attrition~.,data=trk,
              method='gbm',
              trControl=fitc2,
              verbose=F,
              metric='ROC')
gbfit2

#SVMR Model
set.seed(492)
svfit<-train(Attrition~.,data=trk,
             method='svmRadial',
             trControl=fitc2,
             tunelength=8,
             metric='ROC',
             seeds=192)

svfit

#SVML Model
set.seed(492)
svlfit<-train(Attrition~.,data=trk,
             method='svmLinear',
             trControl=fitc2,
             tunelength=8,
             metric='ROC',
             seeds=192)
svlfit

#Random Forest
set.seed(492)
rdfit<-train(Attrition~.,data=trk,
             method='rf',
             trControl=fitc2,
             tunelength=4,
             metric='ROC',
             seeds=192)

rdfit

#GBM
set.seed(492)
gbmfit<-train(Attrition~.,data=trk,
             method='gbm',
             trControl=fitc2,
             metric='ROC')
gbmfit

#C5.0
grid <- expand.grid( .winnow = c(TRUE,FALSE),
                     .trials=c(1,5,10,15,20,25,30,35,40,45,50), .model="tree" )
set.seed(492)
c5fit<-train(Attrition~.,data=trk,
                    method='C5.0',
                    trControl=fitc2,
                    tuneGrid=grid,
                    metric='ROC',
             seeds=192)

c5fit

#vglmCum
set.seed(492)
vglmfit<-train(Attrition~.,data=trk,
                      method='vglmCumulative',
                      trControl=fitc2,
                      metric='ROC',
               seeds=192)

#xgbTree
set.seed(492)
xgbfit<-train(Attrition~.,data=trk,
                      method='xgbTree',
                      trControl=fitc2,
                      metric='ROC',
                      seeds=192)


#Genralized Partial Least Squares
set.seed(492)
require(gpls)
gplsfit<-train(Attrition~.,data=trk,
               method='gpls',
               trControl=fitc2,
               metric='ROC',
               seeds=192)

#GLMnet
set.seed(492)
require('glmnet')
glmnetfit<-train(Attrition~.,data=trk,
               method='glmnet',
               trControl=fitc2,
               metric='ROC',
               seeds=192)

#Naive Bayes
set.seed(492)
nbayesfit<-train(Attrition~.,data=trk,
                 method='nb',
                 trControl=fitc2,
                 metric='ROC',
                 seeds=192)

#Compare across Models
resamps<-resamples(list(
                        SVMR = svfit,
                        SVML = svlfit,
                        RF = rdfit,
                        C5 = c5fit,
                        VGLM = vglmfit,
                        GPLS = gplsfit,
                        GLMNET = glmnetfit,
                        NBAYES = nbayesfit,
                        GBM = gbmfit,
                        XGBT = xgbfit))

diffs<-diff(resamps)
summary(diffs)

#Based on this plot, several models did well.
#We will choose SVMR because it performed well
#in terms of detecting 'yes' cases of attrition.
bwplot(resamps)


###Fine Tuning an XGBTree####
##Create a Tuning Grid for the xbgTreemodel
#First Iteration (max depth, eta)
modelLookup('xgbTree')

#Tuning grid
xgb_grid <- expand.grid(
  nrounds = seq(200,1000,50),
  max_depth = c(2,3,4,5,6),
  eta = c(.025,.05,.1,0.3,.5),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

set.seed(492)
system.time(xgbfit2<-train(Attrition~.,data=trk,
              method='xgbTree',
              trControl=fitc2,
              tuneGrid=xgb_grid,
              metric='ROC',
              seeds=192))

xgbfit2

#Visualize the Tuning Process (See https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret)
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$ROC, probs = probs), min(x$results$ROC))) +
    theme_bw()
}
tuneplot(xgbfit2)

#Second Iteration (max depth, min_child_weight,eta)
xgb_grid2 <- expand.grid(
  nrounds = seq(200,1000,50),
  max_depth = c(2,3),
  eta = c(.01,.02,.025,.03),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1,2,3),
  subsample = 1
)

system.time(xgbfit3<-train(Attrition~.,data=trk,
                           method='xgbTree',
                           trControl=fitc2,
                           tuneGrid=xgb_grid2,
                           metric='ROC',
                           seeds=192))
tuneplot(xgbfit3)

#Third iteration
xgb_grid3 <- expand.grid(
  nrounds = seq(200,1000,50),
  max_depth = c(2,3),
  eta = c(.025,.03,.05,.1),
  gamma = 0,
  colsample_bytree = c(.4,.6,.8,1),
  min_child_weight = 2,
  subsample = c(.5,.75,1)
)

#Note. System time for this iteration on my machine was 5539.24(!)
system.time(xgbfit4<-train(Attrition~.,data=trk,
                           method='xgbTree',
                           trControl=fitc2,
                           tuneGrid=xgb_grid3,
                           metric='ROC',
                           seeds=192))

#tuneplot(xgbfit4) too many parameters were changed to plot

xgbfit4

#Fourth iteration (gamma)
xgb_grid4 <- expand.grid(
  nrounds = seq(200,1000,50),
  max_depth = 2,
  eta = .03,
  gamma = c(0,.05,.1,.5,.7,.9,1),
  colsample_bytree = .4,
  min_child_weight = 2,
  subsample = .5
)

set.seed(678)
system.time(xgbfit5<-train(Attrition~.,data=trk,
                           method='xgbTree',
                           trControl=fitc2,
                           tuneGrid=xgb_grid4,
                           metric='ROC',
                           seeds=856))

tuneplot(xgbfit5)
xgbfit5

#Fifth iteration (eta, nrounds)
xgb_grid5 <- expand.grid(
  nrounds = c(600,650,700),
  max_depth = 2,
  eta = c(.01,.015,.025,.05,.1),
  gamma = 1,
  colsample_bytree = .4,
  min_child_weight = 2,
  subsample = .5
)

set.seed(678)
system.time(xgbfit6<-train(Attrition~.,data=trk,
                           method='xgbTree',
                           trControl=fitc2,
                           tuneGrid=xgb_grid5,
                           metric='ROC',
                           seeds=856))

tuneplot(xgbfit6)
xgbfit6

#Final Model Parameters
xgb_final <- expand.grid(
  nrounds = 700,
  max_depth = 2,
  eta = .03,
  gamma = 1,
  colsample_bytree = .4,
  min_child_weight = 2,
  subsample = .5
)

system.time(xgbfitfinal<-train(Attrition~.,data=trk,
                           method='xgbTree',
                           trControl=fitc2,
                           tuneGrid=xgb_final,
                           metric='ROC',
                           seeds=856))
xgbfitfinal

#Inspect Variable Importance for Top Methods
imp<-varImp(xgbfitfinal,scale=F)
plot(imp)

#Comparing the Final Model to Test Data
spred<-predict(xgbfitfinal,tek)
confusionMatrix(spred,ref=tek$Attrition)


###The Final XGBTree Model Cross-Validation Results#####
#Confusion Matrix and Statistics

#Reference
#Prediction  No Yes
#       No  303  43
#      Yes   5  16

#Accuracy : 0.8692         
#95% CI : (0.8304, 0.902)
#No Information Rate : 0.8392         
#P-Value [Acc > NIR] : 0.06498        

#Kappa : 0.3447         

#Mcnemar's Test P-Value : 9.27e-08       
                                         
#            Sensitivity : 0.9838         
#            Specificity : 0.2712         
#         Pos Pred Value : 0.8757         
#         Neg Pred Value : 0.7619         
#             Prevalence : 0.8392         
#         Detection Rate : 0.8256         
#   Detection Prevalence : 0.9428         
#      Balanced Accuracy : 0.6275         
                                         
#       'Positive' Class : No

#The final XGBTree model produced is highly sensitive,
#while having only modest specificity. When compared to
#This may be desirable if a company wanted a test that
#accurately told them which employees were not at risk of
#attrition while still having a ~25% chance of detecting employees
#who are. For a model significantly higher specificity, the VGLM and
#SVMR can be used (See line #246 for reference) with only
#minor reductions in ROC and Sensisitivity.

##VGLM Training
set.seed(132)
vglmfit2<-train(Attrition~.,data=trk,
               method='vglmCumulative',
               trControl=fitc2,
               metric='ROC',
               tuneLength=10,
               seeds=791)
vglmfit2

spred2<-predict(svfit2,tek)
confusionMatrix(spred2,ref=tek$Attrition)

