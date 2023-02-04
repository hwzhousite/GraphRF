
library(RLT)
library(randomForestSRC)
library(gelnet)

load("./data/kernel_9004_500.Rdata")
load("./data/y.Rdata")

X <- kernel_90004
y <- mortality

n = nrow(X)
trainn = 15
testn = n - trainn

ntrees = 20
ncores = 1
nmin = 10
mtry = 10
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = FALSE

trainid = sample(1:n, trainn,replace = TRUE)
trainX = X[trainid,trainid]
trainY = y[trainid]

testX = X[-trainid,trainid]
testY = y[-trainid]

metric = data.frame(matrix(NA, 4, 4))
rownames(metric) = c("rlt", "rsf", "rf", "ranger")
colnames(metric) = c("fit.time", "pred.time", "pred.accuracy", "obj.size")

options(rf.cores = ncores)
start_time <- Sys.time()
RLTfit <- RLT(trainX, as.factor(trainY), ntrees = ntrees, ncores = ncores, nmin = nmin/2, mtry = mtry,
              split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, importance = importance)
metric[1, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
RLTPred <- predict.RLT(RLTfit, testX, ncores = ncores)
metric[1, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[1, 3] = mean((as.numeric(RLTPred$Prediction >= 0.5) == testY))
metric[1, 4] = object.size(RLTfit)

Ktrain <- graph.kernel(RLTfit,X1 = trainX)
Ktest <- graph.kernel(RLTfit, X1=trainX, X2 = testX)
options(rf.cores = ncores)
start_time <- Sys.time()
rsffit <- rfsrc(y ~ ., data = data.frame(trainX, "y"= trainY), ntree = ntrees, nodesize = nmin, mtry = mtry, 
                nsplit = nsplit, sampsize = trainn*sampleprob, importance = importance)
metric[2, 1] = difftime(Sys.time(), start_time, units = "secs")
start_time <- Sys.time()
rsfpred = predict(rsffit, data.frame(testX))
metric[2, 2] = difftime(Sys.time(), start_time, units = "secs")
metric[2, 3] = mean((as.numeric(rsfpred$predicted >= 0.5) == testY))
metric[2, 4] = object.size(rsffit)




#ker_cla <- gelnet.ker(trainX, as.factor(trainY),lambda = 10)


# predict on a subset of trees 

RLTPred_sub <- predict(RLTfit, testX, treeindex = c(1:10), keep.all = TRUE, ncores = ncores)
