library(bipartite)
library(RLT)
library(MASS)
# SBM
n <- 50
P1 <- matrix(NA, nrow = n, ncol = n)
A1 <- matrix(NA, nrow = n, ncol = n)
u <- runif(n)


for (i in 1:n) {
  
    a <- u[i]
    
  for (j in 1:n) {
    
    b <- u[j]
    
    P1[i,j] <- 1/3 * (a^2 + b^2) * cos(1/(a^2 + b^2)) + 0.15
    A1[i,j] <- rbinom(1,1,P1[i,j])
  }
  
}

Y <- rep(c(0,1), each = n/2)
X <- A
y <- as.factor(Y)
trainn = 0.8 * n
testn = n - trainn

ntrees = 100
ncores = 1
nmin = 5
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

## RLT 

RLTfit <- RLT(trainX, trainY, ntrees = 1, ncores = ncores, nmin = nmin, mtry = mtry, split.gen = rule, nsplit = nsplit, resample.prob = sampleprob, importance = importance)


Ktrain <- graph.kernel(RLTfit,trainX)
  
Ktest <- graph.kernel(RLTfit, testX)
  
RLTKtrain <- Ktrain$Kernel/ntrees
RLTKtest <- Ktest$Kernel/ntrees 
  
 
  
sum((RLTKtrain - P[trainid,trainid])^2)/(length(trainid))^2
sum((RLTKtest - P[-trainid,-trainid])^2)/(n-length(trainid))^2

## LE

k <- 4
le <- svd(trainX) 
p11 <- le$u[,1:k] %*% diag(le$d[1:k]) %*% t(le$v[,1:k])
p22 <- (testX) %*% ginv(p11) %*% t(testX)
sum((p22 - P[-trainid,-trainid])^2)/(n-length(trainid))^2

