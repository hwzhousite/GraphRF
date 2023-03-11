library(RSpectra)
library(irlba)
library(data.tree)
library(data.table)


Binary.Similarity <- function(s1,s2){
  n <- min(length(s1),length(s2))
  min(which(s1[1:n]!=s2[1:n]))
}


gen.A.from.P <- function(P,undirected=TRUE){
  n <- nrow(P)
  if(undirected){
    upper.tri.index <- which(upper.tri(P))
    tmp.rand <- runif(n=length(upper.tri.index))
    #A <- matrix(0,n,n)
    A <- rsparsematrix(n,n,0)
    A[upper.tri.index[tmp.rand<P[upper.tri.index]]] <- 1
    A <- A+t(A)
    diag(A) <- 0
    return(A) }else{
      A <- matrix(0,n,n)
      r.seq <- runif(n=length(P))
      A[r.seq < as.numeric(P)] <- 1
      diag(A) <- 0
      return(A)
    }
}




## n: dimension of the network
## d: number of layers until leaves (excluding the root)
## a.seq: sequence a_r
## lambda: average node degree, only used when alpha is not provided.
## alpha: the common scaling of the a_r sequence, so at the end, essentially the a_r sequence is a.seq*alpha
## N: number of networks one wants to generate from the same model
BTSBM <- function(n,d,a.seq,lambda,alpha=NULL,N=1){
  K <- 2^d
  #outin <- beta
  #beta <- (2*K*outin/(K-1))^((1/((d-1))))
  ## generate binary strings
  b.list <- list()
  for(k in 1:K){
    b.list[[k]] <- as.character(intToBits(k-1))[d:1]
  }
  ## construct B
  comm.sim.mat <- B <- matrix(0,K,K)
  for(i in 1:(K-1)){
    for(j in (i+1):K){
      s <- Binary.Similarity(b.list[[i]],b.list[[j]])-1
      comm.sim.mat[i,j] <- s+1
    }
  }
  comm.sim.mat <- comm.sim.mat+t(comm.sim.mat)
  diag(comm.sim.mat) <- d+1
  B[1:(K^2)] <- a.seq[d+2-as.numeric(comm.sim.mat)]
  w <- floor(n/K)
  g <- c(rep(seq(1,K),each=w),rep(K,n-w*K))
  Z <- matrix(0,n,K)
  Z[cbind(1:n,g)] <- 1
  P <- Z%*%B%*%t(Z)
  if(is.null(alpha)){
    P <- P*lambda/mean(colSums(P))
  }else{
    P <- P*alpha
  }
  node.sim.mat <- Z%*%comm.sim.mat%*%t(Z)
  diag(node.sim.mat) <- 0
  #print(paste("Within community expected edges: ",P[1,1]*w,sep=""))
  A.list <- list()
  for(I in 1:N){
    A.list[[I]] <- gen.A.from.P(P,undirected=TRUE)
  }
  return(list(A.list=A.list,B=B,label=g,P=P,comm.sim.mat=comm.sim.mat,node.sim.mat=node.sim.mat))
}
