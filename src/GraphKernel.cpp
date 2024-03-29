//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Kernel
//  **********************************

// my header file
# include "RLT.h"
# include "Trees/Trees.h"
# include "Utility/Utility.h"
# include "GraphClaForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List GraphKernel_Self(arma::field<arma::uvec>& NodeType,
                      arma::field<arma::field<arma::uvec>> SplitVar,
                      arma::field<arma::field<arma::vec>> SplitLoading,
            					 arma::field<arma::vec>& SplitValue,
            					 arma::field<arma::uvec>& LeftNode,
            					 arma::field<arma::uvec>& RightNode,
            					 arma::field<arma::vec>& NodeSize,
            					 arma::field<arma::vec>& NodeAve,
            					 arma::mat& X,
            					 arma::uvec& Ncat,
            					 size_t verbose)
{

  size_t N = X.n_rows;
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  umat K(N, N, fill::zeros);
  uvec real_id = linspace<uvec>(0, N-1, N);  
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Cla_Multi_Tree_Class OneTree(NodeType(nt), 
                                 SplitVar(nt),
                                 SplitLoading(nt),
                                 SplitValue(nt),
                                 LeftNode(nt),
                                 RightNode(nt),
                                 NodeSize(nt),
                                 NodeAve(nt));
    
    // initiate all observations
    uvec proxy_id = linspace<uvec>(0, N-1, N);
    uvec TermNode(N, fill::zeros);
    
    // get terminal node id
    Multi_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
    
    //record
    uvec UniqueNode = unique(TermNode);
    
    for (auto j : UniqueNode)
    {
      uvec ID = real_id(find(TermNode == j));
      
      K.submat(ID, ID) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}





// [[Rcpp::export()]]
List GraphKernel_Cross(arma::field<arma::uvec>& NodeType,
                  arma::field<arma::field<arma::uvec>> SplitVar,
                  arma::field<arma::field<arma::vec>> SplitLoading,
                  arma::field<arma::vec>& SplitValue,
                  arma::field<arma::uvec>& LeftNode,
                  arma::field<arma::uvec>& RightNode,
                  arma::field<arma::vec>& NodeSize,
                  arma::field<arma::vec>& NodeAve,
                  arma::mat& X1,
                  arma::mat& X2,
                  arma::uvec& Ncat,
                  size_t verbose)
{
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Cla_Multi_Tree_Class OneTree(NodeType(nt), 
                                 SplitVar(nt),
                                 SplitLoading(nt),
                                 SplitValue(nt),
                                 LeftNode(nt),
                                 RightNode(nt),
                                 NodeSize(nt),
                                 NodeAve(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Multi_Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Multi_Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));
    
    for (auto j : UniqueNode)
    {
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j));
      
      K.submat(ID1, ID2) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}

// [[Rcpp::export()]]
List GraphKernel_Train(arma::field<arma::uvec>& NodeType,
                  arma::field<arma::field<arma::uvec>> SplitVar,
                  arma::field<arma::field<arma::vec>> SplitLoading,
                  arma::field<arma::vec>& SplitValue,
                  arma::field<arma::uvec>& LeftNode,
                  arma::field<arma::uvec>& RightNode,
                  arma::field<arma::vec>& NodeSize,
                  arma::field<arma::vec>& NodeAve,
                  arma::mat& X1,
                  arma::mat& X2,
                  arma::uvec& Ncat,
                  arma::imat& ObsTrack,
                  size_t verbose)
{
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Cla_Multi_Tree_Class OneTree(NodeType(nt), 
                                 SplitVar(nt),
                                 SplitLoading(nt),
                                 SplitValue(nt),
                                 LeftNode(nt),
                                 RightNode(nt),
                                 NodeSize(nt),
                                 NodeAve(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Multi_Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Multi_Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));
    ivec intreent = ObsTrack.col(nt);
    
    for (auto j : UniqueNode)
    {
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j && intreent > 0));
      
      K.submat(ID1, ID2) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}
