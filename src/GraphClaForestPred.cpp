//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Graph Classification
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Utility.h"
# include "GraphClaForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List GraphClaForestMultiPred(arma::field<arma::uvec>& NodeType,
                             arma::field<arma::field<arma::uvec>> SplitVar,
                             arma::field<arma::field<arma::vec>> SplitLoading,
                             arma::field<arma::vec>& SplitValue,
                             arma::field<arma::uvec>& LeftNode,
                             arma::field<arma::uvec>& RightNode,
                             arma::field<arma::vec>& NodeSize,
                             arma::field<arma::vec>& NodeAve,
                             arma::mat& X,
                             arma::uvec& Ncat,
                             arma::uvec& treeindex,
                             bool keep_all,
                             int usecores,
                             int verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);
  
  // convert R object to forest
  
  Cla_Multi_Forest_Class CLA_FOREST(NodeType, SplitVar, SplitLoading, SplitValue, LeftNode, RightNode, NodeSize, NodeAve);
  
  mat PredAll;
  
  Cla_Multi_Forest_Pred(PredAll,
                      (const Cla_Multi_Forest_Class&) CLA_FOREST,
                      X,
                      Ncat,
                      treeindex,
                      usecores,
                      verbose);
  
  List ReturnList;
  //cout << PredAll << endl;
  ReturnList["Prediction"] = PredAll;
  
  if (keep_all)
    ReturnList["PredictionAll"] = PredAll;
  
  return ReturnList;
}
