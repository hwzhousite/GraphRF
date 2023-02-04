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
List GraphClaForestMultiFit(arma::mat& X,
              					   arma::uvec& Y,
              					   arma::uvec& Ncat,
              					   List& param,
              					   List& RLTparam,
              					   arma::vec& obsweight,
              					   arma::vec& varweight,
              					   int usecores,
              					   int verbose,
              					   arma::umat& ObsTrack)
{
  //DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF Classification Multi ///" << std::endl;
  //std::cout << Y <<endl;
  
   std::cout << "Start cpp" << endl;
 //std::cout << X <<endl;
  // readin parameters
  PARAM_GLOBAL Param(param);
  if(verbose) Param.print();
  PARAM_RLT Param_RLT(RLTparam);
  if (verbose and Param.reinforcement) Param_RLT.print();
  
  // create data objects  
  RLT_CLA_DATA CLA_DATA(X, Y, Ncat, obsweight, varweight);
  
  size_t N = CLA_DATA.X.n_rows;
  size_t P = CLA_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  size_t seed = Param.seed;
  
  int obs_track = Param.obs_track;
  int importance = Param.importance;
  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  // prediction
  vec Prediction;
  vec OOBPrediction;
  
  // initiate forest
  arma::field<arma::uvec> NodeType(ntrees);
  arma::field<arma::field<arma::uvec>> SplitVar(ntrees);
  arma::field<arma::field<arma::vec>> SplitLoading(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeSize(ntrees);
  arma::field<arma::vec> NodeAve(ntrees);
  
  Cla_Multi_Forest_Class CLA_FOREST(NodeType, SplitVar, SplitLoading, SplitValue, LeftNode, RightNode, NodeSize, NodeAve);
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  
  Graph_Cla_Forest_Build((const RLT_CLA_DATA&) CLA_DATA,
                         CLA_FOREST,
                         (const PARAM_GLOBAL&) Param,
                         (const PARAM_RLT&) Param_RLT,
                         obs_id,
                         var_id,
                         ObsTrack,
                         Prediction,
                         OOBPrediction,
                         VarImp,
                         seed,
                         usecores,
                         verbose);
  
  List ReturnList;
  List Forest_R;
  
  Forest_R["NodeType"] = NodeType;
  Forest_R["SplitVar"] = SplitVar;
  Forest_R["SplitLoading"] = SplitLoading;
  Forest_R["SplitValue"] = SplitValue;
  Forest_R["LeftNode"] = LeftNode;
  Forest_R["RightNode"] = RightNode;
  Forest_R["NodeSize"] = NodeSize;    
  Forest_R["NodeAve"] = NodeAve;
  
  ReturnList["FittedForest"] = Forest_R;

  return ReturnList;
  
}