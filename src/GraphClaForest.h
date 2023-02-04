//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"
# include "Trees//Trees.h"
# include "Utility//Utility.h"

using namespace Rcpp;
using namespace arma;

#ifndef ClaForest_Fun
#define ClaForest_Fun

// univariate tree split functions 

List ClaForestMultiFit(arma::mat& X,
                       arma::uvec& Y,
                       arma::uvec& Ncat,
                       List& param,
                       List& RLTparam,
                       arma::vec& obsweight,
                       arma::vec& varweight,
                       int usecores,
                       int verbose,
                       arma::umat& ObsTrack);

void Graph_Cla_Forest_Build(const RLT_CLA_DATA& CLA_DATA,
                            Cla_Multi_Forest_Class& CLA_FOREST,
                            const PARAM_GLOBAL& Param,
                            const PARAM_RLT& Param_RLT,
                            uvec& obs_id,
                            uvec& var_id,
                            umat& ObsTrack,
                            vec& Prediction,
                            vec& OOBPrediction,
                            vec& VarImp,
                            size_t seed, // this is not done yet
                            int usecores,
                            int verbose);

void Graph_Cla_Split_A_Node(size_t Node,
                            Cla_Multi_Tree_Class& OneTree,
                            const RLT_CLA_DATA& CLA_DATA,
                            const PARAM_GLOBAL& Param,
                            const PARAM_RLT& Param_RLT,
                            uvec& obs_id,
                            uvec& var_id);

void Graph_Cla_Terminate_Node(size_t Node, 
                              Cla_Multi_Tree_Class& OneTree,
                              uvec& obs_id,                            
                              const uvec& Y,
                              const vec& obs_weight,                            
                              const PARAM_GLOBAL& Param,
                              bool useobsweight);


void Graph_Find_A_Split(Multi_Split_Class& OneSplit,
                        const RLT_CLA_DATA& CLA_DATA,
                        const PARAM_GLOBAL& Param,
                        const PARAM_RLT& RLTParam,
                        uvec& obs_id,
                        uvec& var_id,
                        vec& Splitid);

void Graph_Cla_Split(Multi_Split_Class& TempSplit,
                     const vec& x, // x and Y are same length as obs_id 
                     const uvec& Y,
                     double penalty,
                     int split_gen,
                     int split_rule,
                     int nsplit,
                     size_t nmin,
                     double alpha);

double graph_cla_score_gini(uvec& indices,
                            const uvec& Y,
                            size_t& k);

double cla_unsuper_score_var(uvec& indices,
                             const vec& x,
                             size_t& k);

void Cla_Multi_Forest_Pred(mat& Pred,
                           const Cla_Multi_Forest_Class& CLA_FOREST,
                           const mat& X,
                           const uvec& Ncat,
                           const uvec& treeindex,
                           int usecores,
                           int verbose);
#endif
