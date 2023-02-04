//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file

# include <RcppArmadillo.h>
# include <Rcpp.h>
# include <algorithm>

using namespace Rcpp;
using namespace arma;

# include "Definition.h"

// ********************//
// functions for trees //
// ********************//

#ifndef RLT_ARRANGE
#define RLT_ARRANGE

void Uni_Find_Terminal_Node(size_t Node, 
              							const Uni_Tree_Class& OneTree,
              							const mat& X,
              							const uvec& Ncat,
              							uvec& proxy_id,
              							const uvec& real_id,
              							uvec& TermNode);

void Uni_Find_Terminal_Node_ShuffleJ(size_t Node, 
                                     const Uni_Tree_Class& OneTree,
                                     const mat& X,
                                     const uvec& Ncat,
                                     uvec& proxy_id,
                                     const uvec& real_id,
                                     uvec& TermNode,
                                     const vec& tildex,
                                     const size_t j);

void Multi_Find_Terminal_Node(size_t Node, 
                              const Multi_Tree_Class& OneTree,
                              const mat& X,
                              const uvec& Ncat,
                              uvec& proxy_id,
                              const uvec& real_id,
                              uvec& TermNode);

List ForestKernelUni(arma::field<arma::uvec>& NodeType,
                     arma::field<arma::uvec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::field<arma::field<arma::uvec>>& NodeRegi,
                     arma::mat& X,
                     arma::uvec& Ncat,
                     arma::vec& obsweight,
                     int usecores,
                     int verbose);

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
                      size_t verbose);

List GraphKernel_Cross(arma::field<arma::uvec>& NodeType,
                       arma::field<arma::uvec>& SplitVar,
                       arma::field<arma::uvec>& SplitLoadingList,
                       arma::field<arma::vec>& SplitValue,
                       arma::field<arma::uvec>& LeftNode,
                       arma::field<arma::uvec>& RightNode,
                       arma::field<arma::vec>& NodeSize,
                       arma::field<arma::vec>& NodeAve,
                       arma::mat& XTest,
                       arma::mat& XTrain,
                       arma::uvec& Ncat,
                       arma::umat& ObsTrack,
                       int usecores,
                       int verbose);

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
                  size_t verbose);

// ************************//
// miscellaneous functions //
// ************************//

// catigorical variables pack
double pack(const size_t nBits, const uvec& bits);
void unpack(const double pack, const size_t nBits, uvec& bits);
bool unpack_goright(double pack, const size_t cat);

// sample both inbag and oobag samples
void oob_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const size_t size,
                 const bool replacement);

void set_obstrack(arma::umat& ObsTrack,
                  const size_t nt,
                  const size_t size,
                  const bool replacement);

void get_samples(arma::uvec& inbagObs,
                 arma::uvec& oobagObs,
                 const arma::uvec& subj_id,
                 const arma::uvec& ObsTrack_nt);


void move_cont_index(size_t& lowindex, size_t& highindex, const vec& x, const uvec& indices, size_t nmin);
void split_id(const vec& x, double value, uvec& left_id, uvec& obs_id);
void split_id_cat(const vec& x, double value, uvec& left_id, uvec& obs_id, size_t ncat);
void split_id_multi(const vec& Splitid, const Multi_Split_Class& OneSplit, uvec& left_id, uvec& obs_id);


bool cat_reduced_compare(Cat_Class& a, Cat_Class& b);
bool cat_reduced_collapse(Cat_Class& a, Cat_Class& b); 

// bool cat_reduced_compare_score(Cat_Class& a, Cat_Class& b);

void move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Surv_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin);
void move_cat_index(size_t& lowindex, size_t& highindex, std::vector<Reg_Cat_Class>& cat_reduced, size_t true_cat, size_t nmin);

double record_cat_split(std::vector<Surv_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);

double record_cat_split(std::vector<Reg_Cat_Class>& cat_reduced,
                        size_t best_cat, 
                        size_t true_cat,
                        size_t ncat);

double record_cat_split(size_t cat, 
                        std::vector<Surv_Cat_Class>& cat_reduced);
                        
double record_cat_split(arma::uvec& goright_temp, 
                        std::vector<Surv_Cat_Class>& cat_reduced);                        

void goright_roller(arma::uvec& goright_cat);

// other 

double cindex_d(arma::vec& Y,
              arma::uvec& Censor,
              arma::vec& pred);

double cindex_i(arma::uvec& Y,
              arma::uvec& Censor,
              arma::vec& pred);

#endif












