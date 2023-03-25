//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Graph Classification
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../GraphClaForest.h"

using namespace Rcpp;
using namespace arma;

void Graph_Find_A_Split(Multi_Split_Class& OneSplit,
                          const RLT_CLA_DATA& CLA_DATA,
                          const PARAM_GLOBAL& Param,
                          const PARAM_RLT& RLTParam,
                          uvec& obs_id,
                          uvec& var_id,
                          vec& Splitid)
{
  size_t mtry = Param.mtry;
  size_t nmin = Param.nmin;
  double alpha = Param.alpha;
  int nsplit = Param.nsplit;
  int split_gen = Param.split_gen;
  int split_rule = Param.split_rule;
  
  size_t P = obs_id.n_elem;
  mtry = ( (mtry <= P) ? mtry:P ); // take minimum
  
  size_t k = 2;
  k = ( (k <= mtry)? k : mtry);
  
  //cout << " --- Reg_Find_A_Split with mtry = " << mtry << std::endl;
  // SVD Decomposition
  int method = 3;
  
  arma::mat A, L;
  
  if (method == 1) 
  {// submatrix col same as rows
     A = CLA_DATA.X(obs_id, obs_id);
     OneSplit.SplitVar = obs_id;
  }
  
  if (method == 2) // use all cols 
  {
      if (mtry == P)
      {
        A = CLA_DATA.X(obs_id, obs_id);
        OneSplit.SplitVar = obs_id;
        
      }else{
        
          uvec var_try = arma::randperm(P,mtry);
          //std::cout<< P << " / " << var_try << endl;
          A = CLA_DATA.X(obs_id, obs_id(var_try));  
          OneSplit.SplitVar = obs_id(var_try);
        
      }
      
      // Centering
      mat center = mean(A, 0);
      
       L = A - repmat(center, A.n_rows, 1) + pow(10,-6) * mat(A.n_rows, A.n_cols, fill::eye);
      
  }
  
  if (method == 3) // laplacian
  {  
    if (mtry == P)
    {
      
      A = CLA_DATA.X(obs_id, obs_id);
      OneSplit.SplitVar = obs_id;
      
    }else{
      
      uvec var_try = arma::randperm(P,mtry);
      //std::cout<< P << " / " << var_try << endl;
      A = CLA_DATA.X(obs_id, obs_id(var_try));  
      OneSplit.SplitVar = obs_id(var_try);
      
    }
    
     mat K = trans(A) * A;
     L = arma::diagmat(arma::sum(K, 1)) - K;

     
     //cout << A << endl;

  }
  
  if (method == 4) // laplacian
  {  
    if (mtry == P)
    {
      A = CLA_DATA.X(obs_id, obs_id);
      OneSplit.SplitVar = obs_id;
      
    }else{
      uvec var_try = arma::randperm(P,mtry);
      //std::cout<< P << " / " << var_try << endl;
      A = CLA_DATA.X(obs_id, obs_id(var_try));  
      OneSplit.SplitVar = obs_id(var_try);
      
    }
    mat K = trans(A) * A;
    mat D = arma::diagmat(pow(arma::sum(K, 1), 0.5));
    L = inv(D) * K * D;
    
    //cout << A << endl;
    
  }
  
  
  // SVD Decomposition
  arma::mat U; arma::mat V; arma::vec s;
  svd(U,s,V,L);
  
  // Tempmat contains the first k principle component
  // TempLoading contains the corresponding vector
  
  arma::uvec y = CLA_DATA.Y(obs_id);
  

  // select the best variable
  
  for(size_t j = 1; j < k; j++)
  {
    arma::vec TempLoad = V.unsafe_col(j);
    arma::uvec TempSplitVar;
    Multi_Split_Class TempSplit(TempLoad, TempSplitVar);
    TempSplit.value = 0;
    TempSplit.score = -1;
    arma::vec TempSplitid = A * (V.unsafe_col(j));
    cout << "loading : " << V.unsafe_col(j) << endl;
    
    Graph_Cla_Split(TempSplit, 
                        TempSplitid, 
                        y, 
                        0.0, // penalty
                        split_gen, 
                        split_rule, 
                        nsplit, 
                        nmin, 
                        alpha);
    
    if (TempSplit.score > OneSplit.score)
    {
      //cout << "TempSplitid : " << TempSplitid << endl;
      OneSplit.Loading = TempSplit.Loading;
      //cout << "Loading :" << OneSplit.Loading << endl;
      OneSplit.value = TempSplit.value;
      //cout << "Value : " << OneSplit.value << endl;
      OneSplit.score = TempSplit.score;
      //cout << "Score : " << OneSplit.score << endl;
      Splitid = TempSplitid;
    }
    
  }
   
   
   
}