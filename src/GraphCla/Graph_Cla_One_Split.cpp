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

void Graph_Cla_Split(Multi_Split_Class& TempSplit,
                    const vec& x, // x and Y are same length as obs_id 
                    const uvec& Y,
                    double penalty,
                    int split_gen,
                    int split_rule,
                    int nsplit,
                    size_t nmin,
                    double alpha)
{
  size_t N = x.n_elem;
  
  double temp_score;
  
  uvec indices = linspace<uvec>(0, N-1,N);
  indices = indices(sort_index(x)); // this is the sorted obs_id
  
  // check identical 
  if ( x(indices(0)) == x(indices(N-1)) ) return;
  
  // best split, check all splitting point 
  uvec y;
    
  //if(Y.empty())
    
    y = Y(indices);
    
 
  //uvec y = Y(indices);
  
  for(size_t k = 0; k < N-1; k++){

    // get the cut-off point based on the variance
    temp_score = graph_cla_score_gini(indices, y, k);
    
    if (temp_score > TempSplit.score){
      
        TempSplit.value =(x(indices(k)) + x(indices(k+1)))/2;
        TempSplit.score = temp_score;
        
    }
  }
  return;
}


double graph_cla_score_gini(uvec& indices,
                            const uvec& Y,
                            size_t& k)
{
      //DEBUG_Rcout <<" --- Supervised with Gini score --- "<< std::endl;
      
      size_t N = indices.n_elem;
      double left = 0; double right =0;
      
      for(size_t i = 0; i <= k; i++){
        if(Y(i) == Y(0)) left++;
      }
      for(size_t i = k+1; i <= N-1; i++){
        if(Y(i) == Y(0)) right++;
      }
      
      double leftmean = left/(k+1); //arma::mean(Y( linspace<uvec>(0,k,k+1) ));
      double rightmean = right/(N-k-1); //arma::mean(Y( linspace<uvec>(k+1, N-1, N-k-1) ));  
      
      double gini = ((k+1)*leftmean*(1-leftmean) + (N-k-1)*rightmean*(1-rightmean))/N;
    
      return (1 - gini); // larger the better 
  
}

double graph_multicla_score_gini(uvec& indices,
                            const uvec& Y,
                            size_t& k)
{
  //DEBUG_Rcout <<" --- Supervised with Gini score --- "<< std::endl;
  
  size_t N = indices.n_elem;
  uvec y_unique = find_unique(Y);
  size_t q = y_unique.n_elem;
 
  double leftmean = 0;
  double rightmean = 0;
  
  for(size_t j = 0; j < q; j++){
    
      size_t count = 0;
  
      for(size_t i = 0; i <= k; i++){
     
          if(Y(i) == Y(y_unique(j))) count++;
       
      }
      
      leftmean = leftmean + (count/(k+1)) * (count/(k+1)) ;
     
  }
  
  for(size_t j = 0; j < q; j++){
    
    size_t count = 0;
    
    for(size_t i = k+1; i <= N-1; i++){
      
          if(Y(i) == Y(y_unique(j))) count++;
      
    }
    
    rightmean = rightmean + (count/(k+1)) * (count/(k+1)) ;
    
  }
  
 
  double gini = ((k+1)*leftmean + (N-k-1)*rightmean)/N;
  
  
  return (1 - gini); // larger the better 
}

double cla_unsuper_score_var(uvec& indices,
                             const vec& x,
                             size_t& k)
{
  //DEBUG_Rcout << " --- UnSupervised with Variance score --- "<< std::endl;
  
  double score = 0;
  double left = 0; double right =0;
  
  size_t N = indices.size();
  
  
  
  
  return 0;
}

