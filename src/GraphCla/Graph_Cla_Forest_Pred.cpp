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

void Cla_Multi_Forest_Pred(mat& Pred,
                         const Cla_Multi_Forest_Class& CLA_FOREST,
                  			 const mat& X,
                  			 const uvec& Ncat,
                  			 const uvec& treeindex,
                  			 int usecores,
                  			 int verbose)
{
  
  size_t N = X.n_rows;
  size_t ntrees = CLA_FOREST.NodeTypeList.size();
  
  Pred.zeros(N, treeindex.n_elem);

  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < treeindex.n_elem; nt++)
    {
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
        
      Cla_Multi_Tree_Class OneTree(CLA_FOREST.NodeTypeList(nt), 
                                   CLA_FOREST.SplitVarList(nt),
                                   CLA_FOREST.SplitLoadingList(nt),
                                   CLA_FOREST.SplitValueList(nt),
                                   CLA_FOREST.LeftNodeList(nt),
                                   CLA_FOREST.RightNodeList(nt),
                                   CLA_FOREST.NodeSizeList(nt),
                                   CLA_FOREST.NodeAveList(nt));
      
      Multi_Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      //cout << OneTree.NodeAve(TermNode) << endl;
      
      Pred.unsafe_col(nt).rows(real_id) = OneTree.NodeAve(TermNode);
    }
  }
}

