//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Graph Classification
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../GraphClaForest.h"

#include <xoshiro.h>
#include <dqrng_distribution.h>

using namespace Rcpp;
using namespace arma;

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
                            int verbose)
{
   // parameters need to be used
   size_t ntrees = Param.ntrees;
   bool replacement = Param.replacement;
   double resample_prob = Param.resample_prob;
   size_t P = Param.P;
   size_t N = obs_id.n_elem;
   size_t size = (size_t) obs_id.n_elem*resample_prob;
   size_t nmin = Param.nmin;
   
   #pragma omp parallel num_threads(usecores)
   {
     //dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng 
     //lrng.long_jump(omp_get_thread_num() + 1);  // advance rng by 1 ... ncores jumps
      #pragma omp for schedule(static)
      for(size_t nt=0; nt < ntrees; nt++)
      {
        uvec inbagObs;
        
        inbagObs = obs_id;
        
        std::cout << " ---- NEW Tree ---" << std::endl;
        
        Cla_Multi_Tree_Class OneTree(CLA_FOREST.NodeTypeList(nt), 
                                     CLA_FOREST.SplitVarList(nt),
                                     CLA_FOREST.SplitLoadingList(nt),
                                     CLA_FOREST.SplitValueList(nt),
                                     CLA_FOREST.LeftNodeList(nt),
                                     CLA_FOREST.RightNodeList(nt),
                                     CLA_FOREST.NodeSizeList(nt),
                                     CLA_FOREST.NodeAveList(nt));
        
        size_t TreeLength = 1 + size/nmin*3;
        
        OneTree.initiate(TreeLength);
        
        //cout << nt <<" th forest build:" << obs_id.n_elem << endl;
        // start to fit a tree
        // 0: unused, 1: reserved; 2: internal node; 3: terminal node
        OneTree.NodeType(0) = 1;
        
        Graph_Cla_Split_A_Node(0, OneTree, CLA_DATA, Param, Param_RLT, 
                               inbagObs, var_id);
        //cout << OneTree.SplitLoading << endl;
        
        //cout << "trim : " << endl;
        
        // trim tree 
        TreeLength = OneTree.get_tree_length();
        OneTree.trim(TreeLength);  
        
        //cout << OneTree.SplitLoading << endl;
      }
    }
  
}