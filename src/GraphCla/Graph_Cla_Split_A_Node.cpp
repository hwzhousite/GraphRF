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

void Graph_Cla_Split_A_Node(size_t Node,
                            Cla_Multi_Tree_Class& OneTree,
                            const RLT_CLA_DATA& CLA_DATA,
                            const PARAM_GLOBAL& Param,
                            const PARAM_RLT& Param_RLT,
                            uvec& obs_id,
                            uvec& var_id)
{
  size_t N = obs_id.n_elem;
  size_t P = Param.P;
  size_t nmin = Param.nmin;
  bool useobsweight = Param.useobsweight;

  //cout << Node << "th Node" <<" ObsNumber: " << N << endl;
  
  if (N < 2*nmin){
TERMINATENODE:
    
    //cout << "  -- Terminate node --" << Node << std::endl;
    cout << " End Node Size : " << N << std::endl;
    Graph_Cla_Terminate_Node(Node, OneTree, obs_id, CLA_DATA.Y, CLA_DATA.obsweight, Param, useobsweight);
    
  }else{

    //cout << "  -- Do split --" << std::endl;
    arma::vec Loading;// record V
    arma::uvec SplitVar; // record split variable
    arma::vec Splitid; // record U
    Multi_Split_Class OneSplit(Loading, SplitVar);
    
    Graph_Find_A_Split(OneSplit, CLA_DATA, Param, Param_RLT, obs_id, var_id, Splitid);
    //DEBUG_Rcout << "-- Found split on variable --" << OneSplit.Loading << " cut " << OneSplit.value << "Var" << obs_id << std::endl;
    
    
    // store proportion
    OneTree.NodeAve(Node) = arma::mean(CLA_DATA.Y(obs_id));
    
    // if did not find a good split, terminate
    if (OneSplit.score <= 0){
      cout << "No good Split" << endl;
      goto TERMINATENODE;
    }
    // construct indices for left and right nodes
    uvec left_id( obs_id.n_elem );
    //cout << CLA_DATA.X(obs_id, OneSplit.SplitVar) * OneSplit.Loading << endl;
    //cout << "split value: " << OneSplit.value << endl;
    split_id_multi(Splitid, OneSplit, left_id, obs_id);  // get the left and right id
    
//DEBUG_Rcout << "--Left Num--" << left_id.n_elem << std::endl;
  //  DEBUG_Rcout << "--Right Num--" << obs_id.n_elem << std::endl;
    //std::cout << "leftNum: "<< left_id.n_elem <<" rightNum: "<< obs_id.n_elem << endl;
    
    // if this happens something about the splitting rule is wrong
    if (left_id.n_elem == N or obs_id.n_elem == N){
      
      cout << "Splitting Wrong" << endl;
      
      if(obs_id.n_elem == 0) obs_id = left_id;
      
      goto TERMINATENODE;
      
    }
    // check if the current tree is long enough to store two more nodes
    // if not, extend the current tree
    if ( OneTree.NodeType( OneTree.NodeType.size() - 2) > 0 )
    {
      DEBUG_Rcout << "  ------------- extend tree length: this shouldn't happen ----------- " << std::endl;
      // extend tree structure
      OneTree.extend();
     }
  
    // find the locations of next left and right nodes     
    OneTree.NodeType(Node) = 2; // 0: unused, 1: reserved; 2: internal node; 3: terminal node	
    size_t NextLeft = Node;
    size_t NextRight = Node;
    
    // Find next node
    OneTree.find_next_nodes(NextLeft, NextRight); 
    //DEBUG_Rcout << "  -- Next Left at --" << NextLeft << std::endl;
    //DEBUG_Rcout << "  -- Next Right at --" << NextRight << std::endl;
    
    
    // record tree 
    OneTree.SplitLoading(Node) = OneSplit.Loading;
    OneTree.SplitVar(Node) = OneSplit.SplitVar;
    OneTree.SplitValue(Node) = OneSplit.value;
    OneTree.LeftNode(Node) = NextLeft;
    OneTree.RightNode(Node) = NextRight;  
    
    //cout << " Loading : " << OneTree.SplitLoading(Node)  << endl;
    
    OneTree.NodeSize(Node) = left_id.n_elem + obs_id.n_elem;
    
    // split the left and right nodes 
   /*
    Graph_Cla_Split_A_Node(NextLeft,
                           OneTree,
                           CLA_DATA,
                           Param,
                           Param_RLT,
                           left_id,
                           var_id);
    
    Graph_Cla_Split_A_Node(NextRight,                          
                           OneTree,
                           CLA_DATA,
                           Param,
                           Param_RLT, 
                           obs_id, 
                           var_id);
   
   */
  }
}

// terminate and record a node

void Graph_Cla_Terminate_Node(size_t Node, 
                              Cla_Multi_Tree_Class& OneTree,
                              uvec& obs_id,                            
                              const uvec& Y,
                              const vec& obs_weight,                            
                              const PARAM_GLOBAL& Param,
                              bool useobsweight)
{
  //cout << "End" << endl;
  OneTree.NodeType(Node) = 3; // 0: unused, 1: reserved; 2: internal node; 3: terminal node
  OneTree.NodeSize(Node) = obs_id.n_elem;
  
  size_t N = obs_id.n_elem;
  double M = 0;
  
  for (size_t j = 0; j < N; j++)
  {
    if (N <= 0) break;
    
    M = M + Y(obs_id(j));
  }
  
  M = M / N;

  OneTree.NodeAve(Node) = M;
  
  if(isnan(M)) {
    
    cout << " Find NA Value" << endl;
    
      cout << N << endl;
    
  }
  //cout << "--terminate Mean--" << OneTree.NodeAve(Node)   << std::endl;
  
}
