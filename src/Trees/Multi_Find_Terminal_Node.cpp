//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classificaiton
//  **********************************

// my header file
# include "../RLT.h"
# include "../Utility/Utility.h"
# include "Trees.h"

using namespace Rcpp;
using namespace arma;

void Multi_Find_Terminal_Node(size_t Node, 
              							const Multi_Tree_Class& OneTree,
              							const mat& X,
              							const uvec& Ncat,
              							uvec& proxy_id,
              							const uvec& real_id,
              							uvec& TermNode)
{
 
  size_t size = proxy_id.n_elem;
  
  //std::cout << "/// Start at node ///" << Node << " n is " << size << std::endl;
  
  if (OneTree.NodeType[Node] == 3)
  {
    for ( size_t i=0; i < size; i++ )
      TermNode(proxy_id(i)) = Node;
    
  }else{
    
    uvec id_goright(proxy_id.n_elem, fill::zeros);
    uvec SplitVar = OneTree.SplitVar(Node);
    vec SplitLoading = OneTree.SplitLoading(Node);
    double SplitValue = OneTree.SplitValue(Node);
    double xtemp = 0;
    
    vec xvec = X( real_id( proxy_id ), SplitVar) * SplitLoading;
    
    //cout << "Loading : " << SplitLoading << endl;
    //std::cout << "Vec : " << xvec << std::endl;
    //cout << "Splitvalue : " << SplitValue << endl;
    
    for (size_t i = 0; i < size ; i++)
    {
        xtemp = xvec(i);
        
        if (xtemp > SplitValue)
          id_goright(i) = 1;
    }
    
    uvec left_proxy = proxy_id(find(id_goright == 0));
    proxy_id = proxy_id(find(id_goright == 1));
    
    //cout << "left : " << left_proxy.n_elem << endl;
    //cout << "right : " << proxy_id.n_elem << endl;
    
    // left node 
    
    if (left_proxy.n_elem > 0)
    {
      Multi_Find_Terminal_Node(OneTree.LeftNode[Node], OneTree, X, Ncat, left_proxy, real_id, TermNode);
    }
    
    // right node
    if (proxy_id.n_elem > 0)
    {
      Multi_Find_Terminal_Node(OneTree.RightNode[Node], OneTree, X, Ncat, proxy_id, real_id, TermNode);      
    }
    
  }
  
  return;

}



