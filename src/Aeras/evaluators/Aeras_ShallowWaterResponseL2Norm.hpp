//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOW_WATER_RESPONSE_L2_NORM_HPP
#define AERAS_SHALLOW_WATER_RESPONSE_L2_NORM_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"


namespace Aeras {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ShallowWaterResponseL2Norm : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ShallowWaterResponseL2Norm(Teuchos::ParameterList& p,
			  const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    PHX::MDField<const ScalarT,Cell,Node,VecDim> flow_state_field; //flow state field at nodes
    PHX::MDField<const MeshScalarT> weighted_measure;
    std::size_t numQPs, numDims, numNodes, nPrimaryDOFs;
    int responseSize; //length of response vector; 4 for this response

  
  
  };
	
}

#endif
