//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZSCALARADVECTIONRESID_HPP
#define AERAS_XZSCALARADVECTIONRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace Aeras {
/** \brief XZScalarAdvection equation Residual for atmospheric modeling

    This evaluator computes the residual of the XZScalarAdvection equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZScalarAdvectionResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>,
                   public Sacado::ParameterAccessor<EvalT, SPL_Traits>  {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZScalarAdvectionResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<const ScalarT,Cell,QuadPoint> rho;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> rhoGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint> rhoDot;
  PHX::MDField<const MeshScalarT,Cell,Point,Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;

  ScalarT Re; // Reynolds number (demo on how to get info from input file)

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
};
}

#endif
