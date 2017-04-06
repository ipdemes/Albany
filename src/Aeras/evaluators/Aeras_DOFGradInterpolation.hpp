//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_DOFGRAD_INTERPOLATION_HPP
#define AERAS_DOFGRAD_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"

namespace Aeras {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits>
class DOFGradInterpolation : public PHX::EvaluatorWithBaseImpl<Traits>,
 			     public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFGradInterpolation(Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> grad_val_qp;

  const int numNodes;
  const int numDims;
  const int numQPs;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct DOFGradInterpolation_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, DOFGradInterpolation_Tag> DOFGradInterpolation_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFGradInterpolation_Tag& tag, const int& i) const;

#endif
};

// Exact copy as above except data type is RealType instead of ScalarT
// to interpolate quantities without derivative arrays
template<typename EvalT, typename Traits>
class DOFGradInterpolation_noDeriv : public PHX::EvaluatorWithBaseImpl<Traits>,
 			     public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFGradInterpolation_noDeriv(Teuchos::ParameterList& p,
                               const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const RealType,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> grad_val_qp;

  const int numNodes;
  const int numDims;
  const int numQPs;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct DOFGradInterpolation_noDeriv_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, DOFGradInterpolation_noDeriv_Tag> DOFGradInterpolation_noDeriv_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFGradInterpolation_noDeriv_Tag& tag, const int& i) const;

#endif
};
}

#endif
