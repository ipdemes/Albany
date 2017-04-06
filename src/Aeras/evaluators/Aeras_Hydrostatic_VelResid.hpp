//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_VELRESID_HPP
#define AERAS_HYDROSTATIC_VELRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Hydrostatic equation Residual for atmospheric modeling

    This evaluator computes the residual of the Hydrostatic equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class Hydrostatic_VelResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Hydrostatic_VelResid(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubature;
  Kokkos::DynRankView<RealType, PHX::Device>    refPoints;
  Kokkos::DynRankView<RealType, PHX::Device>    refWeights;
  Kokkos::DynRankView<RealType, PHX::Device>    grad_at_cub_points;

  // vorticity only returns the component in the radial direction
  //void get_vorticity(const Kokkos::DynRankView<ScalarT, PHX::Device>  & fieldAtNodes,
  //    std::size_t cell, std::size_t level, Kokkos::DynRankView<ScalarT, PHX::Device>  & curl);

  void get_coriolis(std::size_t cell, Kokkos::DynRankView<ScalarT, PHX::Device>  & coriolis);

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint>         wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim>     GradBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim>     wGradBF;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim>          sphere_coord;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim,Dim>      jacobian;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint>              jacobian_det;

  PHX::MDField<const ScalarT,Cell,QuadPoint,Level,Dim>  keGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level,Dim>  PhiGrad;
  PHX::MDField<const ScalarT,Cell,Node,Level,Dim>  etadotdVelx;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level,Dim>  pGrad;
  PHX::MDField<const ScalarT,Cell,Node,Level,Dim>  Velx;
  PHX::MDField<const ScalarT,Cell,Node,Level,Dim>       VelxNode;
  PHX::MDField<const ScalarT,Cell,Node,Level,Dim>  VelxDot;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level,Dim>  DVelx;
  PHX::MDField<const ScalarT,Cell,Node,Level>      density;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level>      vorticity;


  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> Residual;

  const double viscosity;
  const double hyperviscosity;
  const double AlphaAngle;
  const double Omega;

  const int numNodes;
  const int numQPs;
  const int numDims;
  const int numLevels;

  bool obtainLaplaceOp;
  bool pureAdvection;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Hydrostatic_VelResid_Tag{};
  struct Hydrostatic_VelResid_pureAdvection_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, Hydrostatic_VelResid_Tag> Hydrostatic_VelResid_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, Hydrostatic_VelResid_pureAdvection_Tag> Hydrostatic_VelResid_pureAdvection_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_VelResid_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_VelResid_pureAdvection_Tag& tag, const int& i) const;

#endif
};
}

#endif
