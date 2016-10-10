//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
HydrologyResidualPotentialEqn<EvalT, Traits, HasThicknessEqn, IsStokesCoupling>::
HydrologyResidualPotentialEqn (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl) :
  BF        (p.get<std::string> ("BF Name"), dl->node_qp_scalar),
  GradBF    (p.get<std::string> ("Gradient BF Name"), dl->node_qp_gradient),
  w_measure (p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar),
  q         (p.get<std::string> ("Water Discharge QP Variable Name"), dl->qp_gradient),
  N         (p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar),
  m         (p.get<std::string> ("Melting Rate QP Variable Name"), dl->qp_scalar),
  h         (p.get<std::string> ("Water Thickness QP Variable Name"), dl->qp_scalar),
  omega     (p.get<std::string> ("Surface Water Input QP Variable Name"), dl->qp_scalar),
  u_b       (p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar),
  residual  (p.get<std::string> ("Potential Eqn Residual Name"),dl->node_scalar)
{
  if (IsStokesCoupling)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(2);
    numQPs   = dl->qp_scalar->dimension(2);
    numDims  = dl->qp_gradient->dimension(3);

    sideSetName = p.get<std::string>("Side Set Name");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
    numDims  = dl->qp_gradient->dimension(2);
  }

  this->addDependentField(BF);
  this->addDependentField(GradBF);
  this->addDependentField(w_measure);
  this->addDependentField(q);
  this->addDependentField(N);
  this->addDependentField(h);
  this->addDependentField(m);
  this->addDependentField(omega);
  this->addDependentField(u_b);

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  double rho_i      = physical_params.get<double>("Ice Density", 910.0);
  double rho_w      = physical_params.get<double>("Water Density", 1028.0);
  bool melting_mass = hydrology_params.get<bool>("Use Melting In Conservation Of Mass", false);
  bool melting_cav  = hydrology_params.get<bool>("Use Melting In Cavities Equation", false);
  use_eff_cav       = (hydrology_params.get<bool>("Use Effective Cavities Height", true) ? 1.0 : 0.0);
  eta_i             = physical_params.get<double>("Ice Viscosity",-1.0);

  rho_combo = (melting_mass ? 1 : 0) / rho_w - (melting_cav ? 1 : 0) / rho_i;
  mu_w      = physical_params.get<double>("Water Viscosity");
  h_r       = hydrology_params.get<double>("Bed Bumps Height");
  l_r       = hydrology_params.get<double>("Bed Bumps Length");
  A         = hydrology_params.get<double>("Flow Factor Constant");

  // Scalings, needed to account for different units: ice velocity
  // is in m/yr rather than m/s, while all other quantities are in SI units.
  double yr_to_s = 365.25*24*3600;
  A   *= 1./(1000*yr_to_s);     // Need to adjust A, which is given in k^-{n+1} Pa^-n yr^-1, to [kPa]^-n s^-1.
  l_r *= yr_to_s;               // Need to adjust u_b from m/yr to m/s. Since it's always divided by l_r, we simply scale l_r

  this->setName("HydrologyResidualPotentialEqn"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
void HydrologyResidualPotentialEqn<EvalT, Traits, HasThicknessEqn, IsStokesCoupling>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(q,fm);
  this->utils.setFieldData(N,fm);
  this->utils.setFieldData(h,fm);
  this->utils.setFieldData(m,fm);
  this->utils.setFieldData(omega,fm);
  this->utils.setFieldData(u_b,fm);

  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
void HydrologyResidualPotentialEqn<EvalT, Traits, HasThicknessEqn, IsStokesCoupling>::
evaluateFields (typename Traits::EvalData workset)
{
  // Omega is in mm/d rather than m/s
  double scaling_omega = 0.001/(24*3600);

  if (IsStokesCoupling)
  {
    // Zero out, to avoid leaving stuff from previous workset!
    residual.deep_copy(ScalarT(0.));

    if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
      return;

    ScalarT res_qp, res_node;
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int node=0; node < numNodes; ++node)
      {
        res_node = 0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          res_qp = rho_combo*m(cell,side,qp) + omega(cell,side,qp)
                 - (h_r -h(cell,side,qp))*u_b(cell,side,qp)/l_r
                 + h(cell,side,qp)*std::pow(A*N(cell,side,qp),3);

          res_qp *= BF(cell,side,node,qp);

          for (int dim=0; dim<numDims; ++dim)
          {
            res_qp += q(cell,side,qp,dim) * GradBF(cell,side,node,qp,dim);
          }

          res_node += res_qp * w_measure(cell,side,qp);
        }
        residual (cell,side,node) += res_node;
      }
    }
  }
  else
  {
    if (eta_i>0)
    {
      ScalarT res_qp, res_node;
      for (int cell=0; cell < workset.numCells; ++cell)
      {
        for (int node=0; node < numNodes; ++node)
        {
          res_node = 0;
          for (int qp=0; qp < numQPs; ++qp)
          {
            res_qp = rho_combo*m(cell,qp) + omega(cell,qp)*scaling_omega
                   - (h_r - use_eff_cav*h(cell,qp))*u_b(cell,qp)/l_r
                   + h(cell,qp)*N(cell,qp)/eta_i;

            res_qp *= BF(cell,node,qp);

            for (int dim=0; dim<numDims; ++dim)
            {
              res_qp += q(cell,qp,dim) * GradBF(cell,node,qp,dim);
            }

            res_node += res_qp * w_measure(cell,qp);
          }

          residual (cell,node) = res_node;
        }
      }
    }
    else
    {
      ScalarT res_qp, res_node;
      for (int cell=0; cell < workset.numCells; ++cell)
      {
        for (int node=0; node < numNodes; ++node)
        {
          res_node = 0;
          for (int qp=0; qp < numQPs; ++qp)
          {
            res_qp = rho_combo*m(cell,qp) + omega(cell,qp)*scaling_omega
                   - (h_r - use_eff_cav*h(cell,qp))*u_b(cell,qp)/l_r
                   + h(cell,qp)*A*std::pow(N(cell,qp),3);

            res_qp *= BF(cell,node,qp);

            for (int dim=0; dim<numDims; ++dim)
            {
              res_qp += q(cell,qp,dim) * GradBF(cell,node,qp,dim);
            }

            res_node += res_qp * w_measure(cell,qp);
          }

          residual (cell,node) = res_node;
        }
      }
    }
  }
}

} // Namespace FELIX
