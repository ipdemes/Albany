/*
 * FELIX_LiquidWaterFraction.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_LIQUIDWATERFRACTION_HPP_
#define FELIX_LIQUIDWATERFRACTION_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

  /** \brief Liquid Water Fraction

    This evaluator computes the liquid water fraction in temperate ice
   */

  template<typename EvalT, typename Traits, typename Type>
  class LiquidWaterFraction: public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:

    typedef typename EvalT::ScalarT ScalarT;

    LiquidWaterFraction (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup (typename Traits::SetupData d,
                                PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

  private:
    // Input:
    PHX::MDField<const Type,Cell,Node> 		enthalpyHs;  //[MW s m^{-3}]
    PHX::MDField<const ScalarT,Cell,Node> 	enthalpy;  //[MW s m^{-3}]
    PHX::MDField<const ScalarT,Dim> 		homotopy;

    // Output:
    PHX::MDField<ScalarT,Cell,Node> phi;         //[adim]

    int numNodes;

    double L;      //[J kg^{-1}] = [ m^2 s^{-2}]
    double rho_w;  //[kg m^{-3}]

    ScalarT printedAlpha;

  };

} // Namespace FELIX

#endif /* FELIX_LIQUIDWATERFRACTION_HPP_ */
