//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_AnisotropicViscoplasticModel_hpp)
#define LCM_AnisotropicViscoplasticModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM
{

//! \brief Anisotropic Viscoplastic Constitutive Model
template<typename EvalT, typename Traits>
class AnisotropicViscoplasticModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  using Base = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap = typename Base::FieldMap;

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  // optional temperature support
  using ConstitutiveModel<EvalT, Traits>::have_temperature_;
  using ConstitutiveModel<EvalT, Traits>::expansion_coeff_;
  using ConstitutiveModel<EvalT, Traits>::ref_temperature_;
  using ConstitutiveModel<EvalT, Traits>::heat_capacity_;
  using ConstitutiveModel<EvalT, Traits>::density_;
  using ConstitutiveModel<EvalT, Traits>::temperature_;

  ///
  /// Constructor
  ///
  AnisotropicViscoplasticModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~AnisotropicViscoplasticModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      DepFieldMap dep_fields,
      FieldMap eval_fields);

  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      DepFieldMap dep_fields,
      FieldMap eval_fields){
         TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
 }


private:

  ///
  /// Private to prohibit copying
  ///
  AnisotropicViscoplasticModel(const AnisotropicViscoplasticModel&);

  ///
  /// Private to prohibit copying
  ///
  AnisotropicViscoplasticModel& operator=(const AnisotropicViscoplasticModel&);

};
}

#endif
