//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// 9/12/14: no Epetra!

#ifndef ALBANY_MODELEVALUATORT_HPP
#define ALBANY_MODELEVALUATORT_HPP

#include "Piro_TransientDecorator.hpp"

#include "Albany_Application.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Albany {

class ModelEvaluatorT
    : public Piro::TransientDecorator<ST, LO, GO, KokkosNode> {
 public:
  // Constructor
  ModelEvaluatorT(
      const Teuchos::RCP<Albany::Application>& app,
      const Teuchos::RCP<Teuchos::ParameterList>& appParams);

  /** \name Overridden from Thyra::ModelEvaluator<ST> . */
  //@{

  //! Return solution vector map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST>>
  get_x_space() const;

  //! Return residual vector map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST>>
  get_f_space() const;

  //! Return parameter vector map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST>>
  get_p_space(int l) const;

  //! Return response function map
  Teuchos::RCP<const Thyra::VectorSpaceBase<ST>>
  get_g_space(int j) const;

  //! Return array of parameter names
  Teuchos::RCP<const Teuchos::Array<std::string>>
  get_p_names(int l) const;
  Teuchos::ArrayView<const std::string>
  get_g_names(int j) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed");
  }

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getNominalValues() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getLowerBounds() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getUpperBounds() const;

  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_W_op() const;

  //! Create preconditioner operator
  Teuchos::RCP<Thyra::PreconditionerBase<ST>>
  create_W_prec() const;

  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST>>
  get_W_factory() const;

  //! Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  void
  reportFinalPoint(
      const Thyra::ModelEvaluatorBase::InArgs<ST>& finalPoint,
      const bool wasSolved);

  void
  allocateVectors();

  //@}

 protected:
  /** \name Overridden from Thyra::ModelEvaluatorDefaultBase<ST> . */
  //@{

  //! Create operator form of df/dp for distributed parameters
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DfDp_op_impl(int j) const;

  //! Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_op_impl(int j) const;

  //! Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_dot_op_impl(int j) const;

  //! Create operator form of dg/dx_dotdot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST>>
  create_DgDx_dotdot_op_impl(int j) const;

  //! Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;

  //! Evaluate model on InArgs
  void
  evalModelImpl(
      const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgs,
      const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const;

  //! Application object
  Teuchos::RCP<Albany::Application> app;

  Teuchos::RCP<Teuchos::Time> timer;

  //! Sacado parameter vector
  mutable Teuchos::Array<ParamVec> sacado_param_vec;

  //! Allocated Jacobian for sending to user preconditioner
  mutable Teuchos::RCP<Tpetra_CrsMatrix> Extra_W_crs;

  //! Whether the problem supplies its own preconditioner
  bool supplies_prec;

  //@}

 private:
  //! Number of parameter vectors
  int num_param_vecs;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  //! Cached nominal values
  Thyra::ModelEvaluatorBase::InArgs<ST> nominalValues;

  //! List of free parameter names
  Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string>>> param_names;

  //! Tpetra map for parameter vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Map>> tpetra_param_map;

  //! Tpetra parameter vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector>> tpetra_param_vec;

  //! Tpetra response vector
  Teuchos::Array<Teuchos::RCP<Thyra::VectorBase<ST>>> thyra_response_vec;

  //! Number of distributed parameter vectors
  int num_dist_param_vecs;

  //! List of enabled distributed parameters
  Teuchos::Array<std::string> dist_param_names;

  //! Distributed parameter library
  Teuchos::RCP<DistParamLib> distParamLib;

  //! Model uses time integration (velocities)
  bool supports_xdot;

  //! Model uses time integration (accelerations)
  bool supports_xdotdot;
};
}

#endif  // ALBANY_MODELEVALUATORT_HPP
