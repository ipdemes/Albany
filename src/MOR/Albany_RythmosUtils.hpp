//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_RYTHMOSUTILS_HPP
#define ALBANY_RYTHMOSUTILS_HPP

#include "Rythmos_StepperBase.hpp"

#include "Thyra_ProductVectorBase.hpp"

#include "Thyra_SpmdVectorSpaceBase.hpp"
#include "Teuchos_Comm.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Ptr.hpp"

namespace Albany {

template <typename Scalar>
Teuchos::RCP<const Thyra::VectorBase<Scalar> > getRythmosState(const Teuchos::RCP<const Thyra::VectorBase<Scalar> > &in) {
  typedef Thyra::ProductVectorBase<Scalar> PVB;
  const Teuchos::RCP<const PVB> in_with_sens = Teuchos::rcp_dynamic_cast<const PVB>(in);
  if (Teuchos::nonnull(in_with_sens)) {
    return in_with_sens->getVectorBlock(0);
  } else {
    return in;
  }
}

template <typename Scalar>
Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal> > getComm(const Thyra::VectorSpaceBase<Scalar> &space) {
  typedef Thyra::SpmdVectorSpaceBase<Scalar> SVSB;
  const Teuchos::Ptr<const SVSB> space_downcasted = Teuchos::ptr_dynamic_cast<const SVSB>(Teuchos::ptrFromRef(space));
  if (Teuchos::nonnull(space_downcasted)) {
    return space_downcasted->getComm();
  } else {
    return Teuchos::null;
  }
}

} // namespace Albany

#endif /* ALBANY_RYTHMOSUTILS_HPP */
