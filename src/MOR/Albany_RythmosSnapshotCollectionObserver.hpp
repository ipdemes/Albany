//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP
#define ALBANY_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP

#include "Rythmos_IntegrationObserverBase.hpp"

#include "Albany_SnapshotCollection.hpp"

namespace Albany {

class MultiVectorOutputFile;

class RythmosSnapshotCollectionObserver : public Rythmos::IntegrationObserverBase<double> {
public:
  RythmosSnapshotCollectionObserver(
      int period,
      Teuchos::RCP<MultiVectorOutputFile> snapshotFile);

  // Overridden
  virtual Teuchos::RCP<Rythmos::IntegrationObserverBase<double> > cloneIntegrationObserver() const;

  virtual void resetIntegrationObserver(const Rythmos::TimeRange<double> &integrationTimeDomain);

  virtual void observeStartTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

  virtual void observeCompletedTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

private:
  SnapshotCollection snapshotCollector_;

  virtual void observeTimeStep(
    const Rythmos::StepperBase<double> &stepper,
    const Rythmos::StepControlInfo<double> &stepCtrlInfo,
    const int timeStepIter);

};

} // namespace Albany

#endif /*ALBANY_RYTHMOSSNAPSHOTCOLLECTIONOBSERVER_HPP*/
