//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ObserverFactory.hpp"

#include "MOR_MultiVectorOutputFile.hpp"
#include "MOR_MultiVectorOutputFileFactory.hpp"
#include "MOR_ReducedSpace.hpp"
#include "MOR_ReducedSpaceFactory.hpp"

#include "MOR_SnapshotCollectionObserver.hpp"
#include "MOR_ProjectionErrorObserver.hpp"
#include "MOR_FullStateReconstructor.hpp"

#include "MOR_RythmosSnapshotCollectionObserver.hpp"
#include "MOR_RythmosProjectionErrorObserver.hpp"
#include "MOR_RythmosFullStateReconstructor.hpp"

#include "Rythmos_CompositeIntegrationObserver.hpp"

#include "Teuchos_ParameterList.hpp"

#include <string>

namespace MOR {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;
using ::Teuchos::sublist;

namespace { // anonymous

RCP<ParameterList> fillDefaultOutputParams(const RCP<ParameterList> &params, std::string defaultValue)
{
  params->get("Output File Group Name", defaultValue);
  params->get("Output File Default Base File Name", defaultValue);
  return params;
}

RCP<ParameterList> fillDefaultProjectionErrorOutputParams(const RCP<ParameterList> &params)
{
  return fillDefaultOutputParams(params, "proj_error");
}

RCP<ParameterList> fillDefaultSnapshotOutputParams(const RCP<ParameterList> &params)
{
  return fillDefaultOutputParams(params, "snapshots");
}

RCP<MultiVectorOutputFile> createOutputFile(const RCP<ParameterList> &params)
{
  MultiVectorOutputFileFactory factory(params);
  return factory.create();
}

RCP<MultiVectorOutputFile> createProjectionErrorOutputFile(const RCP<ParameterList> &params)
{
  return createOutputFile(fillDefaultProjectionErrorOutputParams(params));
}

RCP<MultiVectorOutputFile> createSnapshotOutputFile(const RCP<ParameterList> &params)
{
  return createOutputFile(fillDefaultSnapshotOutputParams(params));
}

int getSnapshotPeriod(const RCP<ParameterList> &params)
{
  return params->get("Period", 1);
}

} // end anonymous namespace

ObserverFactory::ObserverFactory(
      const Teuchos::RCP<ReducedSpaceFactory> &spaceFactory,
      const Teuchos::RCP<Teuchos::ParameterList> &parentParams) :
  spaceFactory_(spaceFactory),
  params_(sublist(parentParams, "Model Order Reduction"))
{
  // Nothing to do
}

RCP<NOX::Epetra::Observer> ObserverFactory::create(const RCP<NOX::Epetra::Observer> &child)
{
  RCP<NOX::Epetra::Observer> result = child;

  if (collectSnapshots()) {
    const RCP<ParameterList> params = getSnapParameters();
    const RCP<MultiVectorOutputFile> snapOutputFile = createSnapshotOutputFile(params);
    const int period = getSnapshotPeriod(params);
    result = rcp(new SnapshotCollectionObserver(period, snapOutputFile, result));
  }

  if (computeProjectionError()) {
    const RCP<ParameterList> params = getErrorParameters();
    const RCP<MultiVectorOutputFile> errorOutputFile = createProjectionErrorOutputFile(params);
    const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
    result = rcp(new ProjectionErrorObserver(projectionSpace, errorOutputFile, result));
  }

  if (useReducedOrderModel()) {
    const RCP<ParameterList> params = getReducedOrderModelParameters();
    const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
    result = rcp(new FullStateReconstructor(projectionSpace, result));
  }

  return result;
}

RCP<Rythmos::IntegrationObserverBase<double> > ObserverFactory::create(const RCP<Rythmos::IntegrationObserverBase<double> > &child) {
  RCP<Rythmos::IntegrationObserverBase<double> > result = child;

  if (collectSnapshots()) {
    const RCP<Rythmos::CompositeIntegrationObserver<double> > composite = Rythmos::createCompositeIntegrationObserver<double>();
    composite->addObserver(result);
    {
      const RCP<ParameterList> params = getSnapParameters();
      const RCP<MultiVectorOutputFile> snapOutputFile = createSnapshotOutputFile(params);
      const int period = getSnapshotPeriod(params);
      composite->addObserver(rcp(new RythmosSnapshotCollectionObserver(period, snapOutputFile)));
    }
    result = composite;
  }

  if (computeProjectionError()) {
    const RCP<Rythmos::CompositeIntegrationObserver<double> > composite = Rythmos::createCompositeIntegrationObserver<double>();
    composite->addObserver(result);
    {
      const RCP<ParameterList> params = getErrorParameters();
      const RCP<MultiVectorOutputFile> errorOutputFile = createProjectionErrorOutputFile(params);
      const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
      composite->addObserver(rcp(new RythmosProjectionErrorObserver(projectionSpace, errorOutputFile)));
    }
    result = composite;
  }

  if (useReducedOrderModel()) {
    const RCP<ParameterList> params = getReducedOrderModelParameters();
    const RCP<ReducedSpace> projectionSpace = spaceFactory_->create(params);
    result = rcp(new RythmosFullStateReconstructor(projectionSpace, result));
  }

  return result;
}

bool ObserverFactory::collectSnapshots() const
{
  return getSnapParameters()->get("Activate", false);
}

bool ObserverFactory::computeProjectionError() const
{
  return getErrorParameters()->get("Activate", false);
}

bool ObserverFactory::useReducedOrderModel() const
{
  return getReducedOrderModelParameters()->get("Activate", false);
}

RCP<ParameterList> ObserverFactory::getSnapParameters() const
{
  return sublist(params_, "Snapshot Collection");
}

RCP<ParameterList> ObserverFactory::getErrorParameters() const
{
  return sublist(params_, "Projection Error");
}

RCP<ParameterList> ObserverFactory::getReducedOrderModelParameters() const
{
  return sublist(params_, "Reduced-Order Model");
}

} // namespace MOR
