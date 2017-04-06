//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
QCAD::ResponseCenterOfMass<EvalT, Traits>::
ResponseCenterOfMass(Teuchos::ParameterList& p,
		     const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_vector),
  weights("Weights", dl->qp_scalar)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  // number of quad points per cell and dimension of space
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  //! Get material DB from parameters passed down from problem (if given)
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem = 
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  if(paramsFromProblem != Teuchos::null)
    materialDB = paramsFromProblem->get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");
  else materialDB = Teuchos::null;
  
  // User-specified parameters
  fieldName = plist->get<std::string>("Field Name");
  opRegion  = Teuchos::rcp( new QCAD::MeshRegion<EvalT, Traits>("Coord Vec","Weights",*plist,materialDB,dl) );
  
  // setup field
  field = decltype(field)(fieldName, scalar_dl);

  // add dependent fields
  this->addDependentField(field);
  this->addDependentField(coordVec);
  this->addDependentField(weights);
  opRegion->addDependentFields(this);

  this->setName(fieldName+" Response Center of Mass" );

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = 
    fieldName + " Local Response Center of Mass";
  std::string global_response_name = 
    fieldName + " Global Response Center of Mass";
  int worksetSize = scalar_dl->dimension(0);
  int responseSize = 4;
  Teuchos::RCP<PHX::DataLayout> local_response_layout =
    Teuchos::rcp(new MDALayout<Cell,Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout =
    Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, 
				       local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, 
					global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseCenterOfMass<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  opRegion->postRegistrationSetup(fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseCenterOfMass<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Zero out global response
  PHAL::set(this->global_response_eval, 0.0);  

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseCenterOfMass<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  PHAL::set(this->local_response_eval, 0.0);

  ScalarT integral, moment;

  if(!opRegion->elementBlockIsInRegion(workset.EBName))
    return;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    if(!opRegion->cellIsInRegion(cell)) continue;

    // Add to running total volume and mass moment
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      integral = field(cell,qp) * weights(cell,qp);
      this->local_response_eval(cell,3) += integral;
      this->global_response_eval(3) += integral;

      for(std::size_t i=0; i<numDims && i<3; i++) {
	moment = field(cell,qp) * weights(cell,qp) * coordVec(cell,qp,i);
	this->local_response_eval(cell,i) += moment;
	this->global_response_eval(i) += moment;
      }
    }

  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseCenterOfMass<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll(*workset.comm, Teuchos::REDUCE_SUM, this->global_response_eval);

  int iNormalizer = 3;
  if( fabs(this->global_response(iNormalizer)) > 1e-9 ) {
    for( int i=0; i < this->global_response.size(); i++) {
      if( i == iNormalizer ) continue;
      this->global_response_eval(i) /= this->global_response_eval(iNormalizer);
    }
    this->global_response_eval(iNormalizer) = 1.0;
  }

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseCenterOfMass<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseCenterOfMass Params"));;
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  Teuchos::RCP<const Teuchos::ParameterList> regionValidPL =
    QCAD::MeshRegion<EvalT,Traits>::getValidParameters();
  validPL->setParameters(*regionValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Name", "", "Scalar field from which to compute center of mass");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

