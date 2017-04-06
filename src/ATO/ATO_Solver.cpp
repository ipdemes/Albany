////*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "ATO_TopoTools.hpp"
#include "ATO_Types.hpp"

/* GAH FIXME - Silence warning:
TRILINOS_DIR/../../../include/pecos_global_defs.hpp:17:0: warning: 
        "BOOST_MATH_PROMOTE_DOUBLE_POLICY" redefined [enabled by default]
Please remove when issue is resolved
*/
#undef BOOST_MATH_PROMOTE_DOUBLE_POLICY

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Adapt_NodalDataVector.hpp"
#include "Petra_Converters.hpp"
#include "Epetra_LinearProblem.h"
#include "AztecOO.h"

#ifdef ATO_USES_ISOLIB
#include "Albany_STKDiscretization.hpp"
#include "STKExtract.hpp"
#endif


MPI_Datatype MPI_GlobalPoint;

bool ATO::operator< (ATO::GlobalPoint const & a, ATO::GlobalPoint const & b){return a.gid < b.gid;}
ATO::GlobalPoint::GlobalPoint(){coords[0]=0.0; coords[1]=0.0; coords[2]=0.0;}


/******************************************************************************/
ATO::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Teuchos_Comm>& comm,
       const Teuchos::RCP<const Tpetra_Vector>& initial_guess)
: c_num_parameters(0), // no parameters
  c_num_responses(1),  // only response is solution vector
  _solverComm(comm), 
  _mainAppParams(appParams)
/******************************************************************************/
{
  zeroSet();

  objectiveValue = Teuchos::rcp(new double[1]);
  *objectiveValue = 0.0;

  constraintValue = Teuchos::rcp(new double[1]);
  *constraintValue = 0.0;

  ///*** PROCESS TOP LEVEL PROBLEM ***///
 

  // Validate Problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  _numPhysics = problemParams.get<int>("Number of Subproblems", 1);

  int numHomogProblems = problemParams.get<int>("Number of Homogenization Problems", 0);
  _homogenizationSets.resize(numHomogProblems);

  problemParams.validateParameters(*getValidProblemParameters(),0);


  // Parse topologies
  Teuchos::ParameterList& topoParams = problemParams.get<Teuchos::ParameterList>("Topologies");
  int ntopos = topoParams.get<int>("Number of Topologies");

  if( topoParams.isType<bool>("Read From Restart") )
     _is_restart = topoParams.get<bool>("Read From Restart");

  _topologyInfoStructsT.resize(ntopos);
  _topologyArrayT = Teuchos::rcp( new Teuchos::Array<Teuchos::RCP<ATO::Topology> >(ntopos) );
  for(int itopo=0; itopo<ntopos; itopo++){
    _topologyInfoStructsT[itopo] = Teuchos::rcp(new TopologyInfoStructT);
    Teuchos::ParameterList& tParams = topoParams.sublist(Albany::strint("Topology",itopo));
    _topologyInfoStructsT[itopo]->topologyT = Teuchos::rcp(new Topology(tParams, itopo));
    (*_topologyArrayT)[itopo] = _topologyInfoStructsT[itopo]->topologyT;
  }

  // currently all topologies must have the same entity type
  entityType = _topologyInfoStructsT[0]->topologyT->getEntityType();
  for(int itopo=1; itopo<ntopos; itopo++){
    TEUCHOS_TEST_FOR_EXCEPTION(
    _topologyInfoStructsT[itopo]->topologyT->getEntityType() != entityType,
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  Topologies must all have the same entity type." << std::endl);
  }

  // Parse and create optimizer
  Teuchos::ParameterList& optimizerParams = 
    problemParams.get<Teuchos::ParameterList>("Topological Optimization");
  ATO::OptimizerFactory optimizerFactory;
  _optimizer = optimizerFactory.create(optimizerParams);
  _optimizer->SetInterface(this);
  _optimizer->SetCommunicator(comm);

  _writeDesignFrequency = problemParams.get<int>("Design Output Frequency", 0);

  // Parse and create aggregator
  Teuchos::ParameterList& aggregatorParams = problemParams.get<Teuchos::ParameterList>("Objective Aggregator");

  ATO::AggregatorFactory aggregatorFactory;
  // Parse and create objective aggregator
  Teuchos::ParameterList& objAggregatorParams = problemParams.get<Teuchos::ParameterList>("Objective Aggregator");
  _objAggregator = aggregatorFactory.create(objAggregatorParams, entityType, ntopos);

  // Parse and create constraint aggregator
  if(problemParams.isType<Teuchos::ParameterList>("Constraint Aggregator")){
    Teuchos::ParameterList& conAggregatorParams = problemParams.get<Teuchos::ParameterList>("Constraint Aggregator");
    _conAggregator = aggregatorFactory.create(conAggregatorParams, entityType, ntopos);
  } else {
    _conAggregator = Teuchos::null;
  }

  // Parse filters
  if( problemParams.isType<Teuchos::ParameterList>("Spatial Filters")){
    Teuchos::ParameterList& filtersParams = problemParams.get<Teuchos::ParameterList>("Spatial Filters");
    int nFilters = filtersParams.get<int>("Number of Filters");
    for(int ifltr=0; ifltr<nFilters; ifltr++){
      std::stringstream filterStream;
      filterStream << "Filter " << ifltr;
      Teuchos::ParameterList& filterParams = filtersParams.get<Teuchos::ParameterList>(filterStream.str());
      Teuchos::RCP<ATO::SpatialFilter> newFilter = Teuchos::rcp( new ATO::SpatialFilter(filterParams) );
      filters.push_back(newFilter);
    }
  }
  
  // Assign requested filters to topologies
  for(int itopo=0; itopo<ntopos; itopo++){
    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    Teuchos::RCP<Topology> topoT = topoStructT->topologyT;

    topoStructT->filterIsRecursiveT = topoParams.get<bool>("Apply Filter Recursively", true);

    int topologyFilterIndex = topoT->SpatialFilterIndex();
    if( topologyFilterIndex >= 0 ){
      TEUCHOS_TEST_FOR_EXCEPTION( topologyFilterIndex >= filters.size(),
        Teuchos::Exceptions::InvalidParameter, std::endl 
        << "Error!  Spatial filter " << topologyFilterIndex << "requested but not defined." << std::endl);
      topoStructT->filterT = filters[topologyFilterIndex];
    }

    int topologyOutputFilter = topoT->TopologyOutputFilter();
    if( topologyOutputFilter >= 0 ){
      TEUCHOS_TEST_FOR_EXCEPTION( topologyOutputFilter >= filters.size(),
        Teuchos::Exceptions::InvalidParameter, std::endl 
        << "Error!  Spatial filter " << topologyFilterIndex << "requested but not defined." << std::endl);
      topoStructT->postFilterT = filters[topologyOutputFilter];
    }
  }

  int derivativeFilterIndex = objAggregatorParams.get<int>("Spatial Filter", -1);
  if( derivativeFilterIndex >= 0 ){
    TEUCHOS_TEST_FOR_EXCEPTION( derivativeFilterIndex >= filters.size(),
      Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Spatial filter " << derivativeFilterIndex << "requested but not defined." << std::endl);
    _derivativeFilter = filters[derivativeFilterIndex];
  }

  // Get and set the default Piro parameters from a file, if given
  std::string piroFilename  = problemParams.get<std::string>("Piro Defaults Filename", "");
  if(piroFilename.length() > 0) {
    Teuchos::RCP<Teuchos::ParameterList> defaultPiroParams = 
      Teuchos::createParameterList("Default Piro Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(piroFilename, defaultPiroParams.ptr(), *comm);
    Teuchos::ParameterList& piroList = appParams->sublist("Piro", false);
    piroList.setParametersNotAlreadySet(*defaultPiroParams);
  }
  
  // set verbosity
  _is_verbose = (comm->getRank() == 0) && problemParams.get<bool>("Verbose Output", false);




  ///*** PROCESS SUBPROBLEM(S) ***///
   
  _subProblemAppParams.resize(_numPhysics);
  _subProblems.resize(_numPhysics);
  for(int i=0; i<_numPhysics; i++){

    _subProblemAppParams[i] = createInputFile(appParams, i);
    _subProblems[i] = CreateSubSolver( _subProblemAppParams[i], _solverComm);

    // ensure that all subproblems are topology based (i.e., optimizable)
    Teuchos::RCP<Albany::AbstractProblem> problem = _subProblems[i].app->getProblem();
    ATO::OptimizationProblem* atoProblem = 
      dynamic_cast<ATO::OptimizationProblem*>(problem.get());
    TEUCHOS_TEST_FOR_EXCEPTION( 
      atoProblem == NULL, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Requested subproblem does not support topologies." << std::endl);
  }
  
  ///*** PROCESS HOMOGENIZATION SUBPROBLEM(S) ***///

  for(int iProb=0; iProb<numHomogProblems; iProb++){
    HomogenizationSet& hs = _homogenizationSets[iProb];
    std::stringstream homogStream;
    homogStream << "Homogenization Problem " << iProb;
    Teuchos::ParameterList& 
      homogParams = problemParams.get<Teuchos::ParameterList>(homogStream.str());
    hs.homogDim = homogParams.get<int>("Number of Spatial Dimensions");
    
    // parse the name and type of the homogenized constants
    Teuchos::ParameterList& responsesList = homogParams.sublist("Problem").sublist("Response Functions");
    int nResponses = responsesList.get<int>("Number of Response Vectors");
    bool responseFound = false;
    for(int iResponse=0; iResponse<nResponses; iResponse){
      Teuchos::ParameterList& responseList = responsesList.sublist(Albany::strint("Response Vector",iResponse));
      std::string rname = responseList.get<std::string>("Name");
      if(rname == "Homogenized Constants Response"){
        hs.name = responseList.get<std::string>("Homogenized Constants Name");
        hs.type = responseList.get<std::string>("Homogenized Constants Type");
        responseFound = true; break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION (responseFound == false,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Could not find viable homogenization response." << std::endl);

    int nHomogSubProblems = 0;
    if( hs.type == "4th Rank Voigt" ){
      for(int i=1; i<=hs.homogDim; i++) nHomogSubProblems += i;
    } else
    if( hs.type == "2nd Rank Tensor" ){
      nHomogSubProblems = hs.homogDim;
    } else
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error! Unknown type (" << hs.type << ").  " << 
        std::endl << "Options are '4th Rank Voigt' or '2nd Rank Tensor'" << std::endl);

    hs.homogenizationAppParams.resize(nHomogSubProblems);
    hs.homogenizationProblems.resize(nHomogSubProblems);
 
    for(int iSub=0; iSub<nHomogSubProblems; iSub++){

      hs.homogenizationAppParams[iSub] = createHomogenizationInputFile(appParams, homogParams, iProb, iSub, hs.homogDim);
      hs.homogenizationProblems[iSub] = CreateSubSolver( hs.homogenizationAppParams[iSub], _solverComm);
    }


  }


  // store a pointer to the first problem as an ATO::OptimizationProblem for callbacks
  Teuchos::RCP<Albany::AbstractProblem> problem = _subProblems[0].app->getProblem();
  _atoProblem = dynamic_cast<ATO::OptimizationProblem*>(problem.get());
  _atoProblem->setDiscretization(_subProblems[0].app->getDiscretization());
  _atoProblem->setCommunicator(comm);
  _atoProblem->InitTopOpt();
  


  // get solution map from first subproblem
  const SolverSubSolver& sub = _subProblems[0];
  Teuchos::RCP<const Epetra_Map> sub_x_map = sub.app->getMap();
  TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
  _epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));

  Teuchos::RCP<Albany::Application> app = _subProblems[0].app;
  Teuchos::RCP<Albany::AbstractDiscretization> disc = app->getDiscretization();

  localNodeMapT   = disc->getNodeMapT();
  overlapNodeMapT = disc->getOverlapNodeMapT();

  for(int itopo=0; itopo<ntopos; itopo++){
    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    if(topoStructT->postFilterT != Teuchos::null ){
      topoStructT->filteredOverlapVectorT = Teuchos::rcp(new Tpetra_Vector(overlapNodeMapT));
      topoStructT->filteredVectorT  = Teuchos::rcp(new Tpetra_Vector(localNodeMapT));
    } else {
      topoStructT->filteredOverlapVectorT = Teuchos::null;
      topoStructT->filteredVectorT = Teuchos::null;
    }

    // create overlap topo vector for output purposes
    topoStructT->overlapVectorT = Teuchos::rcp(new Tpetra_Vector(overlapNodeMapT));
    topoStructT->localVectorT   = Teuchos::rcp(new Tpetra_Vector(localNodeMapT));

  } 

  overlapObjectiveGradientVecT.resize(ntopos);
  ObjectiveGradientVecT.resize(ntopos);
  overlapConstraintGradientVecT.resize(ntopos);
  ConstraintGradientVecT.resize(ntopos);
  for(int itopo=0; itopo<ntopos; itopo++){
    overlapObjectiveGradientVecT[itopo] = Teuchos::rcp(new Tpetra_Vector(overlapNodeMapT));
    ObjectiveGradientVecT[itopo]  = Teuchos::rcp(new Tpetra_Vector(localNodeMapT));
    overlapConstraintGradientVecT[itopo] = Teuchos::rcp(new Tpetra_Vector(overlapNodeMapT));
    ConstraintGradientVecT[itopo]  = Teuchos::rcp(new Tpetra_Vector(localNodeMapT));
  } 
  
                                            //* target *//   //* source *//
  importerT = Teuchos::rcp(new Tpetra_Import(localNodeMapT, overlapNodeMapT));


  // create exporter (for integration type operations):
                                            //* source *//   //* target *//
  exporterT = Teuchos::rcp(new Tpetra_Export(overlapNodeMapT, localNodeMapT));

  // this should go somewhere else.  for now ...
  GlobalPoint gp;
  int blockcounts[2] = {1,3};
  MPI_Datatype oldtypes[2] = {MPI_INT, MPI_DOUBLE};
  MPI_Aint offsets[3] = {(MPI_Aint)&(gp.gid)    - (MPI_Aint)&gp, 
                         (MPI_Aint)&(gp.coords) - (MPI_Aint)&gp};
  MPI_Type_create_struct(2,blockcounts,offsets,oldtypes,&MPI_GlobalPoint);
  MPI_Type_commit(&MPI_GlobalPoint);

  // initialize/build the filter operators. these are built once.
  int nFilters = filters.size();
  for(int ifltr=0; ifltr<nFilters; ifltr++){
    filters[ifltr]->buildOperator(
      _subProblems[0].app, 
      overlapNodeMapT, localNodeMapT,
      importerT, exporterT);
  }


  auto nVecs = ObjectiveGradientVecT.size();
  // pass subProblems to the objective aggregator
  if( entityType == "State Variable" ){
    _objAggregator->SetInputVariablesT(_subProblems);
    _objAggregator->SetOutputVariablesT(objectiveValue, overlapObjectiveGradientVecT);
  } else 
  if( entityType == "Distributed Parameter" ){
    _objAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
    _objAggregator->SetOutputVariablesT(objectiveValue, ObjectiveGradientVecT);
  }
  _objAggregator->SetCommunicator(comm);
  
  // pass subProblems to the constraint aggregator
  if( !_conAggregator.is_null() ){
    if( entityType == "State Variable" ){
      _conAggregator->SetInputVariablesT(_subProblems);
      _conAggregator->SetOutputVariablesT(constraintValue, overlapConstraintGradientVecT);
    } else 
    if( entityType == "Distributed Parameter" ){
      _conAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
      _conAggregator->SetOutputVariablesT(constraintValue, ConstraintGradientVecT);
    }
    _conAggregator->SetCommunicator(comm);
  }
  


#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif
}


/******************************************************************************/
void
ATO::Solver::zeroSet()
/******************************************************************************/
{
  // set parameters and responses
  _iteration      = 1;

  _is_verbose = false;
  _is_restart = false;

  _derivativeFilter   = Teuchos::null;
  _objAggregator      = Teuchos::null;
  
}

  
/******************************************************************************/
void
ATO::Solver::evalModel(const InArgs& inArgs,
                       const OutArgs& outArgs ) const
/******************************************************************************/
{
  int numHomogenizationSets = _homogenizationSets.size();
  for(int iHomog=0; iHomog<numHomogenizationSets; iHomog++){
    const HomogenizationSet& hs = _homogenizationSets[iHomog];
    int numColumns = hs.homogenizationProblems.size();
    for(int i=0; i<numColumns; i++){

      // enforce PDE constraints
      hs.homogenizationProblems[i].model->evalModel((*hs.homogenizationProblems[i].params_in),
                                                    (*hs.homogenizationProblems[i].responses_out));
    }

    if(numColumns > 0){
      // collect homogenized values
      Kokkos::DynRankView<RealType, PHX::Device> Cvals("ZZZ", numColumns,numColumns);
      for(int i=0; i<numColumns; i++){
        Teuchos::RCP<const Epetra_Vector> g = hs.homogenizationProblems[i].responses_out->get_g(hs.responseIndex);
        for(int j=0; j<numColumns; j++){
          Cvals(i,j) = (*g)[j];
        }
      }
      if(_solverComm->getRank() == 0){
        Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
        *out << "*****************************************" << std::endl;
        *out << " Homogenized parameters (" << hs.name << ") are: " << std::endl; 
        for(int i=0; i<numColumns; i++){
          for(int j=0; j<numColumns; j++){
            *out << std::setprecision(10) << 1.0/2.0*(Cvals(i,j)+Cvals(j,i)) << " ";
          }
          *out << std::endl;
        }
        *out << "*****************************************" << std::endl;
      }

      for(int iPhys=0; iPhys<_numPhysics; iPhys++){
        Albany::StateManager& stateMgr = _subProblems[iPhys].app->getStateMgr();
        Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
        Albany::StateArrayVec& src = stateArrays.elemStateArrays;
        int numWorksets = src.size();

        for(int ws=0; ws<numWorksets; ws++){
          for(int i=0; i<numColumns; i++){
            for(int j=i; j<numColumns; j++){
              std::stringstream valname;
              valname << hs.name << " " << i+1 << j+1;
              Albany::MDArray& wsC = src[ws][valname.str()]; 
              if( wsC.size() != 0 ) wsC(0) = (Cvals(j,i)+Cvals(i,j))/2.0;
            }
          }
        }
      }
    }
  }

  for(int i=0; i<_numPhysics; i++){
    Albany::StateManager& stateMgr = _subProblems[i].app->getStateMgr();
    const Albany::WorksetArray<std::string>::type& 
      wsEBNames = stateMgr.getDiscretization()->getWsEBNames();
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();
    Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
    Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
    int numWorksets = dest.size();
  
    // initialize topology of fixed blocks
    int ntopos = _topologyInfoStructsT.size();

    for(int ws=0; ws<numWorksets; ws++){

      for(int itopo=0; itopo<ntopos; itopo++){
        Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
        Teuchos::RCP<Topology> topology = topoStructT->topologyT;
        const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();

        if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) != fixedBlocks.end() ){

          if( topology->getEntityType() == "State Variable" ){
            double matVal = topology->getMaterialValue();
            Albany::MDArray& wsTopo = dest[ws][topology->getName()];
            int numCells = wsTopo.dimension(0);
            int numNodes = wsTopo.dimension(1);
            for(int cell=0; cell<numCells; cell++)
              for(int node=0; node<numNodes; node++){
                wsTopo(cell,node) = matVal;
              }
          } else if( topology->getEntityType() == "Distributed Parameter" ){
            Teuchos::ArrayRCP<double> ltopo = topoStructT->localVectorT->get1dViewNonConst(); 
            double matVal = topology->getMaterialValue();
            const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& elNodeID = wsElNodeID[ws];
            int numCells = elNodeID.size();
            int numNodes = elNodeID[0].size();
            for(int cell=0; cell<numCells; cell++)
              for(int node=0; node<numNodes; node++){
                int gid = wsElNodeID[ws][cell][node];
                int lid = localNodeMapT->getLocalElement(gid);
                if(lid != -1) ltopo[lid] = matVal;
              }
          }
        }
      }
    }
  }

  if(_is_verbose){
    Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
    *out << "*** Performing Topology Optimization Loop ***" << std::endl;
  }

  _optimizer->Initialize();

  _optimizer->Optimize();
 
}



/******************************************************************************/
///*************** SOLVER - OPTIMIZER INTERFACE FUNCTIONS *******************///
/******************************************************************************/

/******************************************************************************/
void ATO::Solver::getOptDofsUpperBound( Teuchos::Array<double>& b )
/******************************************************************************/
{
  int nLocal = _topologyInfoStructsT[0]->localVectorT->getLocalLength();
  int nTopos = _topologyInfoStructsT.size();
  int nTerms = nTopos*nLocal;
 
  b.resize(nTerms);

  Teuchos::Array<double>::iterator from = b.begin();
  Teuchos::Array<double>::iterator to = from+nLocal;
  for(int itopo=0; itopo<nTopos; itopo++){
    TopologyInfoStructT& topoIS = *_topologyInfoStructsT[itopo];
    Teuchos::Array<double> bounds = topoIS.topologyT->getBounds();
    std::fill(from, to, bounds[1]);
    from += nLocal; to += nLocal;
  }
}

/******************************************************************************/
void ATO::Solver::getOptDofsLowerBound( Teuchos::Array<double>& b )
/******************************************************************************/
{

  int nLocal = _topologyInfoStructsT[0]->localVectorT->getLocalLength();
  int nTopos = _topologyInfoStructsT.size();
  int nTerms = nTopos*nLocal;
 
  b.resize(nTerms);

  Teuchos::Array<double>::iterator from = b.begin();
  Teuchos::Array<double>::iterator to = from+nLocal;
  for(int itopo=0; itopo<nTopos; itopo++){
    TopologyInfoStructT& topoIS = *_topologyInfoStructsT[itopo];
    Teuchos::Array<double> bounds = topoIS.topologyT->getBounds();
    std::fill(from, to, bounds[0]);
    from += nLocal; to += nLocal;
  }
}

/******************************************************************************/
void
ATO::Solver::InitializeOptDofs(double* p)
/******************************************************************************/
{
  if( _is_restart ){
// JR: this needs to be tested for multimaterial
    Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
    copyTopologyFromStateMgr( p, stateMgr );
  } else {
    int ntopos = _topologyInfoStructsT.size();
    for(int itopo=0; itopo<ntopos; itopo++){
      Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
      int numLocalNodes = topoStructT->localVectorT->getLocalLength();
      double initVal = topoStructT->topologyT->getInitialValue();
      int fromIndex=itopo*numLocalNodes;
      int toIndex=fromIndex+numLocalNodes;
      for(int lid=fromIndex; lid<toIndex; lid++)
        p[lid] = initVal;
    }
  }
}

/******************************************************************************/
void
ATO::Solver::ComputeObjective(const double* p, double& g, double* dgdp)
/******************************************************************************/
{

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "IKT, 12/22/16, WARNING: Tpetra-converted ComputeObjective has not been tested " 
       << "yet and may not work correctly! \n"; 
  for(int i=0; i<_numPhysics; i++){

    // copy data from p into each stateManager
    if( entityType == "State Variable" ){
      Albany::StateManager& stateMgr = _subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr );
    } else 
    if( entityType == "Distributed Parameter"){
      copyTopologyIntoParameter( p, _subProblems[i] );
    }

    // enforce PDE constraints
    _subProblems[i].model->evalModel((*_subProblems[i].params_in),
                                    (*_subProblems[i].responses_out));
  }

  if ( entityType == "Distributed Parameter" ) {
    updateTpetraResponseMaps(); 
    _objAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
  }
  _objAggregator->EvaluateT();
  copyObjectiveFromStateMgr( g, dgdp );

  _iteration++;

  
}

/******************************************************************************/
void
ATO::Solver::ComputeObjective(double* p, double& g, double* dgdp)
/******************************************************************************/
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "IKT, 12/22/16, WARNING: Tpetra-converted ComputeObjective has not been tested " 
       << "yet and may not work correctly! \n"; 
  if(_iteration!=1) smoothTopologyT(p);

  for(int i=0; i<_numPhysics; i++){
    

    // copy data from p into each stateManager
    if( entityType == "State Variable" ){
      Albany::StateManager& stateMgr = _subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr );
    } else 
    if( entityType == "Distributed Parameter"){
      copyTopologyIntoParameter( p, _subProblems[i] );
    }

    // enforce PDE constraints
    _subProblems[i].model->evalModel((*_subProblems[i].params_in),
                                    (*_subProblems[i].responses_out));
  }

  if ( entityType == "Distributed Parameter" ) {
    updateTpetraResponseMaps(); 
    _objAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
  }
  _objAggregator->EvaluateT();
  copyObjectiveFromStateMgr( g, dgdp );

  // See if the user specified a new design frequency.
  GO new_frequency = -1;
  if( _solverComm->getRank() == 0){
    std::ifstream file("update_frequency.txt");
    if(file.is_open())
    {
      file >> new_frequency;
    }
  }
  Teuchos::broadcast(*_solverComm, 0, 1, &new_frequency);

  if(new_frequency != -1)
  {
    // the user has specified a new frequency to use
    _writeDesignFrequency = new_frequency;
   }

   // Output a new result file if requested
   if(_writeDesignFrequency && (_iteration % _writeDesignFrequency == 0) )
     writeCurrentDesign();
  _iteration++;
}

/******************************************************************************/
void
ATO::Solver::writeCurrentDesign()
/******************************************************************************/
{
#ifdef ATO_USES_ISOLIB
  Teuchos::RCP<Albany::AbstractDiscretization>
    disc = _subProblems[0].app->getDiscretization();

  Albany::STKDiscretization *stkmesh = dynamic_cast<Albany::STKDiscretization*>(disc.get());
  TEUCHOS_TEST_FOR_EXCEPTION(
    stkmesh == NULL, Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  Attempted to cast non STK mesh." << std::endl);

  MPI_Comm mpi_comm = Albany::getMpiCommFromTeuchosComm(*_solverComm);
  iso::STKExtract ex;
  ex.create_mesh_apis_Albany(&mpi_comm,
             &(stkmesh->getSTKBulkData()),
             &(stkmesh->getSTKMetaData()),
             "", "iso.exo", "Rho_node", 1e-5, 0.5,
              0, 0, 1, 0);
  ex.run_Albany(_iteration);
#else
  TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
         << "Error! Albany must be compiled with IsoLib support for runtime output." << std::endl);
#endif
}

/******************************************************************************/
void
ATO::Solver::copyTopologyIntoParameter( const double* p, SolverSubSolver& subSolver )
/******************************************************************************/
{

  Teuchos::RCP<Albany::Application> app = subSolver.app;
  Albany::StateManager& stateMgr = app->getStateMgr();

  Teuchos::RCP<DistParamLib> distParams = app->getDistParamLib();

  const Albany::WorksetArray<std::string>::type& wsEBNames = stateMgr.getDiscretization()->getWsEBNames();

  int ntopos = _topologyInfoStructsT.size();
  for(int itopo=0; itopo<ntopos; itopo++ ){
    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    Teuchos::RCP<Topology> topology = topoStructT->topologyT;
    const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();

    const std::vector<Albany::IDArray>& 
      wsElDofs = distParams->get(topology->getName())->workset_elem_dofs();

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

    // enforce fixed blocks
    Teuchos::RCP<Tpetra_Vector> topoVecT = topoStructT->localVectorT;
    Teuchos::ArrayRCP<double> ltopo = topoVecT->get1dViewNonConst(); 
    int numMyNodes = topoVecT->getLocalLength();
    for(int i=0; i<numMyNodes; i++) ltopo[i] = p[i];
  
    smoothTopologyT(topoStructT);
  
    int numWorksets = wsElDofs.size();
    double matVal = topology->getMaterialValue();
    for(int ws=0; ws<numWorksets; ws++){
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) != fixedBlocks.end() ) {
        const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& elNodeID = wsElNodeID[ws];
        int numCells = elNodeID.size();
        int numNodes = elNodeID[0].size();
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = localNodeMapT->getLocalElement(gid);
            if(lid != -1) ltopo[lid] = matVal;
          }
      }
    }

    // save topology to nodal data for output sake
    Teuchos::RCP<Albany::NodeFieldContainer> 
      nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();

    Teuchos::RCP<Tpetra_Vector> overlapTopoVecT = topoStructT->overlapVectorT;

    // JR: fix this.  you don't need to do this every time.  Just once at setup, after topoVec is built
    int distParamIndex = subSolver.params_in->Np()-1;
    Teuchos::RCP<Epetra_Comm> comm = 
      Albany::createEpetraCommFromTeuchosComm(localNodeMapT->getComm());
    Teuchos::RCP<Epetra_Map> localNodeMap = Petra::TpetraMap_To_EpetraMap(localNodeMapT, comm); 
    Teuchos::RCP<Epetra_Vector> topoVec = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
    Petra::TpetraVector_To_EpetraVector(topoVecT, *topoVec, comm); 
    subSolver.params_in->set_p(distParamIndex,topoVec);
  
    overlapTopoVecT->doImport(*topoVecT, *importerT, Tpetra::INSERT);
    std::string nodal_topoName = topology->getName()+"_node";
    (*nodeContainer)[nodal_topoName]->saveFieldVector(overlapTopoVecT,/*offset=*/0);
  }

}
/******************************************************************************/
void
ATO::Solver::copyTopologyFromStateMgr(double* p, Albany::StateManager& stateMgr )
/******************************************************************************/
{

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& src = stateArrays.elemStateArrays;
  int numWorksets = src.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  // copy the topology from the state manager
  int ntopos = _topologyInfoStructsT.size();
  for(int itopo=0; itopo<ntopos; itopo++){
    Teuchos::RCP<TopologyInfoStructT> topologyInfoStructT = _topologyInfoStructsT[itopo];
    int numLocalNodes = topologyInfoStructT->localVectorT->getLocalLength();
    Teuchos::RCP<Topology> topology = topologyInfoStructT->topologyT;
    int offset = itopo*numLocalNodes;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = src[ws][topology->getName()+"_node"];
      int numCells = wsTopo.dimension(0), numNodes = wsTopo.dimension(1);
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = localNodeMapT->getLocalElement(gid);
            if(lid >= 0) p[lid+offset] = wsTopo(cell,node);
          }
    }
  }

}
/******************************************************************************/
void
ATO::Solver::smoothTopologyT(double* p)
/******************************************************************************/
{
  // copy topology into Tpetra_Vectors to apply the filter and/or communicate boundary data
  int ntopos = _topologyInfoStructsT.size();
  
  for(int itopo=0; itopo<ntopos; itopo++){
    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    Teuchos::RCP<Tpetra_Vector> topoVecT = topoStructT->localVectorT;
    Teuchos::ArrayRCP<double> ltopo = topoVecT->get1dViewNonConst(); 
    int numLocalNodes = topoVecT->getLocalLength();
    int offset = itopo*numLocalNodes;
    p += offset;
    for(int lid=0; lid<numLocalNodes; lid++)
      ltopo[lid] = p[lid];

    smoothTopologyT(topoStructT);

    // copy the topology back from the tpetra vectors
    for(int lid=0; lid<numLocalNodes; lid++)
      p[lid] = ltopo[lid];
  }
}

/******************************************************************************/
void
ATO::Solver::smoothTopologyT(Teuchos::RCP<TopologyInfoStructT> topoStructT)
/******************************************************************************/
{
  // apply filter if requested
  if(topoStructT->filterT != Teuchos::null){
    Teuchos::RCP<Tpetra_Vector> topoVecT = topoStructT->localVectorT;
    Teuchos::RCP<Tpetra_Vector> filtered_topoVecT = 
        Teuchos::rcp(new Tpetra_Vector(localNodeMapT));
    int num = topoStructT->filterT->getNumIterations();
    for(int i=0; i<num; i++){
      //IKT, 1/5/17: the use of copies here is somewhat hacky, to get apply 
      //to do the right thing and not throw an exception in a debug build. 
      std::vector<double> vec(topoVecT->getLocalLength()); 
      Teuchos::ArrayView<double> view = Teuchos::arrayViewFromVector(vec); 
      if (i == 0) 
        topoVecT->get1dCopy(view);
      else 
        filtered_topoVecT->get1dCopy(view); 
      Teuchos::RCP<Tpetra_Vector> temp = Teuchos::rcp(new Tpetra_Vector(localNodeMapT, view)); 
      topoStructT->filterT->FilterOperatorT()->apply(*temp, *filtered_topoVecT, Teuchos::NO_TRANS);
    }
    *topoVecT = *filtered_topoVecT; 
  } else
  if(topoStructT->postFilterT != Teuchos::null){
    Teuchos::RCP<Tpetra_Vector> topoVecT = topoStructT->localVectorT;
    Teuchos::RCP<Tpetra_Vector> filteredTopoVecT = topoStructT->filteredVectorT;
    Teuchos::RCP<Tpetra_Vector> filteredOTopoVecT = topoStructT->filteredOverlapVectorT;
    topoStructT->postFilterT->FilterOperatorT()->apply(*topoVecT, *filteredTopoVecT, Teuchos::NO_TRANS);
    filteredOTopoVecT->doImport(*filteredTopoVecT, *importerT, Tpetra::INSERT);
  }
}

/******************************************************************************/
void
ATO::Solver::copyTopologyIntoStateMgr( const double* p, Albany::StateManager& stateMgr )
/******************************************************************************/
{

  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;
  int numWorksets = dest.size();

  Teuchos::RCP<Albany::AbstractDiscretization> disc = stateMgr.getDiscretization();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  int ntopos = _topologyInfoStructsT.size();
  for(int itopo=0; itopo<ntopos; itopo++){
    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    Teuchos::RCP<Topology> topology = topoStructT->topologyT;
    const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();

    // copy topology into Epetra_Vector to apply the filter and/or communicate boundary data
    Teuchos::RCP<Tpetra_Vector> topoVecT = topoStructT->localVectorT;
    Teuchos::ArrayRCP<double> ltopo = topoVecT->get1dViewNonConst(); 
    int numLocalNodes = topoVecT->getLocalLength();
    int offset = itopo*numLocalNodes;
    for(int lid=0; lid<numLocalNodes; lid++)
      ltopo[lid] = p[lid+offset];

    smoothTopologyT(topoStructT);

    Teuchos::RCP<Tpetra_Vector> overlapTopoVecT = topoStructT->overlapVectorT;
    overlapTopoVecT->doImport(*topoVecT, *importerT, Tpetra::INSERT);

    // copy the topology into the state manager
    Teuchos::ArrayRCP<double> otopo = overlapTopoVecT->get1dViewNonConst(); 
    double matVal = topology->getMaterialValue();
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      int numCells = wsTopo.dimension(0), numNodes = wsTopo.dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMapT->getLocalElement(gid);
          wsTopo(cell,node) = otopo[lid];
        }
    }

    Tpetra_Vector overlapFreeNodeMaskT(overlapNodeMapT);
    Tpetra_Vector localFreeNodeMaskT(localNodeMapT);
    overlapFreeNodeMaskT.putScalar(0.0);
    Teuchos::ArrayRCP<double> fMask = overlapFreeNodeMaskT.get1dViewNonConst(); 
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      int numCells = wsTopo.dimension(0), numNodes = wsTopo.dimension(1);
      if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) == fixedBlocks.end() ){
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = overlapNodeMapT->getLocalElement(gid);
            fMask[lid] = 1.0;
          }
      } else {
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
            wsTopo(cell,node) = matVal;
          }
      }
    }
    localFreeNodeMaskT.putScalar(1.0);
    localFreeNodeMaskT.doExport(overlapFreeNodeMaskT, *exporterT, Tpetra::ABSMAX);
    overlapFreeNodeMaskT.doImport(localFreeNodeMaskT, *importerT, Tpetra::INSERT);
  
    // if it is a fixed block, set the topology variable to the material value
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      int numCells = wsTopo.dimension(0), numNodes = wsTopo.dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMapT->getLocalElement(gid);
          if(fMask[lid] != 1.0) otopo[lid] = matVal;
        }
    }

    // save topology to nodal data for output sake
    Teuchos::RCP<Albany::NodeFieldContainer> 
      nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();

    std::string nodal_topoName = topology->getName()+"_node";
    (*nodeContainer)[nodal_topoName]->saveFieldVector(overlapTopoVecT,/*offset=*/0);

    if(topoStructT->postFilterT != Teuchos::null){
      nodal_topoName = topology->getName()+"_node_filtered";
      Teuchos::RCP<Tpetra_Vector> filteredOTopoVecT = topoStructT->filteredOverlapVectorT;
      (*nodeContainer)[nodal_topoName]->saveFieldVector(filteredOTopoVecT,/*offset=*/0);
    }
  }
}

/******************************************************************************/
void
ATO::Solver::copyConstraintFromStateMgr( double& c, double* dcdp )
/******************************************************************************/
{

  c = *constraintValue;

  auto nVecs = ConstraintGradientVecT.size();
  for(int ivec=0; ivec<nVecs; ivec++){

    if( entityType == "State Variable" ) {
      ConstraintGradientVecT[ivec]->putScalar(0.0);
      ConstraintGradientVecT[ivec]->doExport(*overlapConstraintGradientVecT[ivec], *exporterT, Tpetra::ADD);
    }

    if( dcdp != NULL ){
      auto numLocalNodes = ConstraintGradientVecT[ivec]->getLocalLength();
      Teuchos::ArrayRCP<const double> lvec = ConstraintGradientVecT[ivec]->get1dView(); 
      std::memcpy((void*)(dcdp+ivec*numLocalNodes), lvec.getRawPtr(), numLocalNodes*sizeof(double));
    }
  }
}

/******************************************************************************/
void
ATO::Solver::copyObjectiveFromStateMgr( double& g, double* dgdp )
/******************************************************************************/
{

  // aggregated objective derivative is stored in the first subproblem
  Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();

  g = *objectiveValue;

  auto nVecs = ObjectiveGradientVecT.size();
  for(int ivec=0; ivec<nVecs; ivec++){

    if( entityType == "State Variable" ) {
      ObjectiveGradientVecT[ivec]->putScalar(0.0);
      ObjectiveGradientVecT[ivec]->doExport(*overlapObjectiveGradientVecT[ivec], *exporterT, Tpetra::ADD);
    }

    auto numLocalNodes = ObjectiveGradientVecT[ivec]->getLocalLength();
    Teuchos::ArrayRCP<const double> lvec = ObjectiveGradientVecT[ivec]->get1dView(); 
    // apply filter if requested
    Teuchos::RCP<Tpetra_Vector> filtered_ObjectiveGradientVecT = 
        Teuchos::rcp(new Tpetra_Vector(*ObjectiveGradientVecT[ivec]));
    if(_derivativeFilter != Teuchos::null){
      int num = _derivativeFilter->getNumIterations();
      for(int i=0; i<num; i++){
        //IKT, FIXME:
        //Tpetra::CrsMatrix apply method with Teuchos::TRANS mode does not work correctly - 
        //gives a vector of all 0s here.  Waiting for Mark Hoemmen to fix this.  Once he 
	//does, uncomment following code, and get rid of FilterOperatorTransposeT.  
        //_derivativeFilter->FilterOperatorT()->apply(*ObjectiveGradientVecT[ivec],
        //                                            *filtered_ObjectiveGradientVecT,
        //                                            Teuchos::TRANS);
	//IKT, 1/23/17: the use of copies here is somewhat hacky, to get apply 
	//to do the right thing and not throw an exception in a debug build. 
	std::vector<double> vec(ObjectiveGradientVecT[ivec]->getLocalLength());
	Teuchos::ArrayView<double> view = Teuchos::arrayViewFromVector(vec);
	if (i == 0)
	  ObjectiveGradientVecT[ivec]->get1dCopy(view);
        else
	  filtered_ObjectiveGradientVecT->get1dCopy(view);
        Teuchos::RCP<Tpetra_Vector> temp = Teuchos::rcp(new Tpetra_Vector(localNodeMapT, view));
        _derivativeFilter->FilterOperatorTransposeT()->apply(*temp,
                                                             *filtered_ObjectiveGradientVecT,
                                                             Teuchos::NO_TRANS); 
        *ObjectiveGradientVecT[ivec] = *filtered_ObjectiveGradientVecT; 
        //Tpetra_MatrixMarket_Writer::writeDenseFile("filtered_ObjectiveGradientVecT.mm",  
        //    filtered_ObjectiveGradientVecT); exit(1); 
      }
      Teuchos::ArrayRCP<const double> lvec = filtered_ObjectiveGradientVecT->get1dView(); 
      for (int i=0; i< numLocalNodes; i++) 
      std::memcpy((void*)(dgdp+ivec*numLocalNodes), lvec.getRawPtr(), numLocalNodes*sizeof(double));
    } else {
      for (int i=0; i< numLocalNodes; i++) 
      std::memcpy((void*)(dgdp+ivec*numLocalNodes), lvec.getRawPtr(), numLocalNodes*sizeof(double));
    }

    // save dgdp to nodal data for output sake
    overlapObjectiveGradientVecT[ivec]->doImport(*filtered_ObjectiveGradientVecT, *importerT, Tpetra::INSERT);
    Teuchos::RCP<Albany::NodeFieldContainer>
      nodeContainer = stateMgr.getNodalDataBase()->getNodeContainer();
    std::string nodal_derName = Albany::strint(_objAggregator->getOutputDerivativeName()+"_node", ivec);
    (*nodeContainer)[nodal_derName]->saveFieldVector(overlapObjectiveGradientVecT[ivec],/*offset=*/0);
  }
}
/******************************************************************************/
void
ATO::Solver::ComputeMeasure(std::string measureType, double& measure)
/******************************************************************************/
{
  return _atoProblem->ComputeMeasure(measureType, measure);
}

/******************************************************************************/
void
ATO::OptInterface::ComputeMeasure(std::string measureType, const double* p, 
                                  double& measure)
/******************************************************************************/
{
  ComputeMeasure(measureType, p, measure, NULL, "Gauss Quadrature");
}

/******************************************************************************/
void
ATO::OptInterface::ComputeMeasure(std::string measureType, const double* p, 
                                  double& measure, double* dmdp)
/******************************************************************************/
{
  ComputeMeasure(measureType, p, measure, dmdp, "Gauss Quadrature");
}

/******************************************************************************/
void
ATO::OptInterface::ComputeMeasure(std::string measureType, const double* p, 
                                  double& measure, std::string integrationMethod)
/******************************************************************************/
{
  ComputeMeasure(measureType, p, measure, NULL, integrationMethod);
}

/******************************************************************************/
void
ATO::Solver::ComputeMeasure(std::string measureType, const double* p, 
                            double& measure, double* dmdp, 
                            std::string integrationMethod)
/******************************************************************************/
{
  // communicate boundary topo data
  Albany::StateManager& stateMgr = _subProblems[0].app->getStateMgr();
  
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = stateMgr.getDiscretization()->getWsElNodeID();

  int numWorksets = wsElNodeID.size();

  int ntopos = _topologyInfoStructsT.size();

  std::vector<Teuchos::RCP<TopologyStructT> > topologyStructsT(ntopos);

  for(int itopo=0; itopo<ntopos; itopo++){

    topologyStructsT[itopo] = Teuchos::rcp(new TopologyStructT);
  
    Teuchos::RCP<Tpetra_Vector> topoVecT = _topologyInfoStructsT[itopo]->localVectorT;
    int numLocalNodes = topoVecT->getLocalLength();
    int offset = itopo*numLocalNodes;
    Teuchos::ArrayRCP<double> ltopoT = topoVecT->get1dViewNonConst(); 
    for(int ws=0; ws<numWorksets; ws++){
      int numCells = wsElNodeID[ws].size();
      int numNodes = wsElNodeID[ws][0].size();
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = localNodeMapT->getLocalElement(gid);
          if(lid != -1) ltopoT[lid] = p[lid+offset];
        }
    }

    Teuchos::RCP<TopologyInfoStructT> topoStructT = _topologyInfoStructsT[itopo];
    smoothTopologyT(topoStructT);

    Teuchos::RCP<Tpetra_Vector> overlapTopoVecT = _topologyInfoStructsT[itopo]->overlapVectorT;
    overlapTopoVecT->doImport(*topoVecT, *importerT, Tpetra::INSERT);

    topologyStructsT[itopo]->topologyT = _topologyInfoStructsT[itopo]->topologyT; 
    topologyStructsT[itopo]->dataVectorT = overlapTopoVecT;
  }

  return _atoProblem->ComputeMeasureT(measureType, topologyStructsT, 
                                     measure, dmdp, integrationMethod);
}


/******************************************************************************/
void
ATO::Solver::ComputeVolume(double* p, const double* dfdp, 
                           double& v, double threshhold, double minP)
/******************************************************************************/
{
  /*  Assumptions:
      -- dfdp is already consistent across proc boundaries.
      -- the volume computation that's done by the atoProblem updates the topology, p.
      -- Since dfdp is 'boundary consistent', the resulting topology, p, is also
         'boundary consistent', so no communication is necessary.
  */
  return _atoProblem->ComputeVolume(p, dfdp, v, threshhold, minP);
}

/******************************************************************************/
void
ATO::Solver::Compute(double* p, double& g, double* dgdp, double& c, double* dcdp)
/******************************************************************************/
{
  Compute((const double*)p, g, dgdp, c, dcdp);
}

/******************************************************************************/
void
ATO::Solver::Compute(const double* p, double& g, double* dgdp, double& c, double* dcdp)
/******************************************************************************/
{
  for(int i=0; i<_numPhysics; i++){

    // copy data from p into each stateManager
    if( entityType == "State Variable" ){
      Albany::StateManager& stateMgr = _subProblems[i].app->getStateMgr();
      copyTopologyIntoStateMgr( p, stateMgr );
    } else 
    if( entityType == "Distributed Parameter"){
      copyTopologyIntoParameter( p, _subProblems[i] );
    }

    // enforce PDE constraints
    _subProblems[i].model->evalModel((*_subProblems[i].params_in),
                                    (*_subProblems[i].responses_out));
  }

  if ( entityType == "Distributed Parameter" ) {
    updateTpetraResponseMaps(); 
    _objAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
  }
  _objAggregator->EvaluateT();
  copyObjectiveFromStateMgr( g, dgdp );
  
  if( !_conAggregator.is_null()){
    if ( entityType == "Distributed Parameter" ) {
      updateTpetraResponseMaps(); 
      _conAggregator->SetInputVariablesT(_subProblems, responseMapT, responseDerivMapT);
    }
    _conAggregator->EvaluateT();
    copyConstraintFromStateMgr( c, dcdp );
  } else c = 0.0;

  _iteration++;

}


/******************************************************************************/
void
ATO::Solver::ComputeConstraint(double* p, double& c, double* dcdp)
/******************************************************************************/
{
}

/******************************************************************************/
int
ATO::Solver::GetNumOptDofs()
/******************************************************************************/
{
//  return _subProblems[0].app->getDiscretization()->getNodeMap()->NumMyElements();
  auto nVecs = ObjectiveGradientVecT.size();
  return nVecs*ObjectiveGradientVecT[0]->getLocalLength();
}

/******************************************************************************/
///*********************** SETUP AND UTILITY FUNCTIONS **********************///
/******************************************************************************/


/******************************************************************************/
ATO::SolverSubSolver
ATO::Solver::CreateSubSolver( const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                              const Teuchos::RCP<const Teuchos_Comm>& commT,
                              const Teuchos::RCP<const Tpetra_Vector>& initial_guess)
/******************************************************************************/
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  ATO::SolverSubSolver ret; //value to return

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "ATO Solver creating solver from " << appParams->name()
       << " parameter list" << std::endl;

  //! Create solver and application objects via solver factory
  {

    //! Create solver factory, which reads xml input filen
    Albany::SolverFactory slvrfctry(appParams, commT);

    ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, commT, commT, initial_guess);
  }


  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  int numParameters = 0;
  if( problemParams.isType<Teuchos::ParameterList>("Parameters") )
    numParameters = problemParams.sublist("Parameters").get<int>("Number of Parameter Vectors");

  int numResponses = 0;
  if( problemParams.isType<Teuchos::ParameterList>("Response Functions") )
    numResponses = problemParams.sublist("Response Functions").get<int>("Number of Response Vectors");

  ret.params_in = rcp(new EpetraExt::ModelEvaluator::InArgs);
  ret.responses_out = rcp(new EpetraExt::ModelEvaluator::OutArgs);

  *(ret.params_in) = ret.model->createInArgs();
  *(ret.responses_out) = ret.model->createOutArgs();

  // the createOutArgs() function doesn't allocate storage
  RCP<Epetra_Vector> g1;
  int ss_num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  for(int ig=0; ig<ss_num_g; ig++){
    g1 = rcp(new Epetra_Vector(*(ret.model->get_g_map(ig))));
    ret.responses_out->set_g(ig,g1);
  }

  RCP<Epetra_Vector> p1;
  int ss_num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  TEUCHOS_TEST_FOR_EXCEPTION (
    ss_num_p - numParameters > 1,
    Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Cannot have more than one distributed Parameter for topology optimization" << std::endl);
  for(int ip=0; ip<ss_num_p; ip++){
    p1 = rcp(new Epetra_Vector(*(ret.model->get_p_init(ip))));
    ret.params_in->set_p(ip,p1);
  }

  for(int ig=0; ig<numResponses; ig++){
    if(ss_num_p > numParameters){
      int ip = ss_num_p-1;
      Teuchos::ParameterList& resParams = 
        problemParams.sublist("Response Functions").sublist(Albany::strint("Response Vector",ig));
      std::string gName = resParams.get<std::string>("Response Name");
      std::string dgdpName = resParams.get<std::string>("Response Derivative Name");
      if(!ret.responses_out->supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, ig, ip).none()){
        RCP<const Epetra_Vector> p = ret.params_in->get_p(ip);
        RCP<const Epetra_Vector> g = ret.responses_out->get_g(ig);
        RCP<Epetra_MultiVector> dgdp = rcp(new Epetra_MultiVector(p->Map(), g->GlobalLength() ));
        if(ret.responses_out->supports(OUT_ARG_DgDp,ig,ip).supports(DERIV_TRANS_MV_BY_ROW)){
          Derivative dgdp_out(dgdp, DERIV_TRANS_MV_BY_ROW);
          ret.responses_out->set_DgDp(ig,ip,dgdp_out);
        } else 
          ret.responses_out->set_DgDp(ig,ip,dgdp);
        responseMap.insert(std::pair<std::string,RCP<const Epetra_Vector> >(gName,g));
        RCP<const Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorConst(*g, _solverComm); 
        responseMapT.insert(std::pair<std::string,RCP<const Tpetra_Vector> >(gName, gT));
        responseDerivMap.insert(std::pair<std::string,RCP<Epetra_MultiVector> >(dgdpName,dgdp));
        RCP<Tpetra_MultiVector> dgdpT = Petra::EpetraMultiVector_To_TpetraMultiVector(*dgdp, _solverComm); 
        responseDerivMapT.insert(std::pair<std::string,RCP<Tpetra_MultiVector> >(dgdpName, dgdpT));
      }
    }
  }

  RCP<Epetra_Vector> xfinal =
    rcp(new Epetra_Vector(*(ret.model->get_g_map(ss_num_g-1)),true) );
  ret.responses_out->set_g(ss_num_g-1,xfinal);

  return ret;
}

/******************************************************************************/
void 
ATO::Solver::updateTpetraResponseMaps()
/******************************************************************************/
{
  std::map<std::string, Teuchos::RCP<const Epetra_Vector> >::const_iterator git;
  git = responseMap.cbegin();  
  for (int i = 0; i<responseMap.size(); i++) {
    std::string gName = git->first;
    Teuchos::RCP<const Tpetra_Vector> gT = Petra::EpetraVector_To_TpetraVectorConst(*(git->second), _solverComm); 
    responseMapT[gName] = gT;
    git++; 
  }
  std::map<std::string, Teuchos::RCP<Epetra_MultiVector> >::const_iterator git2;
  git2 = responseDerivMap.cbegin();  
  for (int i = 0; i<responseDerivMap.size(); i++) {
    std::string gName = git2->first; 
    Teuchos::RCP<Tpetra_MultiVector> dgdpT = Petra::EpetraMultiVector_To_TpetraMultiVector(*(git2->second), _solverComm); 
    responseDerivMapT[gName] = dgdpT;
    git2++; 
  }
}



/******************************************************************************/
Teuchos::RCP<Teuchos::ParameterList> 
ATO::Solver::createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const
/******************************************************************************/
{   


  ///*** CREATE INPUT FILE FOR SUBPROBLEM: ***///
  

  // Get physics (pde) problem sublist, i.e., Physics Problem N, where N = physIndex.
  std::stringstream physStream;
  physStream << "Physics Problem " << physIndex;
  Teuchos::ParameterList& physics_subList = appParams->sublist("Problem").sublist(physStream.str(), false);

  // Create input parameter list for physics app which mimics a separate input file
  std::stringstream appStream;
  appStream << "Parameters for Subapplication " << physIndex;
  Teuchos::RCP<Teuchos::ParameterList> physics_appParams = Teuchos::createParameterList(appStream.str());

  // get reference to Problem ParameterList in new input file and initialize it 
  // from Parameters in Physics Problem N.
  Teuchos::ParameterList& physics_probParams = physics_appParams->sublist("Problem",false);
  physics_probParams.setParameters(physics_subList);

  // Add topology information
  physics_probParams.set<Teuchos::RCP<TopologyArray> >("Topologies",_topologyArrayT);

  Teuchos::ParameterList& topoParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Topologies");
  physics_probParams.set<Teuchos::ParameterList>("Topologies Parameters",topoParams);

  // Check topology.  If the topology is a distributed parameter, then 1) check for existing 
  // "Distributed Parameter" list and error out if found, and 2) add a "Distributed Parameter" 
  // list to the input file, 
  if( entityType == "Distributed Parameter" ){
    TEUCHOS_TEST_FOR_EXCEPTION (
      physics_subList.isType<Teuchos::ParameterList>("Distributed Parameters"),
      Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error! Cannot have 'Distributed Parameters' in both Topology and subproblems" << std::endl);
    Teuchos::ParameterList distParams;
    int ntopos = _topologyInfoStructsT.size();
    distParams.set("Number of Parameter Vectors",ntopos);
    for(int itopo=0; itopo<ntopos; itopo++){
      distParams.set(Albany::strint("Parameter",itopo), _topologyInfoStructsT[itopo]->topologyT->getName());
    }
    physics_probParams.set<Teuchos::ParameterList>("Distributed Parameters", distParams);
  }

  // Add aggregator information
  Teuchos::ParameterList& aggParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Objective Aggregator");
  physics_probParams.set<Teuchos::ParameterList>("Objective Aggregator",aggParams);

  // Add configuration information
  Teuchos::ParameterList& conParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Configuration");
  physics_probParams.set<Teuchos::ParameterList>("Configuration",conParams);

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& physics_discList = physics_appParams->sublist("Discretization", false);
  physics_discList.setParameters(discList);
  // find the output file name and append "Physics_n_" to it. This only checks for exodus output.
  if( physics_discList.isType<std::string>("Exodus Output File Name") ){
    std::stringstream newname;
    newname << "physics_" << physIndex << "_" 
            << physics_discList.get<std::string>("Exodus Output File Name");
    physics_discList.set("Exodus Output File Name",newname.str());
  }

  int ntopos = _topologyInfoStructsT.size();
  for(int itopo=0; itopo<ntopos; itopo++){
    if( _topologyInfoStructsT[itopo]->topologyT->getFixedBlocks().size() > 0 ){
      physics_discList.set("Separate Evaluators by Element Block", true);
      break;
    }
  }

  if( _writeDesignFrequency != 0 )
    physics_discList.set("Use Automatic Aura", true);

  // Piro sublist processing
  physics_appParams->set("Piro",appParams->sublist("Piro"));



  ///*** VERIFY SUBPROBLEM: ***///


  // extract physics and dimension of the subproblem
  Teuchos::ParameterList& subProblemParams = appParams->sublist("Problem").sublist(physStream.str());
  std::string problemName = subProblemParams.get<std::string>("Name");
  // "xD" where x = 1, 2, or 3
  std::string problemDimStr = problemName.substr( problemName.length()-2 );
  //remove " xD" where x = 1, 2, or 3
  std::string problemNameBase = problemName.substr( 0, problemName.length()-3 );
  
  //// check dimensions
  int numDimensions = 0;
  if(problemDimStr == "1D") numDimensions = 1;
  else if(problemDimStr == "2D") numDimensions = 2;
  else if(problemDimStr == "3D") numDimensions = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (
         true, Teuchos::Exceptions::InvalidParameter, std::endl 
         << "Error!  Cannot extract dimension from problem name: " << problemName << std::endl);
  TEUCHOS_TEST_FOR_EXCEPTION (
    numDimensions == 1, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Topology optimization is not avaliable in 1D." << std::endl);

  
  return physics_appParams;

}

/******************************************************************************/
Teuchos::RCP<Teuchos::ParameterList> 
ATO::Solver::createHomogenizationInputFile( 
    const Teuchos::RCP<Teuchos::ParameterList>& appParams, 
    const Teuchos::ParameterList& homog_subList, 
    int homogProblemIndex, 
    int homogSubIndex, 
    int homogDim) const
/******************************************************************************/
{   

  const Teuchos::ParameterList& homog_problem_subList = 
    homog_subList.sublist("Problem");

  // Create input parameter list for app which mimics a separate input file
  std::stringstream appStream;
  appStream << "Parameters for Homogenization Subapplication " << homogSubIndex;
  Teuchos::RCP<Teuchos::ParameterList> homog_appParams = Teuchos::createParameterList(appStream.str());

  // get reference to Problem ParameterList in new input file and initialize it 
  // from Parameters in homogenization base problem
  Teuchos::ParameterList& homog_probParams = homog_appParams->sublist("Problem",false);
  homog_probParams.setParameters(homog_problem_subList);

  // set up BCs (this is a pretty bad klugde till periodic BC's are available)
  Teuchos::ParameterList& Params = homog_probParams.sublist("Dirichlet BCs",false);

  homog_probParams.set("Add Cell Problem Forcing",homogSubIndex);

  const Teuchos::ParameterList& bcIdParams = homog_subList.sublist("Cell BCs");
  Teuchos::Array<std::string> dofs = bcIdParams.get<Teuchos::Array<std::string> >("DOF Names");
  std::string dofsType = bcIdParams.get<std::string>("DOF Type");
  bool isVector;
  if( dofsType == "Scalar" ){
    isVector = false;
    TEUCHOS_TEST_FOR_EXCEPTION(dofs.size() != 1, Teuchos::Exceptions::InvalidParameter, 
                               std::endl << "Error: Expected DOF Names array to be length 1." << std::endl);
  } else
  if( dofsType == "Vector" ){
    isVector = true;
    TEUCHOS_TEST_FOR_EXCEPTION(dofs.size() != homogDim, Teuchos::Exceptions::InvalidParameter, 
                               std::endl << "Error: Expected DOF Names array to be length " << homogDim << "." << std::endl);
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                     std::endl << "Homogenization DOFs must be of type Scalar or Vector (not " << dofsType << ")." << std::endl);

  if(homogDim == 1){
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
    std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
  } else 
  if(homogDim == 2){
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    int negY = bcIdParams.get<int>("Negative Y Face");
    int posY = bcIdParams.get<int>("Positive Y Face");
    if( homogSubIndex < 2 ){
      std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
      std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      if( isVector ){
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[1]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[1]; Params.set(namePosY.str(),0.0);
      } else {
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
      }
    } else {
      std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
      std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
      if( isVector ){
        std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[1]; Params.set(nameNegX.str(),0.0);
        std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[1]; Params.set(namePosX.str(),0.0);
      } else {
        std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
        std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      }
    }
  } else 
  if(homogDim == 3){
    int negX = bcIdParams.get<int>("Negative X Face");
    int posX = bcIdParams.get<int>("Positive X Face");
    int negY = bcIdParams.get<int>("Negative Y Face");
    int posY = bcIdParams.get<int>("Positive Y Face");
    int negZ = bcIdParams.get<int>("Negative Z Face");
    int posZ = bcIdParams.get<int>("Positive Z Face");
    if( homogSubIndex < 3 ){
      std::stringstream nameNegX; nameNegX << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegX.str(),0.0);
      std::stringstream namePosX; namePosX << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosX.str(),0.0);
      if( isVector ){
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[1]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[1]; Params.set(namePosY.str(),0.0);
        std::stringstream nameNegZ; nameNegZ << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[2]; Params.set(nameNegZ.str(),0.0);
        std::stringstream namePosZ; namePosZ << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[2]; Params.set(namePosZ.str(),0.0);
      } else {
        std::stringstream nameNegY; nameNegY << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegY.str(),0.0);
        std::stringstream namePosY; namePosY << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosY.str(),0.0);
        std::stringstream nameNegZ; nameNegZ << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[0]; Params.set(nameNegZ.str(),0.0);
        std::stringstream namePosZ; namePosZ << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[0]; Params.set(namePosZ.str(),0.0);
      }
    } else {
      std::stringstream nameNegYX; nameNegYX << "DBC on NS nodelist_" << negY << " for DOF " << dofs[0]; Params.set(nameNegYX.str(),0.0);
      std::stringstream namePosYX; namePosYX << "DBC on NS nodelist_" << posY << " for DOF " << dofs[0]; Params.set(namePosYX.str(),0.0);
      std::stringstream nameNegZX; nameNegZX << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[0]; Params.set(nameNegZX.str(),0.0);
      std::stringstream namePosZX; namePosZX << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[0]; Params.set(namePosZX.str(),0.0);
      if( isVector ){
        std::stringstream nameNegXY; nameNegXY << "DBC on NS nodelist_" << negX << " for DOF " << dofs[1]; Params.set(nameNegXY.str(),0.0);
        std::stringstream namePosXY; namePosXY << "DBC on NS nodelist_" << posX << " for DOF " << dofs[1]; Params.set(namePosXY.str(),0.0);
        std::stringstream nameNegZY; nameNegZY << "DBC on NS nodelist_" << negZ << " for DOF " << dofs[1]; Params.set(nameNegZY.str(),0.0);
        std::stringstream namePosZY; namePosZY << "DBC on NS nodelist_" << posZ << " for DOF " << dofs[1]; Params.set(namePosZY.str(),0.0);
        std::stringstream nameNegXZ; nameNegXZ << "DBC on NS nodelist_" << negX << " for DOF " << dofs[2]; Params.set(nameNegXZ.str(),0.0);
        std::stringstream namePosXZ; namePosXZ << "DBC on NS nodelist_" << posX << " for DOF " << dofs[2]; Params.set(namePosXZ.str(),0.0);
        std::stringstream nameNegYZ; nameNegYZ << "DBC on NS nodelist_" << negY << " for DOF " << dofs[2]; Params.set(nameNegYZ.str(),0.0);
        std::stringstream namePosYZ; namePosYZ << "DBC on NS nodelist_" << posY << " for DOF " << dofs[2]; Params.set(namePosYZ.str(),0.0);
      } else {
        std::stringstream nameNegXY; nameNegXY << "DBC on NS nodelist_" << negX << " for DOF " << dofs[0]; Params.set(nameNegXY.str(),0.0);
        std::stringstream namePosXY; namePosXY << "DBC on NS nodelist_" << posX << " for DOF " << dofs[0]; Params.set(namePosXY.str(),0.0);
      }
    }
  }

  

  // Discretization sublist processing
  const Teuchos::ParameterList& discList = homog_subList.sublist("Discretization");
  Teuchos::ParameterList& homog_discList = homog_appParams->sublist("Discretization", false);
  homog_discList.setParameters(discList);
  // find the output file name and append "homog_n_" to it. This only checks for exodus output.
  if( homog_discList.isType<std::string>("Exodus Output File Name") ){
    std::stringstream newname;
    newname << "homog_" << homogProblemIndex << "_" << homogSubIndex << "_" 
            << homog_discList.get<std::string>("Exodus Output File Name");
    homog_discList.set("Exodus Output File Name",newname.str());
  }

  // Piro sublist processing
  homog_appParams->set("Piro",appParams->sublist("Piro"));

  
  
  return homog_appParams;
}

/******************************************************************************/
Teuchos::RCP<const Teuchos::ParameterList>
ATO::Solver::getValidProblemParameters() const
/******************************************************************************/
{

  Teuchos::RCP<Teuchos::ParameterList> validPL = 
    Teuchos::createParameterList("ValidTopologicalOptimizationProblemParams");

  // Basic set-up
  validPL->set<int>("Number of Subproblems", 1, "Number of PDE constraint problems");
  validPL->set<int>("Number of Homogenization Problems", 0, "Number of homogenization problems");
  validPL->set<bool>("Verbose Output", false, "Enable detailed output mode");
  validPL->set<int>("Design Output Frequency", 0, "Write isosurface every N iterations");
  validPL->set<std::string>("Name", "", "String to designate Problem");

  // Specify physics problem(s)
  for(int i=0; i<_numPhysics; i++){
    std::stringstream physStream; physStream << "Physics Problem " << i;
    validPL->sublist(physStream.str(), false, "");
  }
  
  int numHomogProblems = _homogenizationSets.size();
  for(int i=0; i<numHomogProblems; i++){
    std::stringstream homogStream; homogStream << "Homogenization Problem " << i;
    validPL->sublist(homogStream.str(), false, "");
  }

  validPL->sublist("Objective Aggregator", false, "");

  validPL->sublist("Constraint Aggregator", false, "");

  validPL->sublist("Topological Optimization", false, "");

  validPL->sublist("Topologies", false, "");

  validPL->sublist("Configuration", false, "");

  validPL->sublist("Spatial Filters", false, "");

  // Physics solver options
  validPL->set<std::string>(
       "Piro Defaults Filename", "", 
       "An xml file containing a default Piro parameterlist and its sublists");

  // Candidate for deprecation.
  validPL->set<std::string>(
       "Solution Method", "Steady", 
       "Flag for Steady, Transient, or Continuation");

  return validPL;
}





/******************************************************************************/
///*************                   BOILERPLATE                  *************///
/******************************************************************************/



/******************************************************************************/
ATO::Solver::~Solver() { }
/******************************************************************************/


/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_x_map() const
/******************************************************************************/
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_f_map() const
/******************************************************************************/
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::InArgs 
ATO::Solver::createInArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription("ATO Solver Model Evaluator Description");
  inArgs.set_Np(c_num_parameters);
  return inArgs;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::OutArgs 
ATO::Solver::createOutArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("ATO Solver Multipurpose Model Evaluator");
  outArgs.set_Np_Ng(c_num_parameters, c_num_responses);
  return outArgs;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_g_map(int j) const
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(j != 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl << "Error in ATO::Solver::get_g_map():  " <<
                     "Invalid response index j = " << j << std::endl);

  return _epetra_x_map;
}

/******************************************************************************/
ATO::SolverSubSolverData
ATO::Solver::CreateSubSolverData(const ATO::SolverSubSolver& sub) const
/******************************************************************************/
{
  ATO::SolverSubSolverData ret;
  if( sub.params_in->Np() > 0 && sub.responses_out->Ng() > 0 ) {
    ret.deriv_support = sub.model->createOutArgs().supports(OUT_ARG_DgDp, 0, 0);
  }
  else ret.deriv_support = EpetraExt::ModelEvaluator::DerivativeSupport();

  ret.Np = sub.params_in->Np();
  ret.pLength = std::vector<int>(ret.Np);
  for(int i=0; i<ret.Np; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_p = sub.params_in->get_p(i);
    //uses local length (need to modify to work with distributed params)
    if(solver_p != Teuchos::null) ret.pLength[i] = solver_p->MyLength();
    else ret.pLength[i] = 0;
  }

  ret.Ng = sub.responses_out->Ng();
  ret.gLength = std::vector<int>(ret.Ng);
  for(int i=0; i<ret.Ng; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_g = sub.responses_out->get_g(i);
    //uses local length (need to modify to work with distributed responses)
    if(solver_g != Teuchos::null) ret.gLength[i] = solver_g->MyLength();
    else ret.gLength[i] = 0;
  }

  if(ret.Np > 0) {
    Teuchos::RCP<const Epetra_Vector> p_init =
      //only first p vector used - in the future could make ret.p_init an array of Np vectors
      sub.model->get_p_init(0);
    if(p_init != Teuchos::null) ret.p_init = Teuchos::rcp(new const Epetra_Vector(*p_init)); //copy
    else ret.p_init = Teuchos::null;
  }
  else ret.p_init = Teuchos::null;

  return ret;
}

/******************************************************************************/
void
ATO::SpatialFilter::buildOperator(
             Teuchos::RCP<Albany::Application> app,
             Teuchos::RCP<const Tpetra_Map>    overlapNodeMapT,
             Teuchos::RCP<const Tpetra_Map>    localNodeMapT,
             Teuchos::RCP<Tpetra_Import>       importerT,
             Teuchos::RCP<Tpetra_Export>       exporterT)
/******************************************************************************/
{

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
          wsElNodeID = app->getDiscretization()->getWsElNodeID();
  
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
      coords = app->getDiscretization()->getCoords();

    const Albany::WorksetArray<std::string>::type& 
      wsEBNames = app->getDiscretization()->getWsEBNames();

    // if this filter operates on a subset of the blocks in the mesh, create a list
    // of nodes that are not smoothed:
    std::set<int> excludeNodes;
    if( blocks.size() > 0 ){
      size_t num_worksets = coords.size();
      // add to the excludeNodes set all nodes that are not to be smoothed
      for (size_t ws=0; ws<num_worksets; ws++) {
        if( find(blocks.begin(), blocks.end(), wsEBNames[ws]) != blocks.end() ) continue;
        int num_cells = coords[ws].size();
        for (int cell=0; cell<num_cells; cell++) {
          size_t num_nodes = coords[ws][cell].size();
          for (int node=0; node<num_nodes; node++) {
            int gid = wsElNodeID[ws][cell][node];
            excludeNodes.insert(gid);
          }
        }
      }
      // remove from the excludeNodes set all nodes that are on boundaries 
      // between smoothed and non-smoothed blocks
      std::set<int>::iterator it;
      for (size_t ws=0; ws<num_worksets; ws++) {
        if( find(blocks.begin(), blocks.end(), wsEBNames[ws]) == blocks.end() ) continue;
        int num_cells = coords[ws].size();
        for (int cell=0; cell<num_cells; cell++) {
          size_t num_nodes = coords[ws][cell].size();
          for (int node=0; node<num_nodes; node++) {
            int gid = wsElNodeID[ws][cell][node];
            it = excludeNodes.find(gid);
            excludeNodes.erase(it,excludeNodes.end());
          }
        }
      }
    }
  
    std::map< GlobalPoint, std::set<GlobalPoint> > neighbors;
  
    double filter_radius_sqrd = filterRadius*filterRadius;
    // awful n^2 search... all against all
    size_t dimension   = app->getDiscretization()->getNumDim();
    GlobalPoint homeNode;
    size_t num_worksets = coords.size();
    for (size_t home_ws=0; home_ws<num_worksets; home_ws++) {
      int home_num_cells = coords[home_ws].size();
      for (int home_cell=0; home_cell<home_num_cells; home_cell++) {
        size_t num_nodes = coords[home_ws][home_cell].size();
        for (int home_node=0; home_node<num_nodes; home_node++) {
          homeNode.gid = wsElNodeID[home_ws][home_cell][home_node];
          if(neighbors.find(homeNode)==neighbors.end()) {  // if this node was already accessed just skip
            for (int dim=0; dim<dimension; dim++)  {
              homeNode.coords[dim] = coords[home_ws][home_cell][home_node][dim];
            }
            std::set<GlobalPoint> my_neighbors;
            if( excludeNodes.find(homeNode.gid) == excludeNodes.end() ){
              for (size_t trial_ws=0; trial_ws<num_worksets; trial_ws++) {
                if( blocks.size() > 0 && 
                    find(blocks.begin(), blocks.end(), wsEBNames[trial_ws]) == blocks.end() ) continue;
                int trial_num_cells = coords[trial_ws].size();
                for (int trial_cell=0; trial_cell<trial_num_cells; trial_cell++) {
                  size_t trial_num_nodes = coords[trial_ws][trial_cell].size();
                  for (int trial_node=0; trial_node<trial_num_nodes; trial_node++) {
                    int gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                    if( excludeNodes.find(gid) != excludeNodes.end() ) continue; // don't add excluded nodes
                    double tmp;
                    double delta_norm_sqr = 0.;
                    for (int dim=0; dim<dimension; dim++)  { //individual coordinates
                      tmp = homeNode.coords[dim]-coords[trial_ws][trial_cell][trial_node][dim];
                      delta_norm_sqr += tmp*tmp;
                    }
                    if(delta_norm_sqr<=filter_radius_sqrd) {
                      GlobalPoint newIntx;
                      newIntx.gid = wsElNodeID[trial_ws][trial_cell][trial_node];
                      for (int dim=0; dim<dimension; dim++) 
                        newIntx.coords[dim] = coords[trial_ws][trial_cell][trial_node][dim];
                      my_neighbors.insert(newIntx);
                    }
                  }
                }
              }
            }
            neighbors.insert( std::pair<GlobalPoint,std::set<GlobalPoint> >(homeNode,my_neighbors) );
          }
        }
      }
    }

    // communicate neighbor data
    importNeighbors(neighbors,importerT,*localNodeMapT,exporterT,*overlapNodeMapT);
    
    // for each interior node, search boundary nodes for additional interactions off processor.
    
    // now build filter operator
    int numnonzeros = 0;
    Teuchos::RCP<Epetra_Comm> comm = 
      Albany::createEpetraCommFromTeuchosComm(localNodeMapT->getComm());
    Teuchos::RCP<Epetra_Map> localNodeMap = Petra::TpetraMap_To_EpetraMap(localNodeMapT, comm); 
    filterOperatorT = Teuchos::rcp(new Tpetra_CrsMatrix(localNodeMapT,numnonzeros));
    for (std::map<GlobalPoint,std::set<GlobalPoint> >::iterator 
        it=neighbors.begin(); it!=neighbors.end(); ++it) { 
      GlobalPoint homeNode = it->first;
      GO home_node_gid = homeNode.gid;
      std::set<GlobalPoint> connected_nodes = it->second;
      ST weight; 
      ST zero = 0.0;  
      Teuchos::Array<ST> weightT(1);
      if( connected_nodes.size() > 0 ){
        for (std::set<GlobalPoint>::iterator 
             set_it=connected_nodes.begin(); set_it!=connected_nodes.end(); ++set_it) {
           GO neighbor_node_gid = set_it->gid;
           const double* coords = &(set_it->coords[0]);
           double distance = 0.0;
           for (int dim=0; dim<dimension; dim++) 
             distance += (coords[dim]-homeNode.coords[dim])*(coords[dim]-homeNode.coords[dim]);
           distance = (distance > 0.0) ? sqrt(distance) : 0.0;
           weight = filterRadius - distance;
           filterOperatorT->insertGlobalValues(home_node_gid,1,&zero,&neighbor_node_gid); 
           filterOperatorT->replaceGlobalValues(home_node_gid,1,&weight,&neighbor_node_gid); 
        }
      } else {
         // if the list of connected nodes is empty, still add a one on the diagonal.
         weight = 1.0;
         filterOperatorT->insertGlobalValues(home_node_gid,1,&zero,&home_node_gid); 
         filterOperatorT->replaceGlobalValues(home_node_gid,1,&weight,&home_node_gid); 
      }
    }
  
    filterOperatorT->fillComplete();

    // scale filter operator so rows sum to one.
    Teuchos::RCP<Tpetra_Vector> rowSumsT = Teuchos::rcp(new Tpetra_Vector(filterOperatorT->getRowMap()));
    Albany::InvRowSum(rowSumsT, filterOperatorT); 
    filterOperatorT->leftScale(*rowSumsT); 

    //IKT, FIXME: remove the following creation of filterOperatorTransposeT 
    //once Mark Hoemmen fixes apply method with TRANS mode in Tpetra::CrsMatrix.
    Tpetra_RowMatrixTransposer transposer(filterOperatorT);
    filterOperatorTransposeT = transposer.createTranspose();

  return;

}


/******************************************************************************/
ATO::SpatialFilter::SpatialFilter( Teuchos::ParameterList& params )
/******************************************************************************/
{
  filterRadius = params.get<double>("Filter Radius");
  if( params.isType<Teuchos::Array<std::string> >("Blocks") ){
    blocks = params.get<Teuchos::Array<std::string> >("Blocks");
  }
  if( params.isType<int>("Iterations") ){
    iterations = params.get<int>("Iterations");
  } else
    iterations = 1;

}

/******************************************************************************/
void 
ATO::SpatialFilter::importNeighbors( 
  std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >& neighbors,
  Teuchos::RCP<Tpetra_Import> importerT, 
  const Tpetra_Map& impNodeMapT,
  Teuchos::RCP<Tpetra_Export> exporterT, 
  const Tpetra_Map& expNodeMapT)
/******************************************************************************/
{
  // get from the exporter the node global ids and the associated processor ids
  std::map<int, std::set<int> > boundaryNodesByProc;

  Teuchos::ArrayView<const LO> exportLIDsT = exporterT->getExportLIDs(); 
  Teuchos::ArrayView<const int> exportPIDsT = exporterT->getExportPIDs(); 
  int numExportIDsT = exporterT->getNumExportIDs();
  std::map<int, std::set<int> >::iterator procIter;
  for(int i=0; i<numExportIDsT; i++){
    procIter = boundaryNodesByProc.find(exportPIDsT[i]);
    int exportGIDT = expNodeMapT.getGlobalElement(exportLIDsT[i]);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(exportGIDT);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(exportPIDsT[i],newSet) );
    } else {
      procIter->second.insert(exportGIDT);
    }
  }

  exportLIDsT = importerT->getExportLIDs();
  exportPIDsT = importerT->getExportPIDs();
  numExportIDsT = importerT->getNumExportIDs();

  for(int i=0; i<numExportIDsT; i++){
    procIter = boundaryNodesByProc.find(exportPIDsT[i]);
    int exportGIDT = impNodeMapT.getGlobalElement(exportLIDsT[i]);
    if( procIter == boundaryNodesByProc.end() ){
      std::set<int> newSet;
      newSet.insert(exportGIDT);
      boundaryNodesByProc.insert( std::pair<int,std::set<int> >(exportPIDsT[i],newSet) );
    } else {
      procIter->second.insert(exportGIDT);
    }
  }

  int newPoints = 1;
  
  while(newPoints > 0){
    newPoints = 0;

    int numNeighborProcs = boundaryNodesByProc.size();
    std::vector<std::vector<int> > numNeighbors_send(numNeighborProcs);
    std::vector<std::vector<int> > numNeighbors_recv(numNeighborProcs);
    
 
    // determine number of neighborhood nodes to be communicated
    int index = 0;
    std::map<int, std::set<int> >::iterator boundaryNodesIter;
    for( boundaryNodesIter=boundaryNodesByProc.begin(); 
         boundaryNodesIter!=boundaryNodesByProc.end(); 
         boundaryNodesIter++){
   
      int send_to = boundaryNodesIter->first;
      int recv_from = send_to;
  
      std::set<int>& boundaryNodes = boundaryNodesIter->second; 
      int numNodes = boundaryNodes.size();
  
      numNeighbors_send[index].resize(numNodes);
      numNeighbors_recv[index].resize(numNodes);
  
      ATO::GlobalPoint sendPoint;
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator sendPointIter;
      int localIndex = 0;
      std::set<int>::iterator boundaryNodeGID;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        sendPoint.gid = *boundaryNodeGID;
        sendPointIter = neighbors.find(sendPoint);
        TEUCHOS_TEST_FOR_EXCEPT( sendPointIter == neighbors.end() );
        std::set<ATO::GlobalPoint>& sendPointSet = sendPointIter->second;
        numNeighbors_send[index][localIndex] = sendPointSet.size();
        localIndex++;
      }
  
      MPI_Status status;
      MPI_Sendrecv(&(numNeighbors_send[index][0]), numNodes, MPI_INT, send_to, 0,
                   &(numNeighbors_recv[index][0]), numNodes, MPI_INT, recv_from, 0,
                   MPI_COMM_WORLD, &status);
      index++;
    }
  
    // new neighbors can't be immediately added to the neighbor map or they'll be
    // found and added to the list that's communicated to other procs.  This causes
    // problems because the message length has already been communicated.  
    std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> > newNeighbors;
  
    // communicate neighborhood nodes
    index = 0;
    for( boundaryNodesIter=boundaryNodesByProc.begin(); 
         boundaryNodesIter!=boundaryNodesByProc.end(); 
         boundaryNodesIter++){
   
      // determine total message size
      int totalNumEntries_send = 0;
      int totalNumEntries_recv = 0;
      std::vector<int>& send = numNeighbors_send[index];
      std::vector<int>& recv = numNeighbors_recv[index];
      int totalNumNodes = send.size();
      for(int i=0; i<totalNumNodes; i++){
        totalNumEntries_send += send[i];
        totalNumEntries_recv += recv[i];
      }
  
      int send_to = boundaryNodesIter->first;
      int recv_from = send_to;
  
      ATO::GlobalPoint* GlobalPoints_send = new ATO::GlobalPoint[totalNumEntries_send];
      ATO::GlobalPoint* GlobalPoints_recv = new ATO::GlobalPoint[totalNumEntries_recv];
      
      // copy into contiguous memory
      std::set<int>& boundaryNodes = boundaryNodesIter->second;
      ATO::GlobalPoint sendPoint;
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator sendPointIter;
      std::set<int>::iterator boundaryNodeGID;
      int numNodes = boundaryNodes.size();
      int offset = 0;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        // get neighbors for boundary node i
        sendPoint.gid = *boundaryNodeGID;
        sendPointIter = neighbors.find(sendPoint);
        TEUCHOS_TEST_FOR_EXCEPT( sendPointIter == neighbors.end() );
        std::set<ATO::GlobalPoint>& sendPointSet = sendPointIter->second;
        // copy neighbors into contiguous memory
        for(std::set<ATO::GlobalPoint>::iterator igp=sendPointSet.begin(); 
            igp!=sendPointSet.end(); igp++){
          GlobalPoints_send[offset] = *igp;
          offset++;
        }
      }
  
      MPI_Status status;
      MPI_Sendrecv(GlobalPoints_send, totalNumEntries_send, MPI_GlobalPoint, send_to, 0,
                   GlobalPoints_recv, totalNumEntries_recv, MPI_GlobalPoint, recv_from, 0,
                   MPI_COMM_WORLD, &status);
  
      // copy out of contiguous memory
      ATO::GlobalPoint recvPoint;
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator recvPointIter;
      offset = 0;
      int localIndex=0;
      for(boundaryNodeGID=boundaryNodes.begin(); 
          boundaryNodeGID!=boundaryNodes.end();
          boundaryNodeGID++){
        recvPoint.gid = *boundaryNodeGID;
        recvPointIter = newNeighbors.find(recvPoint);
        if( recvPointIter == newNeighbors.end() ){ // not found, add.
          std::set<ATO::GlobalPoint> newPointSet;
          int nrecv = recv[localIndex];
          for(int j=0; j<nrecv; j++){
            newPointSet.insert(GlobalPoints_recv[offset]);
            offset++;
          }
          newNeighbors.insert( std::pair<ATO::GlobalPoint,std::set<ATO::GlobalPoint> >(recvPoint,newPointSet) );
        } else {
          int nrecv = recv[localIndex];
          for(int j=0; j<nrecv; j++){
            recvPointIter->second.insert(GlobalPoints_recv[offset]);
            offset++;
          }
        }
        localIndex++;
      }
   
      delete [] GlobalPoints_send;
      delete [] GlobalPoints_recv;
      
      index++;
    }
  
    // add newNeighbors map to neighbors map
    std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator new_nbr;
    std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator nbr;
    std::set< ATO::GlobalPoint >::iterator newPoint;
    // loop on total neighbor list
    for(nbr=neighbors.begin(); nbr!=neighbors.end(); nbr++){
  
      std::set<ATO::GlobalPoint>& pointSet = nbr->second;
      int pointSetSize = pointSet.size();
  
      ATO::GlobalPoint home_point = nbr->first;
      double* home_coords = &(home_point.coords[0]);
      std::map< ATO::GlobalPoint, std::set<ATO::GlobalPoint> >::iterator nbrs;
      std::set< ATO::GlobalPoint >::iterator remote_point;
      for(nbrs=newNeighbors.begin(); nbrs!=newNeighbors.end(); nbrs++){
        std::set<ATO::GlobalPoint>& remote_points = nbrs->second;
        for(remote_point=remote_points.begin(); 
            remote_point!=remote_points.end();
            remote_point++){
          const double* remote_coords = &(remote_point->coords[0]);
          double distance = 0.0;
          for(int i=0; i<3; i++)
            distance += (remote_coords[i]-home_coords[i])*(remote_coords[i]-home_coords[i]);
          distance = (distance > 0.0) ? sqrt(distance) : 0.0;
          if( distance < filterRadius )
            pointSet.insert(*remote_point);
        }
      }
      // see if any new points where found off processor.  
      newPoints += (pointSet.size() - pointSetSize);
    }
    int globalNewPoints=0;
    Teuchos::reduceAll(*(impNodeMapT.getComm()), Teuchos::REDUCE_SUM, 1, &newPoints, &globalNewPoints); 
    newPoints = globalNewPoints;
  }
}
  
