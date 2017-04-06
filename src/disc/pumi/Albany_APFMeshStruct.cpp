//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_APFMeshStruct.hpp"

#include <iostream>

#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TwoDArray.hpp"
#include <Shards_BasicTopologies.hpp>

#include <sstream>

#if defined(ALBANY_SCOREC) || defined(ALBANY_AMP)
#include <PCU.h>
#endif
#if (defined(ALBANY_SCOREC) && defined(SCOREC_SIMMODEL)) || defined(ALBANY_AMP)
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#ifdef ALBANY_SCOREC
#include <gmi_mesh.h>
#endif
#ifdef ALBANY_AMP
#include <SimPartitionedMesh.h>
#include <MeshSim.h>
#include <SimDiscrete.h>
#include <SimField.h>
#endif

// Capitalize Solution so that it sorts before other fields in Paraview. Saves a
// few button clicks, e.g. when warping by vector.
const char* Albany::APFMeshStruct::solution_name[3] = {"Solution", "SolutionDot", "SolutionDotDot"};
const char* Albany::APFMeshStruct::residual_name = {"residual"};

static void loadSets(
    apf::Mesh* m,
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    apf::StkModels& sets,
    const char* param_name,
    int geom_dim,
    int mesh_dim)
{
  // User has specified associations in the input file
  if(params->isParameter(param_name)) {
    // Get element block associations from input file
    Teuchos::TwoDArray< std::string > pairs;
    pairs = params->get<Teuchos::TwoDArray<std::string> >(param_name);
    int npairs = pairs.getNumCols();
    for(int i = 0; i < npairs; ++i) {
      apf::StkModel* set = new apf::StkModel();
      int geom_tag = atoi(pairs(0, i).c_str());
      set->ents.push_back(m->findModelEntity(geom_dim, geom_tag));
      set->stkName = pairs(1, i);
      sets.models[mesh_dim].push_back(set);
    }
  }
}

static void readSets(
    apf::Mesh* m,
    const char* filename,
    apf::StkModels& sets)
{
  static std::string const dimNames[3] = {
    "node set",
    "side set",
    "element block"};
  int d = m->getDimension();
  int dims[3] = {0, d - 1, d};
  std::ifstream f(filename);
  TEUCHOS_TEST_FOR_EXCEPTION(!f, std::logic_error,
      "Could not open associations file " << filename << '\n');
  std::string sline;
  int lc = 0;
  while (std::getline(f, sline)) {
    if (!sline.length())
      break;
    ++lc;
    int sdi = -1;
    for (int di = 0; di < 3; ++di)
      if (sline.compare(0, dimNames[di].length(), dimNames[di]) == 0)
        sdi = di;
    TEUCHOS_TEST_FOR_EXCEPTION(sdi == -1, std::logic_error,
        "Bad associations line #" << lc << " \"" << sline << "\"\n");
    int sd = dims[sdi];
    std::stringstream strs(sline.substr(dimNames[sdi].length()));
    apf::StkModel* set = new apf::StkModel();
    strs >> set->stkName;
    int nents;
    strs >> nents;
    TEUCHOS_TEST_FOR_EXCEPTION(!strs, std::logic_error,
        "Bad associations line #" << lc << " \"" << sline << "\"\n");
    for (int ei = 0; ei < nents; ++ei) {
      std::string eline;
      std::getline(f, eline);
      TEUCHOS_TEST_FOR_EXCEPTION(!f || !eline.length(), std::logic_error,
          "Missing associations after line #" << lc << "\n");
      ++lc;
      std::stringstream strs2(eline);
      int mdim, mtag;
      strs2 >> mdim >> mtag;
      TEUCHOS_TEST_FOR_EXCEPTION(!strs2, std::logic_error,
          "Bad associations line #" << lc << " \"" << eline << "\"\n");
      set->ents.push_back(m->findModelEntity(mdim, mtag));
      TEUCHOS_TEST_FOR_EXCEPTION(!set->ents.back(), std::logic_error,
          "No model entity (" << mdim << ", " << mtag << ")\n");
    }
    sets.models[sd].push_back(set);
  }
}

static void getEBSizes(
    apf::Mesh* mesh,
    apf::StkModels& sets,
    std::vector<int>& el_blocks)
{
  int d = mesh->getDimension();
  apf::MeshIterator* mit = mesh->begin(mesh->getDimension());
  apf::MeshEntity* e;
  std::map<apf::StkModel*, int> sizeMap;
  while ((e = mesh->iterate(mit))) {
    apf::ModelEntity* me = mesh->toModel(e);
    if (sets.invMaps[d].count(me))
      ++(sizeMap[sets.invMaps[d][me]]);
  }
  mesh->end(mit);
  el_blocks.resize(sets.models[d].size());
  for (size_t i = 0; i < sets.models[d].size(); ++i)
    el_blocks[i] = sizeMap[sets.models[d][i]];
}

void Albany::APFMeshStruct::init(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  out = Teuchos::VerboseObjectBase::getDefaultOStream();

  useNullspaceTranslationOnly = params->get<bool>("Use Nullspace Translation Only", false);
  useTemperatureHack = params->get<bool>("QP Temperature from Nodes", false);
  useDOFOffsetHack = params->get<bool>("Offset DOF Hack", false);
  saveStabilizedStress = params->get<bool>("Save Stabilized Stress", false);

  compositeTet = false;

  mesh->verify();

  int d = mesh->getDimension();
  if (params->isParameter("Model Associations File Name")) {
    std::string afile = params->get<std::string>("Model Associations File Name");
    readSets(mesh, afile.c_str(), sets);
  } else {
    loadSets(mesh, params, sets, "Element Block Associations",   d,     d);
    loadSets(mesh, params, sets, "Node Set Associations",        d - 1, 0);
    loadSets(mesh, params, sets, "Edge Node Set Associations",   1,     0);
    loadSets(mesh, params, sets, "Vertex Node Set Associations", 0,     0);
    loadSets(mesh, params, sets, "Side Set Associations",        d - 1, d - 1);
  }
  sets.computeInverse();

  numDim = mesh->getDimension();

  // Build a map to get the EB name given the index

  int numEB = sets.models[d].size();

  // getEBSizes will seg fault if at least one element block isn't defined in input. Lets exit a little more gracefully.
  TEUCHOS_TEST_FOR_EXCEPTION(numEB == 0, std::logic_error,
     "PUMI requires at least one element block to be defined in the input file.");

  std::vector<int> el_blocks;
  getEBSizes(mesh, sets, el_blocks);

  for (int eb=0; eb < numEB; eb++){
    apf::StkModel* set = sets.models[d][eb];
    this->ebNameToIndex[set->stkName] = eb;
  }

  // Set defaults for cubature and workset size, overridden in input file

  cubatureDegree = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  interleavedOrdering = params->get("Interleaved Ordering",true);
  num_time_deriv = params->get<int>("Number Of Time Derivatives", 0);
  allElementBlocksHaveSamePhysics = true;
  hasRestartSolution = false;
  shouldLoadFELIXData = false;

  // No history available by default
  solutionFieldHistoryDepth = 0;

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);

  // Get number of elements per element block
  // in calculating an upper bound on the worksetSize.

  int ebSizeMax =  *std::max_element(el_blocks.begin(), el_blocks.end());
  worksetSize = computeWorksetSize(worksetSizeMax, ebSizeMax);

  // Node sets
  for(size_t ns = 0; ns < sets.models[0].size(); ns++)
    nsNames.push_back(sets.models[0][ns]->stkName);

  // Side sets
  for(size_t ss = 0; ss < sets.models[d - 1].size(); ss++) {
    ssNames.push_back(sets.models[d - 1][ss]->stkName);
  }

  // Construct MeshSpecsStruct
  const CellTopologyData* ctd = apf::getCellTopology(mesh);
  if (!params->get("Separate Evaluators by Element Block",false))
  {
    // get elements in the first element block
    std::string EB_name = sets.models[d][0]->stkName;
    this->meshSpecs[0] = Teuchos::rcp(
        new Albany::MeshSpecsStruct(
          *ctd, numDim, cubatureDegree,
          nsNames, ssNames, worksetSize, EB_name,
          this->ebNameToIndex, this->interleavedOrdering));
  }
  else
  {
    this->allElementBlocksHaveSamePhysics=false;
    this->meshSpecs.resize(numEB);
    int eb_size;
    std::string eb_name;
    for (int eb=0; eb<numEB; eb++)
    {
      std::string EB_name = sets.models[d][eb]->stkName;
      this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(
          *ctd, numDim, cubatureDegree, nsNames, ssNames, worksetSize, EB_name,
          this->ebNameToIndex, this->interleavedOrdering, true));
    } // for
  } // else

  shouldWriteAsciiVtk = params->get<bool>("Write ASCII VTK Files", false);

}

Albany::APFMeshStruct::~APFMeshStruct()
{
}

void
Albany::APFMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize_,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& /*side_set_sis*/,
                  const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& /*side_set_req*/)
{

  using Albany::StateStruct;

  // Set the number of equation present per node. Needed by Albany_APFDiscretization.
  neq = neq_;

  this->nodal_data_base = sis->getNodalDataBase();

  Teuchos::Array<std::string> defaultLayout; // empty
  solVectorLayout.resize(3);

  // Set this to the user set value or to empty if not specified
  solVectorLayout[0] =
    params->get<Teuchos::Array<std::string> >("Solution Vector Components", defaultLayout);

  // check that the user entered an even number of strings
  TEUCHOS_TEST_FOR_EXCEPTION((solVectorLayout[0].size() % 2), std::logic_error,
      "Error in input file: specification of solution vector layout is incorrect\n");

  // Set up a default layout for solution dot, based on what was specified in the base layout
  Teuchos::Array<std::string> dotLayout;
  dotLayout.resize(solVectorLayout[0].size());
  for(int i = 0; i < solVectorLayout[0].size(); i += 2){
     dotLayout[i] = solVectorLayout[0][i] + "Dot"; // concat the term Dot
     dotLayout[i + 1] = solVectorLayout[0][i + 1];
  }

  // Let the user specify something beyond the default if desired
  solVectorLayout[1] =
    params->get<Teuchos::Array<std::string> >("SolutionDot Vector Components", dotLayout);

  TEUCHOS_TEST_FOR_EXCEPTION((solVectorLayout[1].size() % 2), std::logic_error,
      "Error in input file: specification of solution vector dot layout is incorrect\n");

  // Set up a default layout for solution dot, based on what was specified in the base layout
  Teuchos::Array<std::string> dotdotLayout;
  dotdotLayout.resize(solVectorLayout[0].size());
  for(int i = 0; i < solVectorLayout[0].size(); i += 2){
     dotdotLayout[i] = solVectorLayout[0][i] + "DotDot"; // concat the term DotDot
     dotdotLayout[i + 1] = solVectorLayout[0][i + 1];
  }

  solVectorLayout[2] =
    params->get<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", dotdotLayout);

  TEUCHOS_TEST_FOR_EXCEPTION((solVectorLayout[2].size() % 2), std::logic_error,
      "Error in input file: specification of solution vector dotdot layout is incorrect\n");

  solutionInitialized = true;
  residualInitialized = true;

  if (solVectorLayout[0].size() == 0 && (neq == 2 || neq == 4)) problemDim = 2;
  else problemDim = numDim;

  if (solVectorLayout[0].size() == 0) {

    // If the user has not specified a solution vector layout, provide a simple default
    // Note that the logic here requires that the user enter something for the "Solution Vector Components",
    // or they will get the default.

    for(int i = 0; i <= num_time_deriv; i++){

      int valueType;
      if (neq == 1) {
        valueType = apf::SCALAR;
      } else if (neq == problemDim) {
        valueType = apf::VECTOR;
      } else {
        assert(neq == problemDim * problemDim);
        valueType = apf::MATRIX;
      }
      if(i == 0) {
        residualInitialized = findOrCreateNodalField(residual_name, valueType);
      }
      bool found_field = findOrCreateNodalField(solution_name[i], valueType);
      solutionInitialized = solutionInitialized && found_field;
    }
  } else {
    splitFields(solVectorLayout);
  }

  // Code to parse the vector of StateStructs and save the information

  // dim[0] is the number of cells in this workset
  // dim[1] is the number of QP per cell
  // dim[2] is the number of dimensions of the field
  // dim[3] is the number of dimensions of the field

  std::set<std::string> nameSet;

  for (std::size_t i=0; i<sis->size(); i++) {
    StateStruct& st = *((*sis)[i]);

#ifdef ALBANY_SCOREC
    if (meshSpecsType() == AbstractMeshStruct::PUMI_MS) {
      if(hasRestartSolution)
        st.restartDataAvailable = true;
      if((shouldLoadFELIXData) && (st.entity == StateStruct::NodalDataToElemNode))
        st.restartDataAvailable = true;
    }
#endif

    if ( ! nameSet.insert(st.name).second)
      continue; //ignore duplicates
    std::vector<PHX::DataLayout::size_type>& dim = st.dim;

    if(st.entity == StateStruct::NodalData) { // Data at the node points
       const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer
               = sis->getNodalDataBase()->getNodeContainer();
        (*nodeContainer)[st.name] = Albany::buildPUMINodeField(st.name, dim, st.output);
    }
    else if (dim.size() == 2) {
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode)
        qpscalar_states.push_back(Teuchos::rcp(new PUMIQPData<double, 2>(st.name, dim, st.output)));
      else if(st.entity == StateStruct::NodalDataToElemNode)
        elemnodescalar_states.push_back(Teuchos::rcp(new PUMIQPData<double, 2>(st.name, dim, st.output)));
    }
    else if (dim.size() == 3) {
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode)
        qpvector_states.push_back(Teuchos::rcp(new PUMIQPData<double, 3>(st.name, dim, st.output)));
    }
    else if (dim.size() == 4){
      if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode)
        qptensor_states.push_back(Teuchos::rcp(new PUMIQPData<double, 4>(st.name, dim, st.output)));
    }
    else if ( dim.size() == 1 && st.entity == Albany::StateStruct::WorksetValue)
      scalarValue_states.push_back(Teuchos::rcp(new PUMIQPData<double, 1>(st.name, dim, st.output)));
    else TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "dim.size() < 2 || dim.size()>4 || " <<
         "st.entity != Albany::StateStruct::QuadPoint || " <<
         "st.entity != Albany::StateStruct::ElemNode || " <<
         "st.entity != Albany::StateStruct::NodalData" << std::endl);
  }
}

bool
Albany::APFMeshStruct::findOrCreateNodalField(const char* name, int value_type) {
  apf::Field* f = mesh->findField(name);
  if (f) {
    /* fields may have been created by the restart mechanism,
     * but we should still check that they are of the expected type
     */
    assert(apf::getShape(f) == mesh->getShape());
    assert(apf::getValueType(f) == value_type);
    return true;
  } else {
    this->createNodalField(name, value_type);
    return false;
  }
}

void
Albany::APFMeshStruct::splitFields(Teuchos::Array<Teuchos::Array<std::string> >& fieldLayout)
{ // user is breaking up or renaming solution & residual fields

  for(int fcomp = 0; fcomp <= num_time_deriv; fcomp++){

    TEUCHOS_TEST_FOR_EXCEPTION((fieldLayout[fcomp].size() % 2), std::logic_error,
        "Error in input file: specification of solution vector layout is incorrect\n");

    int valueType;

    for (std::size_t i=0; i < fieldLayout[fcomp].size(); i+=2) {

      TEUCHOS_TEST_FOR_EXCEPTION(mesh->findField(fieldLayout[fcomp][i].c_str()), std::logic_error,
            "Error in input file: specification of solution vector layout is incorrect\n"
            << " Found duplicate field name.");

      if (fieldLayout[fcomp][i+1] == "S")
        valueType = apf::SCALAR;
      else if (fieldLayout[fcomp][i+1] == "V")
        valueType = apf::VECTOR;
      else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
            "Error in input file: specification of solution vector layout is incorrect\n");

      bool found_field =
        findOrCreateNodalField(fieldLayout[fcomp][i].c_str(),valueType);
      solutionInitialized = solutionInitialized && found_field;

      // Add the residual field - based on the text entered in the "Solution" PL only
      if(fcomp == 0){
        std::string res_name = fieldLayout[fcomp][i];
        res_name.append("Res");
        residualInitialized = findOrCreateNodalField(res_name.c_str(), valueType);
      }
    }
  }
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::APFMeshStruct::getMeshSpecs()
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::APFMeshStruct::getMeshSpecs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

int Albany::APFMeshStruct::computeWorksetSize(const int worksetSizeMax,
                                                     const int ebSizeMax) const
{
  // Resize workset size down to maximum number in an element block
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) {
    return ebSizeMax;
  } else {
     // compute numWorksets, and shrink workset size to minimize padding
     const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
     return (1 + (ebSizeMax-1) /  numWorksets);
  }
}

void
Albany::APFMeshStruct::loadSolutionFieldHistory(int step)
{
  TEUCHOS_TEST_FOR_EXCEPT(step < 0 || step >= solutionFieldHistoryDepth);
}

void Albany::APFMeshStruct::setupMeshBlkInfo()
{
   int nBlocks = this->meshSpecs.size();
   for(int i = 0; i < nBlocks; i++){
      const Albany::MeshSpecsStruct &ms = *meshSpecs[i];
      meshDynamicData[i] = Teuchos::rcp(new Albany::CellSpecs(ms.ctd, ms.worksetSize, ms.cubatureDegree,
                      numDim, neq, 0, useCompositeTet()));
   }
}

Teuchos::RCP<Teuchos::ParameterList>
Albany::APFMeshStruct::getValidDiscretizationParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL
     = rcp(new Teuchos::ParameterList("Valid APFParams"));

  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid2");
  validPL->set<int>("Workset Size", DEFAULT_WORKSET_SIZE, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");
  validPL->set<bool>("Separate Evaluators by Element Block", false,
                     "Flag for different evaluation trees for each Element Block");
  Teuchos::Array<std::string> defaultFields;
  validPL->set<Teuchos::Array<std::string> >("Solution Vector Components", defaultFields,
      "Names and layouts of solution vector components");
  validPL->set<Teuchos::Array<std::string> >("SolutionDot Vector Components", defaultFields,
      "Names and layouts of solution_dot vector components");
  validPL->set<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", defaultFields,
      "Names and layouts of solution_dotdot vector components");

  validPL->set<std::string>("Acis Model Input File Name", "", "File Name For ACIS Model Input");
  validPL->set<std::string>("Parasolid Model Input File Name", "", "File Name For PARASOLID Model Input");

  Teuchos::TwoDArray<std::string> defaultData;
  validPL->set<Teuchos::TwoDArray<std::string> >("Element Block Associations", defaultData,
      "Association between region ID and element block string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Node Set Associations", defaultData,
      "Association between geometric face ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Edge Node Set Associations", defaultData,
      "Association between geometric edge ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Vertex Node Set Associations", defaultData,
      "Association between geometric edge ID and node set string");
  validPL->set<Teuchos::TwoDArray<std::string> >("Side Set Associations", defaultData,
      "Association between face ID and side set string");

  validPL->set<bool>("Use Nullspace Translation Only", false,
                     "Temporary hack to get MueLu (possibly) working for us");

  validPL->set<std::string>("Model Associations File Name", "", "File with element block/sideset/nodeset associations");

  validPL->set<bool>("QP Temperature from Nodes", false,
                     "Hack to initialize QP Temperature from Solution");

  validPL->set<bool>("Offset DOF Hack", false,
      "Offset DOF numberings to start at 2^31 - 1 to test GO types");

  validPL->set<bool>("Write ASCII VTK Files", false, "");

  return validPL;
}

void
Albany::APFMeshStruct::initialize_libraries(int* pargc, char*** pargv)
{
#if defined(ALBANY_SCOREC) || defined(ALBANY_AMP)
  PCU_Comm_Init();
#endif
#if (defined(ALBANY_SCOREC) && defined(SCOREC_SIMMODEL)) || defined(ALBANY_AMP)
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();
#endif
#ifdef ALBANY_SCOREC
  gmi_register_mesh();
#endif
#ifdef ALBANY_AMP
  SimPartitionedMesh_start(pargc, pargv);
  MS_init();
  SimDiscrete_start(0);
  SimField_start();
#endif
}

void
Albany::APFMeshStruct::finalize_libraries()
{
#ifdef ALBANY_AMP
  SimField_stop();
  SimDiscrete_stop(0);
  MS_exit();
  SimPartitionedMesh_stop();
#endif
#if (defined(ALBANY_SCOREC) && defined(SCOREC_SIMMODEL)) || defined(ALBANY_AMP)
  gmi_sim_stop();
  Sim_unregisterAllKeys();
#endif
#if defined(ALBANY_SCOREC) || defined(ALBANY_AMP)
  PCU_Comm_Free();
#endif
}
