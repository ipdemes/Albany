//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "FELIX_PopulateMesh.hpp"

namespace FELIX
{

PopulateMesh::PopulateMesh (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                            const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                            const Teuchos::RCP<ParamLib>& paramLib_) :
  Albany::AbstractProblem(params_, paramLib_),
  discParams(discParams_)
{
  neq = 1;

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);

  Teuchos::Array<std::string> empty_str_ar;
  Teuchos::Array<int> empty_int_ar;

  // Need to allocate a fields in mesh database
  Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields",empty_str_ar);
  for (int i(0); i<req.size(); ++i)
    this->requirements.push_back(req[i]);

  Teuchos::ParameterList& p = params->sublist("Side Sets Requirements");

  Teuchos::Array<std::string> ss_names = p.get<Teuchos::Array<std::string>>("Side Sets Names",empty_str_ar);
  Teuchos::Array<int> ss_vec_dims_ar = p.get<Teuchos::Array<int>>("Side Sets Vec Dims",empty_int_ar);

  TEUCHOS_TEST_FOR_EXCEPTION (ss_names.size()!=ss_vec_dims_ar.size(), Teuchos::Exceptions::InvalidParameter,
                              "Error! You must specify a vector dimension for each side set.\n");

  for (int i=0; i<ss_names.size(); ++i)
  {
    ss_vec_dims[ss_names[i]] = ss_vec_dims_ar[i];

    Teuchos::Array<std::string> reqs = p.get<Teuchos::Array<std::string>>(ss_names[i]);

    for (int j=0; j<reqs.size(); ++j)
      this->ss_requirements[ss_names[i]].push_back(reqs[j]);
  }
}

PopulateMesh::~PopulateMesh()
{
  // Nothing to be done here
}

void PopulateMesh::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
                                 Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;
  typedef Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > >    basisType;

  // Building cell basis
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  basisType cellBasis = Albany::getIntrepid2Basis(*cell_top);
  shards::CellTopology cellTopology(cell_top);
  std::string cellEBName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellTopology.getFaceCount();
  const int numCellVertices = cellTopology.getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = 0;  // Not needed
  const int numCellDim      = meshSpecs[0]->numDim;
  const int numCellVecDim   = params->get<int>("Cell Vec Dim",numCellDim);

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numCellDim,numCellVecDim));

  std::map<std::string,std::string> sideEBName;
  if (ss_vec_dims.size()>0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!discParams->isSublist("Side Set Discretizations"), std::logic_error,
                                "Error! There are side set requirements in the problem section, but no side discretizations.\n");

    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    for (auto it : ss_vec_dims)
    {
      const std::string& ss_name = it.first;
      TEUCHOS_TEST_FOR_EXCEPTION (!ss_disc_pl.isSublist(ss_name),std::logic_error,
                                  "Error! Side set '" << ss_name << "' is listed in the problem section but is missing in the discretization section.\n");

      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      const Albany::MeshSpecsStruct& ssMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ss_name)[0];

      // Building also side structures
      const CellTopologyData * const side_top = &ssMeshSpecs.ctd;
      shards::CellTopology sideTopology(side_top);
      basisType sideBasis = Albany::getIntrepid2Basis(*side_top);
      sideEBName[ss_name] = ssMeshSpecs.ebName;

      const int numSideVertices = sideTopology.getNodeCount();
      const int numSideNodes    = sideBasis->getCardinality();
      const int numSideDim      = ssMeshSpecs.numDim;
      const int numSideQPs      = 0;    // Not needed
      const int numSideVecDim   = ss_vec_dims[ss_name];

      dl->side_layouts[ss_name] = rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,numSideQPs,
                                                          numSideDim,numCellDim,numCellSides,numSideVecDim));
    }
  }

  // ---------------------------- Registering state variables ------------------------- //

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // Map string to StateStruct::MeshFieldEntity
  std::map<std::string,Albany::StateStruct::MeshFieldEntity> str2mfe;
  str2mfe["Node Scalar"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Node Vector"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Elem Scalar"] = Albany::StateStruct::ElemData;
  str2mfe["Elem Vector"] = Albany::StateStruct::ElemData;
  str2mfe["Node Layered Scalar"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Node Layered Vector"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Elem Layered Scalar"] = Albany::StateStruct::ElemData;
  str2mfe["Elem Layered Vector"] = Albany::StateStruct::ElemData;

  std::string fname, ftype;
  if (discParams->isSublist("Required Fields Info"))
  {
    // Map string to PHX layout
    std::map<std::string,Teuchos::RCP<PHX::DataLayout>> str2dl;
    str2dl["Node Scalar"] = dl->node_scalar;
    str2dl["Node Vector"] = dl->node_vector;
    str2dl["Elem Scalar"] = dl->cell_scalar2;
    str2dl["Elem Vector"] = dl->cell_vector;


    Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

      fname   = thisFieldList.get<std::string>("Field Name");
      ftype = thisFieldList.get<std::string>("Field Type");

      if (ftype.find("Layered")!=std::string::npos)
      {
        Teuchos::RCP<PHX::DataLayout> ldl;
        int numLayers = thisFieldList.get<int>("Number Of Layers");
        if (ftype=="Node Layered Scalar")
          ldl = PHAL::ExtendLayout<LayerDim,Cell,Node>::apply(dl->node_scalar,numLayers);
        else if (ftype=="Node Layered Vector")
          ldl = PHAL::ExtendLayout<LayerDim,Cell,Node,Dim>::apply(dl->node_vector,numLayers);
        else if (ftype=="Elem Layered Scalar")
          ldl = PHAL::ExtendLayout<LayerDim,Cell>::apply(dl->cell_scalar2,numLayers);
        else if (ftype=="Elem Layered Vector")
          ldl = PHAL::ExtendLayout<LayerDim,Cell,Dim>::apply(dl->cell_vector,numLayers);
        else
          TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid layout for field '" << fname << "'.\n");

        p = stateMgr.registerStateVariable(fname, ldl, cellEBName, true, &str2mfe[ftype]);
      }
      else
        p = stateMgr.registerStateVariable(fname, str2dl[ftype], cellEBName, true, &str2mfe[ftype]);
    }
  }

  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    for (int is=0; is<ss_names.size(); ++is)
    {
      const std::string& ss_name = ss_names[is];
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      if (this_ss_pl.isSublist("Required Fields Info"))
      {
        Teuchos::RCP<Albany::Layouts> sdl = dl->side_layouts[ss_name];

        // Map string to PHX layout
        std::map<std::string,Teuchos::RCP<PHX::DataLayout>> str2dl;
        str2dl["Node Scalar"] = sdl->node_scalar;
        str2dl["Node Vector"] = sdl->node_vector;
        str2dl["Elem Scalar"] = sdl->cell_scalar2;
        str2dl["Elem Vector"] = sdl->cell_vector;

        Teuchos::ParameterList& req_fields_info = this_ss_pl.sublist("Required Fields Info");
        int num_fields = req_fields_info.get<int>("Number Of Fields",0);
        for (int ifield=0; ifield<num_fields; ++ifield)
        {
          const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

          fname   = thisFieldList.get<std::string>("Field Name");
          ftype = thisFieldList.get<std::string>("Field Type");

          if (ftype.find("Layered")!=std::string::npos)
          {
            Teuchos::RCP<PHX::DataLayout> ldl;
            int numLayers = thisFieldList.get<int>("Number Of Layers");
            if (ftype=="Node Layered Scalar")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side,Node>::apply(sdl->node_scalar,numLayers);
            else if (ftype=="Node Layered Vector")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side,Node,Dim>::apply(sdl->node_vector,numLayers);
            else if (ftype=="Elem Layered Scalar")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side>::apply(sdl->cell_scalar2,numLayers);
            else if (ftype=="Elem Layered Vector")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side,Dim>::apply(sdl->cell_vector,numLayers);
            else
              TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid layout for field '" << fname << "'.\n");

            p = stateMgr.registerSideSetStateVariable(ss_name, fname, fname, ldl, sideEBName[ss_name], true, &str2mfe[ftype]);
          }
          else
          {
            p = stateMgr.registerSideSetStateVariable(ss_name, fname, fname, str2dl[ftype], sideEBName[ss_name], true, &str2mfe[ftype]);
          }
        }
      }
    }
  }

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
PopulateMesh::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const Albany::MeshSpecsStruct& meshSpecs,
                               Albany::StateManager& stateMgr,
                               Albany::FieldManagerChoice fmchoice,
                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<PopulateMesh> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

Teuchos::RCP<const Teuchos::ParameterList>
PopulateMesh::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidPopulateMeshProblemParams");
  validPL->set<int> ("Cell Vec Dim", 0, "");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->sublist("Side Sets Requirements", false, "");

  return validPL;
}

} // Namespace FELIX
