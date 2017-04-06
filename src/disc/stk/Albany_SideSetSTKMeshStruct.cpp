//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_SideSetSTKMeshStruct.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Albany_Utils.hpp"

namespace Albany
{

SideSetSTKMeshStruct::SideSetSTKMeshStruct (const MeshSpecsStruct& inputMeshSpecs,
                                            const Teuchos::RCP<Teuchos::ParameterList>& params,
                                            const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct(params, Teuchos::null)
{
  TEUCHOS_TEST_FOR_EXCEPTION (inputMeshSpecs.numDim!=3, std::logic_error,
                              "Error! For now input mesh has to be 3D.\n");

  params->validateParameters(*getValidDiscretizationParameters(),0);

  // Initializing the MetaData (default numDim=-1 prevents that in GenericSTKMeshStruct)
  this->numDim = inputMeshSpecs.numDim - 1;
  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
  if (this->buildEMesh)
    entity_rank_names.push_back("FAMILY_TREE");
  metaData->initialize(this->numDim, entity_rank_names);

  std::string ebn = "Element Block 0";
  partVec[0] = &metaData->declare_part(ebn, stk::topology::ELEMENT_RANK);
  ebNameToIndex[ebn] = 0;
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  std::vector<std::string> nsNames;
  std::string nsn = "all_nodes";
  nsNames.push_back(nsn);
  nsPartVec[nsn] = &metaData->declare_part(nsn, stk::topology::NODE_RANK);
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif

  std::string input_elem_name = inputMeshSpecs.ctd.base->name;
  if (input_elem_name=="Tetrahedron_4" || input_elem_name=="Wedge_6")
  {
    stk::mesh::set_cell_topology<shards::Triangle<3> >(*partVec[0]);
  }
  else if (input_elem_name=="Hexahedron_8")
  {
    stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*partVec[0]);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The side-set mesh extraction has not yet been implemented for this type of element.\n");
  }

  std::vector<std::string> ssNames; // Empty
  int cub = params->get("Cubature Degree", 3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  int worksetSize = this->computeWorksetSize(worksetSizeMax,inputMeshSpecs.worksetSize);
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  this->meshSpecs[0] = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, this->numDim, cub, nsNames, ssNames, worksetSize,
                                                                ebn, ebNameToIndex, this->interleavedOrdering));

  const Teuchos::MpiComm<int>* mpiComm = dynamic_cast<const Teuchos::MpiComm<int>* > (commT.get());
  bulkData = Teuchos::rcp(new stk::mesh::BulkData(*metaData, *mpiComm->getRawMpiComm(),
                                                   stk::mesh::BulkData::NO_AUTO_AURA,
                                                   false, NULL, NULL, worksetSize));
}

SideSetSTKMeshStruct::~SideSetSTKMeshStruct()
{
  // Nothing to be done here
}

void SideSetSTKMeshStruct::setParentMeshInfo (const AbstractSTKMeshStruct& parentMeshStruct_,
                                              const std::string& sideSetName)
{
  parentMeshStruct      = Teuchos::rcpFromRef(parentMeshStruct_);
  parentMeshSideSetName = sideSetName;
}

void SideSetSTKMeshStruct::setFieldAndBulkData (
      const Teuchos::RCP<const Teuchos_Comm>& commT,
      const Teuchos::RCP<Teuchos::ParameterList>& /*params*/,
      const unsigned int neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<StateInfoStruct>& sis,
      const unsigned int worksetSize,
      const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& /*side_set_sis*/,
      const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& /*side_set_req*/)
{
  this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  TEUCHOS_TEST_FOR_EXCEPTION (parentMeshStruct->ssPartVec.find(parentMeshSideSetName)==parentMeshStruct->ssPartVec.end(), std::logic_error,
                              "Error! The side set " << parentMeshSideSetName << " is not present in the input mesh.\n");

  // Extracting the side part and updating the selector
  const stk::mesh::Part& ss_part = *parentMeshStruct->ssPartVec.find(parentMeshSideSetName)->second;
  stk::mesh::Selector select_required_ss(ss_part);

  const stk::mesh::MetaData& inputMetaData = *parentMeshStruct->metaData;
  const stk::mesh::BulkData& inputBulkData = *parentMeshStruct->bulkData;

  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  const VectorFieldType& parent_coordinates_field = *parentMeshStruct->getCoordinatesField();
  VectorFieldType&       coordinates_field        = *fieldContainer->getCoordinatesField();

  // Now we can extract the entities
  std::vector<stk::mesh::Entity> sides, nodes;
  stk::mesh::get_selected_entities (select_required_ss, inputBulkData.buckets(inputMetaData.side_rank()), sides);
  stk::mesh::get_selected_entities (select_required_ss, inputBulkData.buckets(stk::topology::NODE_RANK), nodes);

  // Insertion of the entities begins
  bulkData->modification_begin();
  stk::mesh::PartVector singlePartVec(1);

  // Adding nodes
  stk::mesh::Entity node;
  stk::mesh::EntityId nodeId;
  singlePartVec[0] = nsPartVec["all_nodes"];
  for (int inode(0); inode<nodes.size(); ++inode)
  {
    // Adding the node (same Id)
    nodeId = inputBulkData.identifier(nodes[inode]);
    node = bulkData->declare_entity(stk::topology::NODE_RANK, nodeId, singlePartVec);

    double* coord = stk::mesh::field_data(coordinates_field, node);
    double const* p_coord = stk::mesh::field_data(parent_coordinates_field, nodes[inode]);

    for (int idim=0; idim<metaData->spatial_dimension(); ++idim)
      coord[idim] = p_coord[idim];

    // Checking for shared node
    std::vector<int> sharing_procs;
    inputBulkData.comm_shared_procs( inputBulkData.entity_key(nodes[inode]), sharing_procs );
    for(int iproc(0); iproc<sharing_procs.size(); ++iproc)
      bulkData->add_node_sharing(node, sharing_procs[iproc]);
  }

  // Adding sides (aka elements in the boundary mesh)
  stk::mesh::Entity elem;
  stk::mesh::EntityId elemId;
  singlePartVec[0] = partVec[0];
  for (int iside(0); iside<sides.size(); ++iside)
  {
    // Adding the element (same Id as the side)
    elemId = inputBulkData.identifier(sides[iside]);
    elem = bulkData->declare_entity(stk::topology::ELEM_RANK, elemId, singlePartVec);

    // Adding the relation elem->node
    stk::mesh::Entity const* node_rels = inputBulkData.begin_nodes(sides[iside]);
    const int num_local_nodes = inputBulkData.num_nodes(sides[iside]);
    for (int j(0); j<num_local_nodes; ++j)
    {
      node = bulkData->get_entity(stk::topology::NODE_RANK, inputBulkData.identifier(node_rels[j]));
      bulkData->declare_relation(elem, node, j);
    }
  }

  // Insertion of entities end
  bulkData->modification_end();

  // Loading the fields from file
  this->loadRequiredInputFields (req,commT);
}

Teuchos::RCP<const Teuchos::ParameterList> SideSetSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("Valid SideSetSTK DiscParams");

  return validPL;
}

} // Namespace Albany
