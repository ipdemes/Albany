//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_ParameterList.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFSideToCellBase<EvalT, Traits, ScalarT>::
DOFSideToCellBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  sideSetName (p.get<std::string> ("Side Set Name"))
{
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);
  std::string layout_str = p.get<std::string>("Data Layout");

  if (layout_str=="Cell Scalar")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->cell_scalar2);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->cell_scalar2);

    layout = CELL_SCALAR;
  }
  else if (layout_str=="Cell Vector")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->cell_vector);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->cell_vector);

    layout = CELL_VECTOR;
  }
  else if (layout_str=="Cell Tensor")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->cell_tensor);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->cell_tensor);

    layout = CELL_TENSOR;
  }
  else if (layout_str=="Node Scalar")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->node_scalar);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->node_scalar);

    layout = NODE_SCALAR;
  }
  else if (layout_str=="Node Vector")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->node_vector);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->node_vector);

    layout = NODE_VECTOR;
  }
  else if (layout_str=="Node Tensor")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->node_tensor);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->node_tensor);

    layout = NODE_TENSOR;
  }
  else if (layout_str=="Vertex Vector")
  {
    val_cell = PHX::MDField<ScalarT>(p.get<std::string> ("Cell Variable Name"), dl->vertices_vector);
    val_side = PHX::MDField<ScalarT>(p.get<std::string> ("Side Variable Name"), dl_side->vertices_vector);

    layout = VERTEX_VECTOR;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid field layout.\n");
  }

  val_side.dimensions(dims);
  this->addDependentField(val_side);
  this->addEvaluatedField(val_cell);

  this->setName("DOFSideToCell");

  if (layout==NODE_SCALAR || layout==NODE_VECTOR || layout==NODE_TENSOR || layout==VERTEX_VECTOR)
  {
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

    int sideDim = dl_side->cell_gradient->dimension(2);
    sideNodes.resize(dims[1]);
    for (int side=0; side<dims[1]; ++side)
    {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node)
      {
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
      }
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFSideToCellBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_cell,fm);
  this->utils.setFieldData(val_side,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFSideToCellBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    switch (layout)
    {
      case CELL_SCALAR:
        val_cell(cell) = val_side(cell,side);
        break;

      case CELL_VECTOR:
        for (int i=0; i<dims[2]; ++i)
          val_cell(cell,i) = val_side(cell,side,i);
        break;

      case CELL_TENSOR:
        for (int i=0; i<dims[2]; ++i)
          for (int j=0; j<dims[3]; ++j)
            val_cell(cell,i,j) = val_side(cell,side,i,j);
        break;

      case NODE_SCALAR:
        for (int node=0; node<dims[2]; ++node)
          val_cell(cell,sideNodes[side][node]) = val_side(cell,side,node);
        break;

      case NODE_VECTOR:
      case VERTEX_VECTOR:
        for (int node=0; node<dims[2]; ++node)
          for (int i=0; i<dims[3]; ++i)
            val_cell(cell,sideNodes[side][node],i) = val_side(cell,side,node,i);
        break;
      case NODE_TENSOR:
        for (int node=0; node<dims[2]; ++node)
          for (int i=0; i<dims[3]; ++i)
            for (int j=0; j<dims[4]; ++j)
              val_cell(cell,sideNodes[side][node],i,j) = val_side(cell,side,node,i,j);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid layout (this error should have happened earlier though).\n");
    }
  }
}

} // Namespace PHAL
