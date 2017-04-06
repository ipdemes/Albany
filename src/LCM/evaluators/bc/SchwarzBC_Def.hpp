//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "MiniTensor.h"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

#if defined(ALBANY_DTK)
#include "Albany_OrdinarySTKFieldContainer.hpp"
#endif

//
// Generic Template Code for Constructor and PostRegistrationSetup
//

namespace LCM {

//
//
//
template<typename EvalT, typename Traits>
SchwarzBC_Base<EvalT, Traits>::
SchwarzBC_Base(Teuchos::ParameterList & p) :
    PHAL::DirichletBase<EvalT, Traits>(p),
    app_(p.get<Teuchos::RCP<Albany::Application>>(
        "Application", Teuchos::null)),
    coupled_apps_(app_->getApplications()),
    coupled_app_name_(p.get<std::string>("Coupled Application", "SELF")),
    coupled_block_name_(p.get<std::string>("Coupled Block", "NONE"))
{
  std::string const &
  nodeset_name = this->nodeSetID;

  app_->setCoupledAppBlockNodeset(
      coupled_app_name_,
      coupled_block_name_,
      nodeset_name);

  std::string const &
  this_app_name = app_->getAppName();

  auto const &
  app_name_index_map = *(app_->getAppNameIndexMap());

  auto
  it = app_name_index_map.find(this_app_name);

  ALBANY_EXPECT(it != app_name_index_map.end());

  auto const
  this_app_index = it->second;

  setThisAppIndex(this_app_index);

  it = app_name_index_map.find(coupled_app_name_);

  ALBANY_EXPECT(it != app_name_index_map.end());

  auto const
  coupled_app_index = it->second;

  setCoupledAppIndex(coupled_app_index);
}

//
//
//
template<typename EvalT, typename Traits>
template<typename T>
void
SchwarzBC_Base<EvalT, Traits>::
computeBCs(size_t const ns_node, T & x_val, T & y_val, T & z_val)
{
  auto const
  coupled_app_index = getCoupledAppIndex();

  Albany::Application const &
  coupled_app = getApplication(coupled_app_index);

  Teuchos::RCP<Tpetra_Vector const>
  coupled_solution = coupled_app.getX();

  if (coupled_solution == Teuchos::null) {
    x_val = 0.0;
    y_val = 0.0;
    z_val = 0.0;
    return;
  }

  auto const
  this_app_index = getThisAppIndex();

  Albany::Application const &
  this_app = getApplication(this_app_index);

  Teuchos::RCP<Albany::AbstractDiscretization>
  this_disc = this_app.getDiscretization();

  auto *
  this_stk_disc = static_cast<Albany::STKDiscretization *>(this_disc.get());

  Teuchos::RCP<Albany::AbstractDiscretization>
  coupled_disc = coupled_app.getDiscretization();

  auto *
  coupled_stk_disc =
      static_cast<Albany::STKDiscretization *>(coupled_disc.get());

  auto &
  coupled_gms = dynamic_cast<Albany::GenericSTKMeshStruct &>
      (*(coupled_stk_disc->getSTKMeshStruct()));

  auto const &
  coupled_ws_eb_names = coupled_disc->getWsEBNames();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
  coupled_mesh_specs = coupled_gms.getMeshSpecs();

  // Get cell topology of the application and block to which this node set
  // is coupled.
  std::string const &
  this_app_name = this_app.getAppName();

  std::string const &
  coupled_app_name = coupled_app.getAppName();

  std::string const
  coupled_block_name = this_app.getCoupledBlockName(coupled_app_index);

  bool const
  use_block = coupled_block_name != "NONE";

  std::map<std::string, int> const &
  coupled_block_name_to_index = coupled_gms.ebNameToIndex;

  auto
  it = coupled_block_name_to_index.find(coupled_block_name);

  bool const
  missing_block = it == coupled_block_name_to_index.end();

  if (use_block == true && missing_block == true) {
    std::cerr << "\nERROR: " << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "Unknown coupled block: " << coupled_block_name << '\n';
    std::cerr << "Coupling application : " << this_app_name << '\n';
    std::cerr << "To application       : " << coupled_app_name << '\n';
    exit(1);
  }

  // When ignoring the block, set the index to zero to get defaults
  // corresponding to the first block.
  auto const
  coupled_block_index = use_block == true ? it->second : 0;

  CellTopologyData const
  coupled_cell_topology_data = coupled_mesh_specs[coupled_block_index]->ctd;

  shards::CellTopology
  coupled_cell_topology(&coupled_cell_topology_data);

  auto const
  coupled_dimension = coupled_cell_topology_data.dimension;

  auto const
  coupled_node_count = coupled_cell_topology_data.node_count;

  std::string const &
  coupled_nodeset_name = this_app.getNodesetName(coupled_app_index);

  std::vector<double *> const &
  ns_coord =
      this_stk_disc->getNodeSetCoords().find(coupled_nodeset_name)->second;

  auto const &
  ws_elem_to_node_id = coupled_stk_disc->getWsElNodeID();

  std::vector<minitensor::Vector<double>>
  coupled_element_nodes(coupled_node_count);

  std::vector<minitensor::Vector<double>>
  coupled_element_solution(coupled_node_count);

  for (auto i = 0; i < coupled_node_count; ++i) {
    coupled_element_nodes[i].set_dimension(coupled_dimension);
    coupled_element_solution[i].set_dimension(coupled_dimension);
  }

  // This tolerance is used for geometric approximations. It will be used
  // to determine whether a node of this_app is inside an element of
  // coupled_app within that tolerance.
  double const
  tolerance = 5.0e-2;

  auto const
  parametric_dimension = coupled_dimension;

  auto const
  coupled_vertex_count = coupled_cell_topology_data.vertex_count;

  auto const
  coupled_element_type =
        minitensor::find_type(coupled_dimension, coupled_vertex_count);

  minitensor::Vector<double>
  lo(parametric_dimension, minitensor::ONES);

  minitensor::Vector<double>
  hi(parametric_dimension, minitensor::ONES);

  hi = hi * (1.0 + tolerance);

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
  basis;

  switch (coupled_element_type) {

  default:
    MT_ERROR_EXIT("Unknown element type");
    break;

  case minitensor::ELEMENT::TETRAHEDRAL:
    basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_TET_C1_FEM<PHX::Device>());
    lo = - tolerance * lo;
    break;

  case minitensor::ELEMENT::HEXAHEDRAL:
    basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device>());
    lo = - lo * (1.0 + tolerance);
    break;
  }

  double * const
  coord = ns_coord[ns_node];

  minitensor::Vector<double>
  point;

  point.set_dimension(coupled_dimension);

  point.fill(coord);

  // Determine the element that contains this point.
  Teuchos::ArrayRCP<double> const &
  coupled_coordinates = coupled_stk_disc->getCoordinates();

  Teuchos::ArrayRCP<ST const>
  coupled_solution_view = coupled_solution->get1dView();

  Teuchos::RCP<Tpetra_Map const>
  coupled_overlap_node_map = coupled_stk_disc->getOverlapNodeMapT();

  // We do this element by element
  auto const
  number_cells = 1;

  // We do this point by point
  auto const
  number_points = 1;

  // Container for the parametric coordinates
  Kokkos::DynRankView<RealType, PHX::Device>
  parametric_point(
      "par_point",
      number_cells,
      number_points,
      parametric_dimension);

  for (auto j = 0; j < parametric_dimension; ++j) {
    parametric_point(0, 0, j) = 0.0;
  }

  // Container for the physical point
  Kokkos::DynRankView<RealType, PHX::Device>
  physical_coordinates(
      "phys_point",
      number_cells,
      number_points,
      coupled_dimension);

  for (auto i = 0; i < coupled_dimension; ++i) {
    physical_coordinates(0, 0, i) = point(i);
  }

  // Container for the physical nodal coordinates
  Kokkos::DynRankView<RealType, PHX::Device>
  nodal_coordinates(
      "coords",
      number_cells,
      coupled_node_count,
      coupled_dimension);

  bool
  found = false;

  for (auto workset = 0; workset < ws_elem_to_node_id.size(); ++workset) {

    std::string const &
    coupled_element_block = coupled_ws_eb_names[workset];

    bool const
    block_names_differ = coupled_element_block != coupled_block_name;

    if (use_block == true && block_names_differ == true) continue;

    auto const
    elements_per_workset = ws_elem_to_node_id[workset].size();

    for (auto element = 0; element < elements_per_workset; ++element) {

      for (auto node = 0; node < coupled_node_count; ++node) {

        auto const
        global_node_id = ws_elem_to_node_id[workset][element][node];

        auto const
        local_node_id =
            coupled_overlap_node_map->getLocalElement(global_node_id);

        double * const
        pcoord = &(coupled_coordinates[coupled_dimension * local_node_id]);

        coupled_element_nodes[node].fill(pcoord);

        for (auto i = 0; i < coupled_dimension; ++i) {
          coupled_element_solution[node](i) =
              coupled_solution_view[coupled_dimension * local_node_id + i];
        } // dimension loop

      } // node loop

      for (auto i = 0; i < coupled_node_count; ++i) {
        for (auto j = 0; j < coupled_dimension; ++j) {
          nodal_coordinates(0, i, j) = coupled_element_nodes[i](j);
        }
      }

      // Get parametric coordinates
      Intrepid2::CellTools<PHX::Device>::mapToReferenceFrame(
          parametric_point,
          physical_coordinates,
          nodal_coordinates,
          coupled_cell_topology);

      bool
      in_element = true;

      for (auto i = 0; i < parametric_dimension; ++i) {
        auto const
        xi = parametric_point(0, 0, i);
        in_element = in_element && lo(i) <= xi && xi <= hi(i);
      }

      if (in_element == true) {
        found = true;
        break;
      }

    } // element loop

    if (found == true) {
      break;
    }

  } // workset loop

  ALBANY_EXPECT(found == true);

  // Evaluate shape functions at parametric point.
  Kokkos::DynRankView<RealType, PHX::Device>
  basis_values("basis", coupled_node_count, number_points);

  // Another container for the parametric coordinates. Needed because above
  // it is required that parametric_points has rank 3 for mapToReferenceFrame
  // but here basis->getValues requires a rank 2 view :(
  Kokkos::DynRankView<RealType, PHX::Device>
  pp_reduced("par_point", number_points, parametric_dimension);

  for (auto j = 0; j < parametric_dimension; ++j) {
    pp_reduced(0, j) = parametric_point(0, 0, j);
  }
  basis->getValues(basis_values, pp_reduced, Intrepid2::OPERATOR_VALUE);

  // Evaluate solution at parametric point using values of shape
  // functions just computed.
  minitensor::Vector<double>
  value(coupled_dimension, minitensor::ZEROS);

  for (auto i = 0; i < coupled_node_count; ++i) {
    value += basis_values(i, 0) * coupled_element_solution[i];
  }

  x_val = value(0);
  y_val = value(1);
  z_val = value(2);

  return;
}

//
//
//
#if defined(ALBANY_DTK)
template<typename EvalT, typename Traits>
Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
SchwarzBC_Base<EvalT, Traits>::
computeBCsDTK()
{
  auto const
  this_app_index = getThisAppIndex();

  auto const
  coupled_app_index = getCoupledAppIndex();

  Albany::Application const &
  this_app = getApplication(this_app_index);

  Albany::Application const &
  coupled_app = getApplication(coupled_app_index);

  // neq should be the same for this_app and coupled_app.
  ALBANY_EXPECT(this_app.getNumEquations() == coupled_app.getNumEquations());

  //Get number of equations from this_app 
  int neq = this_app.getNumEquations();

  //this_disc = target mesh
  Teuchos::RCP<Albany::AbstractDiscretization>
  this_disc = this_app.getDiscretization();

  auto *
  this_stk_disc = static_cast<Albany::STKDiscretization *>(this_disc.get());

  //coupled_disc = source mesh
  Teuchos::RCP<Albany::AbstractDiscretization>
  coupled_disc = coupled_app.getDiscretization();

  auto *
  coupled_stk_disc =
      static_cast<Albany::STKDiscretization *>(coupled_disc.get());

  //Source Mesh
  Teuchos::RCP<Albany::AbstractSTKMeshStruct> const
  coupled_stk_mesh_struct = coupled_stk_disc->getSTKMeshStruct();

  //get pointer to metadata from coupled_stk_disc
  Teuchos::RCP<stk::mesh::MetaData const> const
  coupled_meta_data = Teuchos::rcpFromRef(coupled_stk_disc->getSTKMetaData());

  //Get coupled_app parameter list 
  Teuchos::RCP<const Teuchos::ParameterList>
  coupled_app_params = coupled_app.getAppPL();

  //Get discretization sublist from coupled_app parameter list
  Teuchos::ParameterList
  dtk_params = coupled_app_params->sublist("DataTransferKit");

  //Get solution name from Discretization sublist
  std::string map_name =
      dtk_params.get("Map Type", "Consistent Interpolation");

  Albany::AbstractSTKFieldContainer::VectorFieldType*
  coupled_field =
      Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true>>(
          coupled_stk_disc->getSTKMeshStruct()->getFieldContainer()
          )->getSolutionField();

  stk::mesh::Selector
  coupled_stk_selector =
      stk::mesh::Selector(coupled_meta_data->universal_part());

  Teuchos::RCP<stk::mesh::BulkData>
  coupled_bulk_data = Teuchos::rcpFromRef(coupled_field->get_mesh());

  //Target Mesh

  //get pointer to metadata from this_stk_disc
  Teuchos::RCP<stk::mesh::MetaData const>
  this_meta_data = Teuchos::rcpFromRef(this_stk_disc->getSTKMetaData());

  Albany::AbstractSTKFieldContainer::VectorFieldType*
  this_field =
      Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer<true>>(
          this_stk_disc->getSTKMeshStruct()->getFieldContainer()
          )->getSolutionFieldDTK();

  // Get the part corresponding to this nodeset.
  std::string const &
  nodeset_name = this->nodeSetID;

  stk::mesh::Part *
  this_part = this_meta_data->get_part(nodeset_name);

  Teuchos::RCP<stk::mesh::BulkData>
  this_bulk_data = Teuchos::rcpFromRef(this_field->get_mesh());

  //Solution Transfer Setup

  // Create a manager for the source part elements.
  DataTransferKit::STKMeshManager
  coupled_manager(coupled_bulk_data, coupled_stk_selector);

  // Create a manager for the target part nodes.
  stk::mesh::Selector
  this_stk_selector(*this_part);

  DataTransferKit::STKMeshManager
  this_manager(this_bulk_data, this_stk_selector);

  // Create a solution vector for the source.
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
  coupled_vector =
      coupled_manager.createFieldMultiVector <
          Albany::AbstractSTKFieldContainer::VectorFieldType
          > (Teuchos::ptr(coupled_field), neq);

  // Create a solution vector for the target.
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
  this_vector =
      this_manager.createFieldMultiVector <
          Albany::AbstractSTKFieldContainer::VectorFieldType
          > (Teuchos::ptr(this_field), neq);

  //Solution transfer

  DataTransferKit::MapOperatorFactory
  op_factory;

  Teuchos::RCP < DataTransferKit::MapOperator >
      map_op =
      op_factory.create(coupled_vector->getMap(),
          this_vector->getMap(),
          dtk_params);

  // Setup the map operator. This creates the underlying linear operators.
  map_op->setup(coupled_manager.functionSpace(), this_manager.functionSpace());

  // Apply the map operator. This interpolates the data from one STK field
  // to the other.
  map_op->apply(*coupled_vector, *this_vector);

  return this_vector;
}
#endif //ALBANY_DTK

//
// Fill residual, used in both residual and Jacobian
//
template<typename SchwarzBC, typename Traits>
void
fillResidual(SchwarzBC & sbc, typename Traits::EvalData dirichlet_workset)
{
  // Solution
  Teuchos::RCP<Tpetra_Vector const>
  xT = dirichlet_workset.xT;

  Teuchos::ArrayRCP<ST const>
  xT_const_view = xT->get1dView();

  // Residual
  Teuchos::RCP<Tpetra_Vector>
  fT = dirichlet_workset.fT;

  Teuchos::ArrayRCP<ST>
  fT_view = fT->get1dViewNonConst();

  std::vector<std::vector<int>> const &
  ns_dof = dirichlet_workset.nodeSets->find(sbc.nodeSetID)->second;

  auto const
  ns_number_nodes = ns_dof.size();

#if defined(ALBANY_DTK)

  Teuchos::RCP<
      Tpetra::MultiVector<double, int, DataTransferKit::SupportId>
  > const
  schwarz_bcs = sbc.computeBCsDTK();

  Teuchos::RCP<const Teuchos::Comm<int>>
  commT = schwarz_bcs->getMap()->getComm();

  Teuchos::ArrayRCP<ST const>
  schwarz_bcs_const_view_x = schwarz_bcs->getData(0);

  Teuchos::ArrayRCP<ST const>
  schwarz_bcs_const_view_y = schwarz_bcs->getData(1);

  Teuchos::ArrayRCP<ST const>
  schwarz_bcs_const_view_z = schwarz_bcs->getData(2);

  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    auto const
    x_dof = ns_dof[ns_node][0];

    auto const
    y_dof = ns_dof[ns_node][1];

    auto const
    z_dof = ns_dof[ns_node][2];

    auto const
    dof = x_dof / 3;

    std::set<int> const &
    fixed_dofs = dirichlet_workset.fixed_dofs_;

    if (fixed_dofs.find(x_dof) == fixed_dofs.end()) {
      fT_view[x_dof] = xT_const_view[x_dof] - schwarz_bcs_const_view_x[dof];
    }
    if (fixed_dofs.find(y_dof) == fixed_dofs.end()) {
      fT_view[y_dof] = xT_const_view[y_dof] - schwarz_bcs_const_view_y[dof];
    }
    if (fixed_dofs.find(z_dof) == fixed_dofs.end()) {
      fT_view[z_dof] = xT_const_view[z_dof] - schwarz_bcs_const_view_z[dof];
    }

  }
#else // ALBANY_DTK
  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    ST
    x_val, y_val, z_val;

    sbc.computeBCs(ns_node, x_val, y_val, z_val);

    auto const
    x_dof = ns_dof[ns_node][0];

    auto const
    y_dof = ns_dof[ns_node][1];

    auto const
    z_dof = ns_dof[ns_node][2];

    std::set<int> const &
    fixed_dofs = dirichlet_workset.fixed_dofs_;

    if (fixed_dofs.find(x_dof) == fixed_dofs.end()) {
      fT_view[x_dof] = xT_const_view[x_dof] - x_val;
    }
    if (fixed_dofs.find(y_dof) == fixed_dofs.end()) {
      fT_view[y_dof] = xT_const_view[y_dof] - y_val;
    }
    if (fixed_dofs.find(z_dof) == fixed_dofs.end()) {
      fT_view[z_dof] = xT_const_view[z_dof] - z_val;
    }

  } // node in node set loop
#endif //ALBANY_DTK
  return;
}

//
// Specialization: Residual
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  fillResidual<SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>, Traits>
  (*this, dirichlet_workset);
  return;
}

//
// Specialization: Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_Vector>
  fT = dirichlet_workset.fT;

  Teuchos::ArrayRCP<ST>
  fT_view;

  Teuchos::RCP<Tpetra_CrsMatrix>
  jacT = dirichlet_workset.JacT;

  Teuchos::RCP<Tpetra_Vector const>
  xT = dirichlet_workset.xT;

  Teuchos::ArrayRCP<ST const>
  xT_const_view = xT->get1dView();

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::Array<LO>
  index(1);

  Teuchos::Array<ST>
  value(1);

  value[0] = j_coeff;

  Teuchos::Array<ST>
  matrix_entries;

  Teuchos::Array<LO>
  matrix_indices;

  bool const
  fill_residual = (fT != Teuchos::null);

  if (fill_residual == true) {
    fT_view = fT->get1dViewNonConst();
  }

  for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {

    auto const
    x_dof = ns_nodes[ns_node][0];

    auto const
    y_dof = ns_nodes[ns_node][1];

    auto const
    z_dof = ns_nodes[ns_node][2];

    std::set<int> const &
    fixed_dofs = dirichlet_workset.fixed_dofs_;

    if (fixed_dofs.find(x_dof) == fixed_dofs.end()) {
      // replace jac values for the X dof
      auto
      num_entries = jacT->getNumEntriesInLocalRow(x_dof);

      matrix_entries.resize(num_entries);
      matrix_indices.resize(num_entries);

      jacT->getLocalRowCopy(
          x_dof,
          matrix_indices(),
          matrix_entries(),
          num_entries);

      for (auto i = 0; i < num_entries; ++i) {
        matrix_entries[i] = 0;
      }

      jacT->replaceLocalValues(x_dof, matrix_indices(), matrix_entries());
      index[0] = x_dof;
      jacT->replaceLocalValues(x_dof, index(), value());
    }

    if (fixed_dofs.find(y_dof) == fixed_dofs.end()) {
      // replace jac values for the y dof
      auto
      num_entries = jacT->getNumEntriesInLocalRow(y_dof);

      matrix_entries.resize(num_entries);
      matrix_indices.resize(num_entries);

      jacT->getLocalRowCopy(
          y_dof,
          matrix_indices(),
          matrix_entries(),
          num_entries);

      for (auto i = 0; i < num_entries; ++i) {
        matrix_entries[i] = 0;
      }

      jacT->replaceLocalValues(y_dof, matrix_indices(), matrix_entries());
      index[0] = y_dof;
      jacT->replaceLocalValues(y_dof, index(), value());
    }

    if (fixed_dofs.find(z_dof) == fixed_dofs.end()) {
      // replace jac values for the z dof
      auto
      num_entries = jacT->getNumEntriesInLocalRow(z_dof);

      matrix_entries.resize(num_entries);
      matrix_indices.resize(num_entries);

      jacT->getLocalRowCopy(
          z_dof,
          matrix_indices(),
          matrix_entries(),
          num_entries);

      for (auto i = 0; i < num_entries; ++i) {
        matrix_entries[i] = 0;
      }

      jacT->replaceLocalValues(z_dof, matrix_indices(), matrix_entries());
      index[0] = z_dof;
      jacT->replaceLocalValues(z_dof, index(), value());
    }
  }

  if (fill_residual == true) {
    fillResidual<
    SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>, Traits>
    (*this, dirichlet_workset);
  }
  return;
}

//
// Specialization: Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_Vector>
  fT = dirichlet_workset.fT;

  Teuchos::RCP<Tpetra_MultiVector>
  fpT = dirichlet_workset.fpT;

  Teuchos::RCP<Tpetra_MultiVector>
  JVT = dirichlet_workset.JVT;

  Teuchos::RCP<Tpetra_Vector const>
  xT = dirichlet_workset.xT;

  Teuchos::RCP<Tpetra_MultiVector const>
  VxT = dirichlet_workset.VxT;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::ArrayRCP<ST const>
  VxT_const_view;

  Teuchos::ArrayRCP<ST>
  fT_view;

  Teuchos::ArrayRCP<ST const>
  xT_const_view = xT->get1dView();

  if (fT != Teuchos::null) {
    fT_view = fT->get1dViewNonConst();
  }

  for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {

    auto const
    x_dof = ns_nodes[ns_node][0];

    auto const
    y_dof = ns_nodes[ns_node][1];

    auto const
    z_dof = ns_nodes[ns_node][2];

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST>
      JVT_view;

      for (auto i = 0; i < dirichlet_workset.num_cols_x; ++i) {
        JVT_view = JVT->getDataNonConst(i);
        VxT_const_view = VxT->getData(i);
        JVT_view[x_dof] = j_coeff * VxT_const_view[x_dof];
        JVT_view[y_dof] = j_coeff * VxT_const_view[y_dof];
        JVT_view[z_dof] = j_coeff * VxT_const_view[z_dof];
      }
    }
  }

  if (fT != Teuchos::null || fpT != Teuchos::null) {

#if defined(ALBANY_DTK)
    if (fT != Teuchos::null) {

      Teuchos::RCP<
          Tpetra::MultiVector<double, int, DataTransferKit::SupportId>
      > const
      schwarz_bcs = this->computeBCsDTK();

      Teuchos::RCP<const Teuchos::Comm<int>>
      commT = schwarz_bcs->getMap()->getComm();

      Teuchos::ArrayRCP<ST const>
      schwarz_bcs_const_view_x = schwarz_bcs->getData(0);

      Teuchos::ArrayRCP<ST const>
      schwarz_bcs_const_view_y = schwarz_bcs->getData(1);

      Teuchos::ArrayRCP<ST const>
      schwarz_bcs_const_view_z = schwarz_bcs->getData(2);

      for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {

        auto const
        x_dof = ns_nodes[ns_node][0];

        auto const
        y_dof = ns_nodes[ns_node][1];

        auto const
        z_dof = ns_nodes[ns_node][2];

        auto const
        dof = x_dof / 3;

        fT_view[x_dof] = xT_const_view[x_dof] - schwarz_bcs_const_view_x[dof];
        fT_view[y_dof] = xT_const_view[y_dof] - schwarz_bcs_const_view_y[dof];
        fT_view[z_dof] = xT_const_view[z_dof] - schwarz_bcs_const_view_z[dof];
      }
    }

    if (fpT != Teuchos::null) {
      std::cout << "WARNING: fpT requested but unset when ALBANY_DTK is ON!\n";
    }
#else  
    for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {

      auto const
      x_dof = ns_nodes[ns_node][0];

      auto const
      y_dof = ns_nodes[ns_node][1];

      auto const
      z_dof = ns_nodes[ns_node][2];

      ScalarT
      x_val, y_val, z_val;

      this->computeBCs(ns_node, x_val, y_val, z_val);

      if (fT != Teuchos::null) {
        fT_view[x_dof] = xT_const_view[x_dof] - x_val.val();
        fT_view[y_dof] = xT_const_view[y_dof] - y_val.val();
        fT_view[z_dof] = xT_const_view[z_dof] - z_val.val();
      }
      if (fpT != Teuchos::null) {
        Teuchos::ArrayRCP<ST>
        fpT_view;

        for (auto i = 0; i < dirichlet_workset.num_cols_p; ++i) {
          fpT_view = fpT->getDataNonConst(i);
          fpT_view[x_dof] = -x_val.dx(dirichlet_workset.param_offset + i);
          fpT_view[y_dof] = -y_val.dx(dirichlet_workset.param_offset + i);
          fpT_view[z_dof] = -z_val.dx(dirichlet_workset.param_offset + i);
        }
      }
    }
#endif //ALBANY_DTK
  }
  return;
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_MultiVector>
  fpVT = dirichlet_workset.fpVT;

  Teuchos::ArrayRCP<ST>
  fpVT_view;

  bool const
  trans = dirichlet_workset.transpose_dist_param_deriv;

  auto const
  num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double *> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans == true) {
    Teuchos::RCP<Tpetra_MultiVector>
    VpT = dirichlet_workset.Vp_bcT;

    Teuchos::ArrayRCP<ST>
    VpT_view;

    for (auto inode = 0; inode < ns_nodes.size(); ++inode) {

      auto const
      x_dof = ns_nodes[inode][0];

      auto const
      y_dof = ns_nodes[inode][1];

      auto const
      z_dof = ns_nodes[inode][2];

      for (auto col = 0; col < num_cols; ++col) {

        VpT_view = VpT->getDataNonConst(col);
        VpT_view[x_dof] = 0.0;
        VpT_view[y_dof] = 0.0;
        VpT_view[z_dof] = 0.0;
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (auto inode = 0; inode < ns_nodes.size(); ++inode) {

      auto const
      x_dof = ns_nodes[inode][0];

      auto const
      y_dof = ns_nodes[inode][1];

      auto const
      z_dof = ns_nodes[inode][2];

      for (auto col = 0; col < num_cols; ++col) {

        fpVT_view = fpVT->getDataNonConst(col);
        fpVT_view[x_dof] = 0.0;
        fpVT_view[y_dof] = 0.0;
        fpVT_view[z_dof] = 0.0;
      }
    }
  }
}

//
// Specialization: Stochastic Galerkin Residual
//
#if defined(ALBANY_SG)
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGResidual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>
  f = dirichlet_workset.sg_f;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
  x = dirichlet_workset.sg_x;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  int const
  nblock = x->size();

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(ns_node, x_val, y_val, z_val);

    for (int block = 0; block < nblock; ++block) {
      (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.coeff(block);
      (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.coeff(block);
      (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.coeff(block);
    }
  }
}

//
// Specialization: Stochastic Galerkin Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly>
  f = dirichlet_workset.sg_f;

  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>>
  jac = dirichlet_workset.sg_Jac;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
  x = dirichlet_workset.sg_x;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType *
  matrix_entries;

  int *
  matrix_indices;

  int
  num_entries;

  RealType
  diag = j_coeff;

  bool
  fill_residual = (f != Teuchos::null);

  int
  nblock = 0;

  if (f != Teuchos::null) {
    nblock = f->size();
  }

  int const
  nblock_jac = jac->size();

  // local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(ns_node, x_val, y_val, z_val);

    // replace jac values for the X dof
    for (int block = 0; block < nblock_jac; ++block) {
      (*jac)[block].ExtractMyRowView(x_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(y_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;

      // replace jac values for the z dof
      (*jac)[block].ExtractMyRowView(z_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
    }

    (*jac)[0].ReplaceMyValues(x_dof, 1, &diag, &x_dof);
    (*jac)[0].ReplaceMyValues(y_dof, 1, &diag, &y_dof);
    (*jac)[0].ReplaceMyValues(z_dof, 1, &diag, &z_dof);

    if (fill_residual == true) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }
  }
}

//
// Specialization: Stochastic Galerkin Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGTangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>
  f = dirichlet_workset.sg_f;

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly>
  fp = dirichlet_workset.sg_fp;

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly>
  JV = dirichlet_workset.sg_JV;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
  x = dirichlet_workset.sg_x;

  Teuchos::RCP<Epetra_MultiVector const>
  Vx = dirichlet_workset.Vx;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  int const
  nblock = x->size();

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(ns_node, x_val, y_val, z_val);

    if (f != Teuchos::null) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i = 0; i < dirichlet_workset.num_cols_x; ++i) {
        (*JV)[0][i][x_dof] = j_coeff*(*Vx)[i][x_dof];
        (*JV)[0][i][y_dof] = j_coeff*(*Vx)[i][y_dof];
        (*JV)[0][i][z_dof] = j_coeff*(*Vx)[i][z_dof];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; ++i) {
        for (int block = 0; block < nblock; ++block) {
          (*fp)[block][i][x_dof] =
          -x_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][y_dof] =
          -y_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][z_dof] =
          -z_val.dx(dirichlet_workset.param_offset+i).coeff(block);
        }
      }
    }

  }
}
#endif // ALBANY_SG

#if defined(ALBANY_ENSEMBLE)
//
// Specialization: Multi-point Residual
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPResidual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>
  f = dirichlet_workset.mp_f;

  Teuchos::RCP<Stokhos::ProductEpetraVector const>
  x = dirichlet_workset.mp_x;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  int const
  nblock = x->size();

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(ns_node, x_val, y_val, z_val);

    for (int block = 0; block < nblock; ++block) {
      (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.coeff(block);
      (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.coeff(block);
      (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.coeff(block);
    }
  }
}

//
// Specialization: Multi-point Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>
  f = dirichlet_workset.mp_f;

  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix>>
  jac = dirichlet_workset.mp_Jac;

  Teuchos::RCP<Stokhos::ProductEpetraVector const>
  x = dirichlet_workset.mp_x;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType *
  matrix_entries;

  int *
  matrix_indices;

  int
  num_entries;

  RealType
  diag = j_coeff;

  bool
  fill_residual = (f != Teuchos::null);

  int
  nblock = 0;

  if (f != Teuchos::null) {
    nblock = f->size();
  }

  int const
  nblock_jac = jac->size();

  // local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(ns_node, x_val, y_val, z_val);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; ++block) {
      (*jac)[block].ExtractMyRowView(x_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
      (*jac)[block].ReplaceMyValues(x_dof, 1, &diag, &x_dof);

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(y_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
      (*jac)[block].ReplaceMyValues(y_dof, 1, &diag, &y_dof);

      // replace jac values for the z dof
      (*jac)[block].ExtractMyRowView(z_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
      (*jac)[block].ReplaceMyValues(z_dof, 1, &diag, &z_dof);
    }

    if (fill_residual == true) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }
  }
}

//
// Specialization: Multi-point Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPTangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>
  f = dirichlet_workset.mp_f;

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector>
  fp = dirichlet_workset.mp_fp;

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector>
  JV = dirichlet_workset.mp_JV;

  Teuchos::RCP<Stokhos::ProductEpetraVector const>
  x = dirichlet_workset.mp_x;

  Teuchos::RCP<Epetra_MultiVector const>
  Vx = dirichlet_workset.Vx;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  int const
  nblock = x->size();

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(ns_node, x_val, y_val, z_val);

    if (f != Teuchos::null) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i = 0; i<dirichlet_workset.num_cols_x; ++i) {
        for (int block = 0; block < nblock; ++block) {
          (*JV)[block][i][x_dof] = j_coeff*(*Vx)[i][x_dof];
          (*JV)[block][i][y_dof] = j_coeff*(*Vx)[i][y_dof];
          (*JV)[block][i][z_dof] = j_coeff*(*Vx)[i][z_dof];
        }
      }
    }

    if (fp != Teuchos::null) {

      for (int i = 0; i < dirichlet_workset.num_cols_p; ++i) {
        for (int block = 0; block < nblock; ++block) {
          (*fp)[block][i][x_dof] =
          -x_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][y_dof] =
          -y_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][z_dof] =
          -z_val.dx(dirichlet_workset.param_offset+i).coeff(block);
        }
      }
    }

  }
}
#endif // ALBANY_ENSEMBLE

} // namespace LCM
