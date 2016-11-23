//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_SimLayerAdapt.hpp"
#include "Albany_SimDiscretization.hpp"
#include <MeshSimAdapt.h>
#include <SimPartitionedMesh.h>
#include <SimField.h>
/* BRD */
#include <SimModel.h>
/* BRD */
#include <apfSIM.h>
#include <spr.h>
#include <EnergyIntegral.hpp>

/* BRD */
#include "PHAL_AlbanyTraits.hpp"
extern void DM_undoSlicing(pPList regions,int layerNum, pMesh mesh);
extern void PM_localizePartiallyConnected(pParMesh);
extern void MSA_setPrebalance(pMSAdapt,int);
/* BRD */

namespace AAdapt {

enum { ABSOLUTE = 1, RELATIVE = 2 };
enum { DONT_GRADE = 0, DO_GRADE = 1 };
enum { ONLY_CURV_TYPE = 2 };

SimLayerAdapt::SimLayerAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const Albany::StateManager& StateMgr_,
                   const Teuchos::RCP<const Teuchos_Comm>& commT_):
  AbstractAdapterT(params_, paramLib_, StateMgr_, commT_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  errorBound = params_->get<double>("Error Bound", 0.1);
  // get inititial temperature for new added layer
  initTempNewLayer = params_->get<double>("Uniform Temperature New Layer", 20.0);

  // Tell user that Uniform temperature is in effect
  *out << "***********************" << std::endl;
  *out << "Uniform Temperature New Layer = " << initTempNewLayer << std::endl;
  *out << "***********************" << std::endl;

  /* BRD */
  Simmetrix_numLayers = -1;
  Simmetrix_currentLayer = 0;
  Simmetrix_model = 0;
  /* BRD */
}

bool SimLayerAdapt::queryAdaptationCriteria(int iteration)
{
  /* BRD */
  if (!Simmetrix_model) {
    Teuchos::RCP<Albany::AbstractDiscretization> disc =
      state_mgr_.getDiscretization();
    Teuchos::RCP<Albany::SimDiscretization> sim_disc =
      Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
    Teuchos::RCP<Albany::APFMeshStruct> apf_ms =
      sim_disc->getAPFMeshStruct();
    apf::Mesh* apf_m = apf_ms->getMesh();
    apf::MeshSIM* apf_msim = dynamic_cast<apf::MeshSIM*>(apf_m);
    pParMesh sim_pm = apf_msim->getMesh();
    Simmetrix_model =  M_model(sim_pm);
    computeLayerTimes();
  }
  double currentTime = param_lib_->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  *out << "queryAdaptationCriteria\n";
  *out << "currentTime " << currentTime << '\n';
  *out << "Simmetrix_currentLayer " << Simmetrix_currentLayer << '\n';
  *out << "Simmetrix_numLayers " << Simmetrix_numLayers << '\n';
  assert(Simmetrix_currentLayer < Simmetrix_numLayers);
  if (currentTime >= Simmetrix_layerTimes[Simmetrix_currentLayer]) {
    *out << "Need to remesh and add next layer\n";
    return true;
  }
  /* BRD */
  std::string strategy = adapt_params_->get<std::string>("Remesh Strategy", "Step Number");
  if (strategy == "None")
    return false;
  if (strategy == "Continuous")
    return iteration > 1;
  if (strategy == "Step Number") {
    TEUCHOS_TEST_FOR_EXCEPTION(!adapt_params_->isParameter("Remesh Step Number"),
        std::logic_error,
        "Remesh Strategy " << strategy << " but no Remesh Step Number" << '\n');
    Teuchos::Array<int> remesh_iter = adapt_params_->get<Teuchos::Array<int> >("Remesh Step Number");
    for(int i = 0; i < remesh_iter.size(); i++)
      if(iteration == remesh_iter[i])
        return true;
    return false;
  }
  if (strategy == "Every N Step Number") {
            TEUCHOS_TEST_FOR_EXCEPTION(!adapt_params_->isParameter("Remesh Every N Step Number"),
                    std::logic_error,
                    "Remesh Strategy " << strategy << " but no Remesh Every N Step Number" << '\n');
            int remesh_iter = adapt_params_->get<int>("Remesh Every N Step Number", -1);
            // check user do not specify a zero or negative value
            TEUCHOS_TEST_FOR_EXCEPTION(remesh_iter <= 0, std::logic_error,
                    "Value must be positive" << '\n');
            if (iteration % remesh_iter == 0)
                return true;
            return false;
        }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
      "Unknown Remesh Strategy " << strategy << '\n');
  return false;
}

/* BRD */
void meshCurrentLayerOnly(pGModel model,pParMesh mesh,int currentLayer,double layerSize)
{
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model), RELATIVE, 1.0, NULL);
  MS_setMeshCurv(mcase,GM_domain(model), ONLY_CURV_TYPE, 0.025);
  MS_setMinCurvSize(mcase,GM_domain(model), ONLY_CURV_TYPE, 0.0025);
  MS_setSurfaceShapeMetric(mcase, GM_domain(model),ShapeMetricType_AspectRatio, 25);
  MS_setVolumeShapeMetric(mcase, GM_domain(model), ShapeMetricType_AspectRatio, 25);

  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int layer;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      if (layer==currentLayer) {
        MS_setMeshSize(mcase,gr, ABSOLUTE, layerSize, NULL);
        pPList regFaces = GR_faces(gr);
        void *fiter = 0;
        pGFace gf;
        while (gf = static_cast<pGFace>(PList_next(regFaces,&fiter))) {
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            GEN_nativeIntAttribute(gf,"SimLayer",&layer);
            /*
            if (layer==currentLayer+1) {
              MS_limitSurfaceMeshModification(mcase,gf,1);
              MS_useDiscreteGeometryMesh(mcase,gf,1);
            }
            */
          }
        }
        PList_delete(regFaces);
      }
      else if (layer > currentLayer)
        MS_setNoMesh(mcase,gr,1);
    }
  }
  GRIter_delete(regions);
  
  pSurfaceMesher sm = SurfaceMesher_new(mcase,mesh);
  // SurfaceMesher_setParamForDiscrete(sm, 1);
  SurfaceMesher_execute(sm,0);
  SurfaceMesher_delete(sm);
  if (currentLayer==1)
    PM_setTotalNumParts(mesh,PMU_size());
  pVolumeMesher vm  = VolumeMesher_new(mcase,mesh);
  VolumeMesher_setEnforceSize(vm, 1);
  VolumeMesher_execute(vm,0);
  VolumeMesher_delete(vm);

  MS_deleteMeshCase(mcase);
}

void addNextLayer(pParMesh sim_pm,double layerSize,int nextLayer, double initTempNewLayer,int nSolFlds,pPList flds) {
  //! Output stream, defaults to printing just Proc 0
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  
  pGModel model = M_model(sim_pm);
  
  // Collect the layer 0 regions
  GRIter regions = GM_regionIter(model);
  pGRegion gr1;
  int layer, maxLayer = -1;
  pPList combinedRegions = PList_new();
  while (gr1=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
      if (layer==0)
        PList_appUnique(combinedRegions,gr1);
      if (layer > maxLayer)
        maxLayer = layer;
    }
  }
  GRIter_delete(regions);
  if ( nextLayer > maxLayer )
    return;

  PM_localizePartiallyConnected(sim_pm);
  //PM_merge(sim_pm);
  if (nextLayer>1) {
    *out << "Combine layer " << nextLayer-1 << "\n";
    pMesh oneMesh = PM_numParts(sim_pm) == 1 ? PM_mesh(sim_pm, 0) : 0;
    DM_undoSlicing(combinedRegions,nextLayer-1,oneMesh);
  }
  PList_clear(combinedRegions);
  *out << "Mesh top layer\n";
  meshCurrentLayerOnly(model,sim_pm,nextLayer,layerSize);

  if (flds) {
    // Add temperature and residual fields to top layer
    // Add temperature HACK fields to top layer
    *out << "Add field to top layer\n";
    pField sim_sol_flds[3] = {0,0,0};  // at most 3 - see calling routine
    int i;
    for (i=0;i<nSolFlds;i++) {
      sim_sol_flds[i] = static_cast<pField>(PList_item(flds,i));
    }
    pField sim_res_fld  = static_cast<pField>(PList_item(flds,i++));
    pField sim_hak_fld = 0;
    if (PList_size(flds)==i+1)
      sim_hak_fld = static_cast<pField>(PList_item(flds,i));
    pMEntitySet topLayerVerts = MEntitySet_new(PM_mesh(sim_pm,0));
    regions = GM_regionIter(model);
    pVertex mv;
    *out << "Collect new mesh verts\n";
    while (gr1=GRIter_next(regions)) {
      if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
        GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
        if (layer == nextLayer) {
          for(int np=0;np<PM_numParts(sim_pm);np++) {
            VIter allVerts = M_classifiedVertexIter(PM_mesh(sim_pm,np),gr1,1);
            while ( mv = VIter_next(allVerts) )
              MEntitySet_add(topLayerVerts,mv);
            VIter_delete(allVerts);
          }
        }
      }
    }
    pPList unmapped = PList_new();
    pDofGroup dg;
    MESIter viter = MESIter_iter(topLayerVerts);
    //*out << "Create fields\n";
    while ( mv = reinterpret_cast<pVertex>(MESIter_next(viter)) ) {
      dg = Field_entDof(sim_sol_flds[0],mv,0);
      if (!dg) {
        PList_append(unmapped,mv);
        for(i=0;i<nSolFlds;i++)
          Field_applyEnt(sim_sol_flds[i],mv);
        Field_applyEnt(sim_res_fld,mv);
        if (sim_hak_fld)
          Field_applyEnt(sim_hak_fld,mv);
      } 
    }
    MESIter_delete(viter);
    pEntity ent;
    void *vptr;
    int c, ncs;
    int nc2 = (sim_res_fld ? Field_numComp(sim_res_fld) : 0);
    int nc3 = (sim_hak_fld ? Field_numComp(sim_hak_fld) : 0);
    void *iter = 0;
    //*out << "Set field values\n";
    while (vptr = PList_next(unmapped,&iter)) {
      ent = reinterpret_cast<pEntity>(vptr);
      for(i=0;i<nSolFlds;i++) {
        dg = Field_entDof(sim_sol_flds[i],ent,0);
        ncs = Field_numComp(sim_sol_flds[i]);
        for (c=0; c < ncs; c++)
          DofGroup_setValue(dg,c,0,initTempNewLayer);
      }
      if (sim_res_fld) {
        dg = Field_entDof(sim_res_fld,ent,0);
        for (c=0; c < nc2; c++)
          DofGroup_setValue(dg,c,0,0.0);
      }
      if (sim_hak_fld) {
        dg = Field_entDof(sim_hak_fld,ent,0);
        for (c=0; c < nc3; c++)
          DofGroup_setValue(dg,c,0,initTempNewLayer);
      }
    }
    PList_delete(unmapped);
    MEntitySet_delete(PM_mesh(sim_pm,0),topLayerVerts);
    GRIter_delete(regions);
  }
  return;
}

void SimLayerAdapt::computeLayerTimes() {

  GRIter regions = GM_regionIter(Simmetrix_model);
  pGRegion gr;
  int i, layer, maxLayer = -1;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      if (layer > maxLayer)
        maxLayer = layer;
    }
  }
  Simmetrix_numLayers = maxLayer+1;

  Simmetrix_layerTimes = new double[Simmetrix_numLayers];
  for(i=0;i<Simmetrix_numLayers;i++)
    Simmetrix_layerTimes[i] = 0.0;

  double ls = 85.0; // 0.070995;  // laser speed
  double tw = 0.013; // 0.00013;   // track width
  if (GIP_numNativeDoubleAttribute(GM_part(Simmetrix_model),"speed")==1)
    GIP_nativeDoubleAttribute(GM_part(Simmetrix_model),"speed",&ls);
  if (GIP_numNativeDoubleAttribute(GM_part(Simmetrix_model),"width")==1)
    GIP_nativeDoubleAttribute(GM_part(Simmetrix_model),"width",&tw);
  *out << "Laser speed " << ls << "\n";
  *out << "Track width " << tw << "\n";

  GRIter_reset(regions);
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      double area = 0.0;
      pPList faces = GR_faces(gr);
      pGFace gf;
      for(i=0;i<PList_size(faces);i++) {
        gf = static_cast<pGFace>(PList_item(faces,i));
        if(layer == maxLayer) {
          // top face of last layer is not tagged so
          // count faces that aren't on the boundary of the
          // previous layer.  Good enough for RoyalMess.
          pPList fregs = GF_regions(gf);
          if (PList_size(fregs)==1) {
            area += GF_area(gf,0);
          }
          PList_delete(fregs);
        }
        else {
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            int faceLayer;
            GEN_nativeIntAttribute(gf,"SimLayer",&faceLayer);
            if (faceLayer == layer+1) {
              area += GF_area(gf,0);
            }
          }
        }
      }
      PList_delete(faces);
      Simmetrix_layerTimes[layer] += area/(ls*tw);
    }
  }
  GRIter_delete(regions);

  double totalTime = Simmetrix_layerTimes[0];
  for(i=1;i < Simmetrix_numLayers; i++) {
    totalTime += Simmetrix_layerTimes[i];
    Simmetrix_layerTimes[i] = totalTime;
  }

  if (adapt_params_->isParameter("First Layer Time")) {
    Simmetrix_layerTimes[0] = adapt_params_->get<double>("First Layer Time", 0.0);
  }

  for (i = 0; i < Simmetrix_numLayers; ++i)
    *out << "Simmetrix_layerTimes[" << i << "] = "
      << Simmetrix_layerTimes[i] << '\n';
}
/* BRD */

bool SimLayerAdapt::adaptMesh()
{
  /* dig through all the abstrations to obtain pointers
     to the various structures needed */
  static int callcount = 0;
  Teuchos::RCP<Albany::AbstractDiscretization> disc =
    state_mgr_.getDiscretization();
  Teuchos::RCP<Albany::SimDiscretization> sim_disc =
    Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
  Teuchos::RCP<Albany::APFMeshStruct> apf_ms =
    sim_disc->getAPFMeshStruct();
  apf::Mesh* apf_m = apf_ms->getMesh();
  apf::MeshSIM* apf_msim = dynamic_cast<apf::MeshSIM*>(apf_m);
  pParMesh sim_pm = apf_msim->getMesh();
  /* ensure that users don't expect Simmetrix to transfer IP state */
  bool should_transfer_ip_data = adapt_params_->get<bool>("Transfer IP Data", false);
  /* remove this assert when Simmetrix support IP transfer */
  assert(!should_transfer_ip_data);
  /* compute the size field via SPR error estimation
     on the solution gradient */
  apf::Field* sol_flds[3];
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    sol_flds[i] = apf_m->findField(Albany::APFMeshStruct::solution_name[i]);
  apf::Field* grad_ip_fld = spr::getGradIPField(sol_flds[0], "grad_sol",
      apf_ms->cubatureDegree);
  apf::Field* size_fld = spr::getSPRSizeField(grad_ip_fld, errorBound);
  apf::destroyField(grad_ip_fld);

  pPartitionOpts popts = PM_newPartitionOpts();
  PartitionOpts_setAdaptive(popts, 1);
  PM_partition(sim_pm, popts, sthreadDefault, 0);
  PartitionOpts_delete(popts);

  double sliceThickness;
  GIP_nativeDoubleAttribute(GM_part(Simmetrix_model),"SimLayerThickness",&sliceThickness);

  // Slice thickness
  // Bracket = 0.0003/0.0001 - real part but way too slow
  // sliced_cube.smd = 0.003/0.001 - best model/settings for testing
  // sliced_cube300microns.smd = 0.0003/0.0001 - realistic slices but way too slow
  // Clevis  = 0.03/0.01
  // Use a mesh size for the current layer that is 1/3 the slice thickness
  double layerSize = adapt_params_->get<double>("Layer Mesh Size", sliceThickness / 3.0);

  double max_size = adapt_params_->get<double>("Max Size", 1e10);
  double min_size = adapt_params_->get<double>("Min Size", 1e-2);
  double gradation = adapt_params_->get<double>("Gradation", 0.3);
  assert(min_size <= max_size);

  bool should_debug = adapt_params_->get<bool>("Debug", false);

  /* create the Simmetrix adapter */
  pACase mcase = MS_newMeshCase(Simmetrix_model);
  pModelItem domain = GM_domain(Simmetrix_model);
  MS_setMeshCurv(mcase,domain, ONLY_CURV_TYPE, 0.025);
  MS_setMinCurvSize(mcase,domain, ONLY_CURV_TYPE, 0.0025);
  MS_setMeshSize(mcase,domain, RELATIVE, 1.0, NULL);
  pMSAdapt adapter = MSA_createFromCase(mcase,sim_pm);
  MSA_setSizeGradation(adapter, DO_GRADE, gradation);  // no broomsticks allowed

  /* BRD */
  /* copy the size field from APF to the Simmetrix adapter */
  apf::MeshEntity* v;
  apf::MeshIterator* it = apf_m->begin(0);
  while ((v = apf_m->iterate(it))) {
    double size = apf::getScalar(size_fld, v, 0);
    size = std::min(max_size, size);
    size = std::max(min_size, size);
    MSA_setVertexSize(adapter, (pVertex) v, size);
    apf::setScalar(size_fld, v, 0, size);
  }
  apf_m->end(it);
  /* tell the adapter to transfer the solution and residual fields */
  apf::Field* res_fld = apf_m->findField(Albany::APFMeshStruct::residual_name);
  pField sim_sol_flds[3];
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    sim_sol_flds[i] = apf::getSIMField(sol_flds[i]);
  pField sim_res_fld = apf::getSIMField(res_fld);
  pPList sim_fld_lst = PList_new();
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    PList_append(sim_fld_lst, sim_sol_flds[i]);
  PList_append(sim_fld_lst, sim_res_fld);
  if (apf_ms->useTemperatureHack) {
    /* transfer Temperature_old at the nodes */
    apf::Field* told_fld = apf_m->findField("temp_old");
    pField sim_told_fld = apf::getSIMField(told_fld);
    PList_append(sim_fld_lst, sim_told_fld);
  }
  MSA_setMapFields(adapter, sim_fld_lst);
  /* BRD */
  //PList_delete(sim_fld_lst);
  /* BRD */

  /* BRD */
  GRIter regions = GM_regionIter(Simmetrix_model);
  pGRegion gr1;

  // Constrain the top face & reset sizes
  int layer;
  while (gr1=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
      if (layer==Simmetrix_currentLayer) {
        pPList faceList = GR_faces(gr1);
        void *ent, *iter = 0;
        while(ent = PList_next(faceList,&iter)) {
          pGFace gf = static_cast<pGFace>(ent);
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            GEN_nativeIntAttribute(gf,"SimLayer",&layer);
            if (layer==Simmetrix_currentLayer+1) {
              MSA_setNoModification(adapter,gf);
              for(int np=0;np<PM_numParts(sim_pm);np++) {
                pVertex mv;
                VIter allVerts = M_classifiedVertexIter(PM_mesh(sim_pm,np),gf,1);
                while ( mv = VIter_next(allVerts) ) {
                  MSA_setVertexSize(adapter,mv,layerSize);  // should be same as top layer size in meshModel
                  apf::setScalar(size_fld, reinterpret_cast<apf::MeshEntity*>(mv), 0, layerSize);
                }
                VIter_delete(allVerts);
              }
            }
          }
        }
        PList_delete(faceList);
      }
    }
  }
  GRIter_delete(regions);
  /* BRD */

  if (should_debug) {
    std::stringstream ss;
    ss << "preadapt_" << callcount;
    std::string s = ss.str();
    apf::writeVtkFiles(s.c_str(), apf_m);
  }

  apf::destroyField(size_fld);

  /* run the adapter */
  pProgress progress = Progress_new();
  /* BRD */ 
  MSA_setPrebalance(adapter, 0);
  auto est_nelems = MSA_estimate(adapter);
  std::cout << "MSA estimates " << est_nelems << " elements\n";
  /* BRD */
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
  MS_deleteMeshCase(mcase);

  if (should_debug) {
    std::stringstream ss;
    ss << "postadapt_" << callcount;
    std::string s = ss.str();
    apf::writeVtkFiles(s.c_str(), apf_m);
  }

  /* BRD */
  /*IMPORTANT: next line will not work with current implementation of CTM, because
   CTM does not use param_lib*/
  double currentTime = param_lib_->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  assert(Simmetrix_currentLayer < Simmetrix_numLayers);
  if (currentTime >= Simmetrix_layerTimes[Simmetrix_currentLayer]) {
    char meshFile[80];
    *out << "Adding layer " << Simmetrix_currentLayer+1 << "\n";
    addNextLayer(sim_pm,layerSize,Simmetrix_currentLayer+1,initTempNewLayer,apf_ms->num_time_deriv+1,sim_fld_lst);
    sprintf(meshFile, "layerMesh%d.sms", Simmetrix_currentLayer+1);
    PM_write(sim_pm, meshFile, sthreadDefault, 0);
    Simmetrix_currentLayer++;
  }
  PList_delete(sim_fld_lst);
  /* BRD */

  if (should_debug) {
    std::stringstream ss;
    ss << "postlayer_" << callcount;
    std::string s = ss.str();
    apf::writeVtkFiles(s.c_str(), apf_m);
  }

  /* run APF verification on the resulting mesh */
  apf_m->verify();
  /* update Albany structures to reflect the adapted mesh */
  sim_disc->updateMesh(should_transfer_ip_data);
  /* see the comment in Albany_APFDiscretization.cpp */
  sim_disc->initTemperatureHack();
  ++callcount;
  return true;
}


Teuchos::RCP<const Teuchos::ParameterList> SimLayerAdapt::getValidAdapterParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSimLayerAdaptParams");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<double>("Max Size", 1e10, "Maximum allowed edge length (size field)");
  validPL->set<double>("Min Size", 1e-2, "Minimum allowed edge length (size field)");
  validPL->set<double>("Layer Mesh Size", 1e-2, "Mesh size to use for top layer (default thickness/3)");
  validPL->set<double>("Gradation", 0.3, "Mesh size gradation parameter");
  validPL->set<bool>("Debug", false, "Print debug VTK files");
  validPL->set<bool>("Add Layer", true, "Turn on/off adding layer");
  validPL->set<double>("Uniform Temperature New Layer", 20.0, "Uniform Layer Temperature");
  validPL->set<double>("First Layer Time", 0.0, "Overrides time to place first layer");
  return validPL;
}

}
