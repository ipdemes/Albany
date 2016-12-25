//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_KineticEnergy<EvalT, Traits>::
XZHydrostatic_KineticEnergy(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  u  (p.get<std::string> ("Velx"),           dl->node_vector_level),
  ke (p.get<std::string> ("Kinetic Energy"), dl->node_scalar_level),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{

  this->addDependentField(u);
  this->addEvaluatedField(ke);

  this->setName("Aeras::XZHydrostatic_KineticEnergy" + PHX::typeAsString<EvalT>());

  ke0 = 0.0;

}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_KineticEnergy<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u,fm);
  this->utils.setFieldData(ke,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
//original version
#if 0
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_KineticEnergy<EvalT, Traits>::
operator() (const XZHydrostatic_KineticEnergy_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) {
    for (int level=0; level < numLevels; ++level) {
      ke(cell,node,level) = 0;
      for (int dim=0; dim < numDims; ++dim) {
        ke(cell,node,level) += 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
      }
    }
  }
}
#endif
#if 0
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_KineticEnergy<EvalT, Traits>::
operator() ( const team_member & thread) const{
  int cell = thread.league_rank() * thread.team_size() + thread.team_rank();
#if 0
  for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT temp=0;
        //ke(cell,node,level) = 0;
        for (int dim=0; dim < numDims; ++dim) {
          temp += 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
        }
        ke(cell,node,level)=temp;
      }
    }
  
#endif

#if 0

  for (int node=0; node < numNodes; ++node) {
     for (int level=0; level < numLevels; ++level) {
        ScalarT tsum=0;
   
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread,numDims), [=] (const int& dim, ScalarT & vsum) {
          vsum+= 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
        },tsum);//parallel reduce

        ke(cell,node,level)=tsum;
       }
    }

#endif

#if 0

  cell = thread.league_rank();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(thread , numNodes*numLevels),
    KOKKOS_LAMBDA (const int& j)  {
        int node = j/numLevels;
        int level = j-j/numLevels;
        ScalarT tsum=0;

        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread,numDims), [=] (const int& dim, ScalarT & vsum) {
          int node = j/numLevels;
        int level = j-j/numLevels;
          vsum+= 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
        },tsum);//parallel reduce
        Kokkos::single(Kokkos::PerThread(thread),[=] () {
           ke(cell,node,level)=tsum;
        });
  });

#endif

#if 0

//  shared_1d_int count(thread.team_shmem(),numNodes*numLevels);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(thread,numNodes*numLevels), [=] (const int& j) {
    int node = j/numLevels;
    int level = j-j/numLevels;
    ScalarT tsum=0;
// Kokkos::single(Kokkos::PerThread(thread),[=] () {
//    ke(cell,node,level) = 0;
//  });
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread,numDims), [=] (const int& dim, ScalarT & vsum) {
        vsum+= 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
      },tsum);//parallel reduce

     Kokkos::single(Kokkos::PerThread(thread),[=] () {
        ke(cell,node,level)=tsum;
     });//single

  });//parallel_for
#endif
}
#endif
//option 3
#if 1
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_KineticEnergy<EvalT, Traits>::
operator() ( const team_member & thread) const{
  int cell = thread.league_rank();
  int node = thread.team_rank();
  for (int level=0; level < numLevels; ++level) {
        ScalarT tsum=0;
   
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread,numDims), 
		[=] (const int& dim, ScalarT & vsum) 
	{
          vsum+= 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
        },tsum);//parallel reduce

        ke(cell,node,level)=tsum;
       }
}
#endif
#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_KineticEnergy<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        ke(cell,node,level) = 0;
        for (int dim=0; dim < numDims; ++dim) {
          ke(cell,node,level) += 0.5*u(cell,node,level,dim)*u(cell,node,level,dim);
        }
      }
    }
  }

#else
//original version
#if 0
  Kokkos::parallel_for(XZHydrostatic_KineticEnergy_Policy(0,workset.numCells),*this);
//version2
#endif
#if 0
  int team_size=32;
  int num_teams = workset.numCells/team_size;
  if ((workset.numCells-num_teams*team_size)>0) num_teams++;
  int vector_length = 8;
  //const Kokkos::TeamPolicy<> policy( num_teams , team_size , vector_length);
  const Kokkos::TeamPolicy<> policy( workset.numCells , 1 , vector_length);
//  Kokkos::TeamPolicy<> policy( 1 , workset.numCells);
  Kokkos::parallel_for( policy , *this );
#endif

//version 3
#if 1
  int team_size=numNodes;
  int num_teams = workset.numCells;
  int vector_length = 2;
  const Kokkos::TeamPolicy<> policy( num_teams, team_size , vector_length);
  Kokkos::parallel_for( policy , *this );
#endif
#endif

//std::cout<< "num_nodes="<< numNodes<<" , nuCells = "<< workset.numCells<< ", numLevels = " <<numLevels<< ", numDims = "<<numDims<<std::endl;
#if 0
std::cout<<"print ke"<<std::endl;
for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int level=0; level < numLevels; ++level) 
	std::cout<<ke(cell,node,level)<<"  , " <<u(cell,node,level,0)<<std::endl;
#endif
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename XZHydrostatic_KineticEnergy<EvalT,Traits>::ScalarT& 
XZHydrostatic_KineticEnergy<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="KineticEnergy") return ke0;
  return ke0;
}

}
