//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: no Epetra except SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace LCM {

template <typename EvalT, typename Traits>
KfieldBC_Base<EvalT, Traits>::
KfieldBC_Base(Teuchos::ParameterList& p) :
offset(p.get<int>("Equation Offset")),
  PHAL::DirichletBase<EvalT, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI  =  KIval;
  KII = KIIval;

  KI_name  = p.get< std::string >("Kfield KI Name");
  KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib>>
    ("Parameter Library", Teuchos::null);

  this->registerSacadoParameter(KI_name, paramLib);
  this->registerSacadoParameter(KII_name, paramLib);

  timeValues = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  KIValues = p.get<Teuchos::Array<RealType>>("KI Values").toVector();
  KIIValues = p.get<Teuchos::Array<RealType>>("KII Values").toVector();


  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == KIValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"KI Values\" do not match" );

  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == KIIValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"KII Values\" do not match" );



}

// **********************************************************************
template<typename EvalT, typename Traits>
typename KfieldBC_Base<EvalT, Traits>::ScalarT&
KfieldBC_Base<EvalT, Traits>::
getValue(const std::string &n)
{
  if (n == KI_name)
    return KI;
 // else if (n== timeValues)
//        return timeValues;
  else
        return KII;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void
KfieldBC_Base<EvalT, Traits>::
computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, RealType time)
{

  TEUCHOS_TEST_FOR_EXCEPTION( time > timeValues.back(),
                                      Teuchos::Exceptions::InvalidParameter,
                                      "Time is growing unbounded!" );

  RealType X, Y, R, theta;
  ScalarT coeff_1, coeff_2;
  RealType tau = 6.283185307179586;
  ScalarT KI_X, KI_Y, KII_X, KII_Y;

  X = coord[0];
  Y = coord[1];
  R = std::sqrt(X*X + Y*Y);
  theta = std::atan2(Y,X);

  ScalarT KIFunctionVal, KIIFunctionVal;
  RealType KIslope, KIIslope;
  unsigned int Index(0);

  while( timeValues[Index] < time )
    Index++;

  if (Index == 0) {
    KIFunctionVal  = KIValues[Index];
    KIIFunctionVal = KIIValues[Index];
  }
  else
  {
    KIslope = ( KIValues[Index] - KIValues[Index - 1] ) / ( timeValues[Index] - timeValues[Index - 1] );
    KIFunctionVal = KIValues[Index-1] + KIslope * ( time - timeValues[Index - 1] );

    KIIslope = ( KIIValues[Index] - KIIValues[Index - 1] ) / ( timeValues[Index] - timeValues[Index - 1] );
    KIIFunctionVal = KIIValues[Index-1] + KIIslope * ( time - timeValues[Index - 1] );
  }

  coeff_1 = ( KI*KIFunctionVal / mu ) * std::sqrt( R / tau );
  coeff_2 = ( KII*KIIFunctionVal / mu ) * std::sqrt( R / tau );

  KI_X  = coeff_1 * ( 1.0 - 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );
  KI_Y  = coeff_1 * ( 2.0 - 2.0 * nu - std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );

  KII_X = coeff_2 * ( 2.0 - 2.0 * nu + std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );
  KII_Y = coeff_2 * (-1.0 + 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );

  Xval = KI_X + KII_X;
  Yval = KI_Y + KII_Y;

/*
//  JTO: I am going to leave this here for now...
     std::cout << "================" << std::endl;
     std::cout.precision(15);
     std::cout << "X : " << X << ", Y: " << Y << ", R: " << R << std::endl;
//     std::cout << "Node : " << nsNodes[inode] << std::endl;
     std::cout << "KI : " << KI << ", KII: " << KII << std::endl;
     std::cout << "theta: " << theta << std::endl;
     std::cout << "coeff_1: " << coeff_1 << ", coeff_2: " << coeff_2 << std::endl;
     std::cout << "KI_X: " << KI_X << ", KI_Y: " << KI_Y << std::endl;
     std::cout << "Xval: " << Xval << ", Yval: " << Yval << std::endl;
     std::cout << "nu: " << nu << std::endl;
//     std::cout << "dx: " << (*x)[xlunk] << std::endl;
//     std::cout << "dy: " << (*x)[ylunk] << std::endl;
//     std::cout << "fx: " << ((*x)[xlunk] - Xval) << std::endl;
//     std::cout << "fy: " << ((*x)[ylunk] - Yval) << std::endl;
     std::cout << "sin(theta/2): " << std::sin( theta / 2.0 ) << std::endl;
     std::cout << "cos(theta/2): " << std::cos( theta / 2.0 ) << std::endl;

     */
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();


  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;


  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    fT_nonconstView[xlunk] = xT_constView[xlunk] - Xval;
    fT_nonconstView[ylunk] = xT_constView[ylunk] - Yval;

    // Record DOFs to avoid setting Schwarz BCs on them.
    dirichletWorkset.fixed_dofs_.insert(xlunk);
    dirichletWorkset.fixed_dofs_.insert(ylunk);
  }

}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;


  RealType time = dirichletWorkset.current_time;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;

  ScalarT Xval, Yval; 
  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;


  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) 
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);
    
    // replace jac values for the X dof 
    numEntriesT = jacT->getNumEntriesInLocalRow(xlunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(xlunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
    jacT->replaceLocalValues(xlunk, matrixIndicesT(), matrixEntriesT());

    index[0] = xlunk;
    jacT->replaceLocalValues(xlunk, index(), value());


    // replace jac values for the y dof
    numEntriesT = jacT->getNumEntriesInLocalRow(ylunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(ylunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
    jacT->replaceLocalValues(ylunk, matrixIndicesT(), matrixEntriesT());

    index[0] = ylunk;
    jacT->replaceLocalValues(ylunk, index(), value());    


    if (fillResid)
    {
      fT_nonconstView[xlunk] = xT_constView[xlunk] - Xval.val();
      fT_nonconstView[ylunk] = xT_constView[ylunk] - Yval.val();
    } 
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = dirichletWorkset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = dirichletWorkset.VxT;

  Teuchos::ArrayRCP<const ST> VxT_constView; 
  Teuchos::ArrayRCP<ST> fT_nonconstView;                                         
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();                                       
  
RealType time = dirichletWorkset.current_time;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (fT != Teuchos::null)
    {
      fT_nonconstView[xlunk] = xT_constView[xlunk] - Xval.val();
      fT_nonconstView[ylunk] = xT_constView[ylunk] - Yval.val();
    } 

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        JVT_nonconstView = JVT->getDataNonConst(i); 
        VxT_constView = VxT->getData(i); 
	JVT_nonconstView[xlunk] = j_coeff*VxT_constView[xlunk];
	JVT_nonconstView[ylunk] = j_coeff*VxT_constView[ylunk];
      }
    }
    
    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        fpT_nonconstView = fpT->getDataNonConst(i); 
	fpT_nonconstView[xlunk] = -Xval.dx(dirichletWorkset.param_offset+i);
	fpT_nonconstView[ylunk] = -Yval.dx(dirichletWorkset.param_offset+i);
      }
    }

  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  Teuchos::ArrayRCP<ST> fpVT_nonconstView; 
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  // RealType time = dirichletWorkset.current_time;

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int xlunk, ylunk; // global and local indicies into unknown vector
  // double* coord;
  // ScalarT Xval, Yval;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
      Teuchos::ArrayRCP<ST> VpT_nonconstView; 
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][xlunk] = 0.0;
        //(*Vp)[col][ylunk] = 0.0;
        VpT_nonconstView = VpT->getDataNonConst(col); 
        VpT_nonconstView[xlunk] = 0.0; 
        VpT_nonconstView[ylunk] = 0.0; 
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][xlunk] = 0.0;
        //(*fpV)[col][ylunk] = 0.0;
        fpVT_nonconstView = fpVT->getDataNonConst(col);
        fpVT_nonconstView[xlunk] = 0.0; 
        fpVT_nonconstView[ylunk] = 0.0; 
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
#ifdef ALBANY_SG
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::SGResidual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
 const std::vector<double*>& nsNodeCoords =
   dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    for (int block=0; block<nblock; block++) {
      (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.coeff(block));
      (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.coeff(block));
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>> jac =
    dirichletWorkset.sg_Jac;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = jac->size();

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; block++) {
      (*jac)[block].ExtractMyRowView(xlunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(ylunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    }
    (*jac)[0].ReplaceMyValues(xlunk, 1, &diag, &xlunk);
    (*jac)[0].ReplaceMyValues(ylunk, 1, &diag, &ylunk);

    if (fillResid)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.val().coeff(block));
        (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.val().coeff(block));
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::SGTangent, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> fp =
    dirichletWorkset.sg_fp;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> JV =
    dirichletWorkset.sg_JV;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  Teuchos::RCP<const Epetra_MultiVector> Vx = dirichletWorkset.Vx;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int nblock = x->size();

  RealType time = dirichletWorkset.current_time;
  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (f != Teuchos::null)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = (*x)[block][xlunk] - Xval.val().coeff(block);
        (*f)[block][ylunk] = (*x)[block][ylunk] - Yval.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        (*JV)[0][i][xlunk] = j_coeff*(*Vx)[i][xlunk];
        (*JV)[0][i][ylunk] = j_coeff*(*Vx)[i][ylunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        for (int block=0; block<nblock; block++) {
          (*fp)[block][i][xlunk] =
            -Xval.dx(dirichletWorkset.param_offset+i).coeff(block);
          (*fp)[block][i][ylunk] =
            -Yval.dx(dirichletWorkset.param_offset+i).coeff(block);
        }
      }
    }

  }
}

#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::MPResidual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;


  RealType time = dirichletWorkset.current_time;
  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    for (int block=0; block<nblock; block++) {
      (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.coeff(block));
      (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.coeff(block));
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix>> jac =
    dirichletWorkset.mp_Jac;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = jac->size();

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; block++) {
      (*jac)[block].ExtractMyRowView(xlunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
      (*jac)[block].ReplaceMyValues(xlunk, 1, &diag, &xlunk);

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(ylunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
      (*jac)[block].ReplaceMyValues(ylunk, 1, &diag, &ylunk);
    }

    if (fillResid)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.val().coeff(block));
        (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.val().coeff(block));
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::MPTangent, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> fp =
    dirichletWorkset.mp_fp;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> JV =
    dirichletWorkset.mp_JV;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  Teuchos::RCP<const Epetra_MultiVector> Vx = dirichletWorkset.Vx;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int nblock = x->size();
  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (f != Teuchos::null)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = (*x)[block][xlunk] - Xval.val().coeff(block);
        (*f)[block][ylunk] = (*x)[block][ylunk] - Yval.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        for (int block=0; block<nblock; block++) {
          (*JV)[block][i][xlunk] = j_coeff*(*Vx)[i][xlunk];
          (*JV)[block][i][ylunk] = j_coeff*(*Vx)[i][ylunk];
        }
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        for (int block=0; block<nblock; block++) {
          (*fp)[block][i][xlunk] =
            -Xval.dx(dirichletWorkset.param_offset+i).coeff(block);
          (*fp)[block][i][ylunk] =
            -Yval.dx(dirichletWorkset.param_offset+i).coeff(block);
        }
      }
    }

  }
}
#endif

} // namespace LCM

