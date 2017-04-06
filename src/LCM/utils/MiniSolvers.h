//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_MiniSolvers_h)
#define LCM_MiniSolvers_h

#include "MiniTensor_Solvers.h"
#include "MiniNonlinearSolver.h"

//
// Define some nonlinear systems (NLS) to test nonlinear solution methods.
//
namespace LCM {

//
//
//
template<typename S, minitensor::Index M = 2>
class Banana : public minitensor::Function_Base<Banana<S, M>, S, M>
{
public:

  Banana()
  {
  }

  static constexpr
  char const * const
  NAME{"Rosenbrock's Banana"};

  using Base = minitensor::Function_Base<Banana<S, M>, S, M>;

  // Default value.
  template<typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const & x)
  {
    return Base::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const & x) const
  {
    minitensor::Index const
    dimension = x.get_dimension();

    assert(dimension == Base::DIMENSION);

    minitensor::Vector<T, N>
    r(dimension);

    r(0) = 2.0 * (x(0) - 1.0) + 400.0 * x(0) * (x(0) * x(0) - x(1));
    r(1) = 200.0 * (x(1) - x(0) * x(0));

    return r;
  }

  // Default AD hessian.
  template<typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const & x)
  {
    return Base::hessian(*this, x);
  }

};

//
//
//
template<typename EvalT, minitensor::Index M = 2>
class Banana_Traits : public
minitensor::Function_Base<Banana_Traits<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

public:

  Banana_Traits(S a = 1.0, S b = 100.0) : a_(a), b_(b)
  {
  }

  static constexpr
  char const * const
  NAME{"Banana_Traits' Function 2D"};

  using Base =
    minitensor::Function_Base<Banana_Traits<EvalT, M>, typename EvalT::ScalarT, M>;

  // Explicit value.
  template<typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const & x)
  {
    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const
    a = peel<EvalT, T, N>()(a_);

    T const
    b = peel<EvalT, T, N>()(b_);

    T const
    c = (a - x(0));

    T const
    d = (x(1) - x(0) * x(0));

    return c * c + b * d * d;
  }

  // Default AD gradient.
  template<typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const & x)
  {
    return Base::gradient(*this, x);
  }

  // Default AD hessian.
  template<typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const & x)
  {
    return Base::hessian(*this, x);
  }

private:
  S
  a_{1.0};

  S
  b_{100.0};
};

} // namespace LCM

#endif // LCM_MiniSolvers_h
