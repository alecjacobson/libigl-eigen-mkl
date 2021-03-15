#include <igl/triangulated_grid.h>
#include <igl/cotmatrix.h>
#include <igl/find.h>
#include <igl/slice.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>

//#define USE_MIN_QUAD_WITH_FIXED
//#define USE_MKL
//#define USE_CHOLMOD
//#define USE_GPL
#ifdef USE_MIN_QUAD_WITH_FIXED
#  include <igl/min_quad_with_fixed.h>
#else
#  include <igl/setdiff.h>
#  include <igl/slice_into.h>
#  ifdef USE_CHOLMOD
#    include <Eigen/CholmodSupport>
#  elif defined(USE_MKL)
#    include <Eigen/PardisoSupport>
#  endif
#endif

int main(int argc, char * argv[])
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::triangulated_grid(64,64,V,F);
  // Build Laplacian for entire mesh
  Eigen::SparseMatrix<double> A;
  {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V,F,L);
    A = -L;
  }
  // Select inner and outer (square) rings
  Eigen::VectorXi b;
  const auto outside = 
        ((V.col(1).array()<0.01) ||
        (V.col(1).array()>0.99) ||
        (V.col(0).array()<0.01) ||
        (V.col(0).array()>0.99)).eval();
  const auto inside = 
       ((V.col(1).array()>0.4) &&
        (V.col(1).array()<0.6) &&
        (V.col(0).array()>0.4) &&
        (V.col(0).array()<0.6)).eval();
  igl::find( (outside || inside).eval() , b);
  // Construct corresponding boundary conditions for those rings
  Eigen::VectorXd bc;
  igl::slice(inside.cast<double>().matrix().eval(),b,bc);
  Eigen::VectorXd Z;
#ifdef USE_MIN_QUAD_WITH_FIXED
  {
    Eigen::VectorXd B = Eigen::VectorXd::Zero(A.rows(),1);
    igl::min_quad_with_fixed(A,B,b,bc,{},Eigen::VectorXd(),true,Z);
  }
#else
  {
    Z.setZero(A.rows());
    // list of indices of unknown vertices
    Eigen::VectorXi u;
    {
      Eigen::VectorXi _;
      igl::setdiff(
        Eigen::VectorXi::LinSpaced(A.rows(),0,A.rows()-1).eval(),b,u,_);
    }
    Eigen::SparseMatrix<double> Auu,Aub;
    igl::slice(A,u,u,Auu);
    igl::slice(A,u,b,Aub);
#ifdef USE_CHOLMOD
    typedef Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> Factor;
#warning "CHOLMOD"
#elif defined(USE_MKL)
    typedef Eigen::PardisoLLT<Eigen::SparseMatrix<double>> Factor;
#warning "MKL"
#elif defined(USE_GPL)
    typedef Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> Factor;
#warning "GPL"
#else
#warning "LU"
    typedef Eigen::SparseLU<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> Factor;
    //typedef Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>,Eigen::IncompleteLUT<double> > Factor;
#endif
    const Factor factor(Auu);
    if(factor.info()!=Eigen::Success) { assert(false && "Solver failed."); }
    const Eigen::VectorXd Zu = factor.solve((-Aub*bc).eval());
    igl::slice_into(Zu,u,1,Z);
    igl::slice_into(bc,b,1,Z);
  }
#endif


  igl::opengl::glfw::Viewer vr;
  vr.data().set_mesh(V,F);
  vr.data().set_data(Z);
  vr.launch();
}
