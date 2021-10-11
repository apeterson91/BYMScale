#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::depends(RcppEigen)]]
//

typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseMatrix<double> SpMat;
Eigen::SimplicialLLT <Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholesky;

// [[Rcpp::export]]
double bym_scale(const SEXP &Q_) {
    
    //Map SparseMatrix
    const SpMat Q(Rcpp::as<MSpMat>(Q_));
    
    MatrixXd L = cholesky.compute(Q).matrixL();
    MatrixXd Sigma;
    Sigma.setZero(Q.rows(),Q.cols());
    Sigma.diagonal() = 1 / pow(L.diagonal().array(), 2);
    int n = Sigma.rows();
    for(int i = (n-2); i >= 0; --i){ 
        for(int j = (n-1); j >= i; --j){
            Sigma(i,j) -= 1.0 / L(i,i) * Sigma.col(j).tail(n-i-1).dot(L.col(i).tail(n-i-1));
            Sigma(j,i) = Sigma(i,j);
        }
    }
    
    MatrixXd A = Eigen::MatrixXd::Constant(1,n,1);
    MatrixXd W = Sigma * A.transpose();
    Sigma = Sigma - W * (A*W).inverse() * W.transpose(); 
    
    
    return(exp(Sigma.diagonal().array().log().mean()));
}