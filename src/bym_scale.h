#ifndef _BYMSCALE_
#define _BYMSCALE_

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
Eigen::SimplicialLLT <Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholesky;

#endif