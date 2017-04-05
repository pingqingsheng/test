#include <iostream>
#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;
// 
// // Matrix  Multiplication
// // [[Rcpp::export]]
// NumericMatrix MatProd( NumericMatrix X, NumericMatrix Y ){
// 
//   int nRow = X.nrow() ;
//   int nCol = Y.ncol() ;
// 
//   NumericMatrix Cmat(nRow,nCol) ;
// 
//   for (int i=0; i<nRow; i++) {
// 
//     for (int j=0; j<nCol; j++){
// 
//       Cmat(i,j) = sum(X(i,_) * Y(_,j)) ;
// 
//     }
// 
//   }
// 
//   return(Cmat);
// }
// 
// // Matrix  Addition
// // [[Rcpp::export]]
// NumericMatrix MatAdd( NumericMatrix X, NumericMatrix Y ){
// 
//   int nRow = X.nrow();
//   int nCol = Y.ncol();
// 
//   NumericMatrix Cmat(nRow, nCol) ;
// 
//   for (int i=0; i<nRow; i++){
// 
//     for(int j=0; j<nCol; j++){
// 
//       Cmat(i,j) = X(i,j) + Y(i,j);
// 
//     }
// 
//   }
// 
//   return(Cmat);
// 
// }
// 
// // Matrix Elementwise Multiplication
// // [[Rcpp::export]]
// NumericMatrix MatEleProd( NumericMatrix X, NumericMatrix Y ){
// 
//   int nRow = X.nrow() ;
//   int nCol = Y.ncol() ;
// 
//   NumericMatrix Cmat(nRow,nCol) ;
// 
//   for (int i=0; i<nRow; i++) {
// 
//     for (int j=0; j<nCol; j++){
// 
//       Cmat(i,j) = X(i,j) * Y(i,j) ;
// 
//     }
// 
//   }
// 
//   return(Cmat);
// }
// 
// // Matrix Elementwise Division
// // [[Rcpp::export]]
// NumericMatrix MatEleDivd( NumericMatrix X, NumericMatrix Y ){
// 
//   int nRow = X.nrow() ;
//   int nCol = Y.ncol() ;
// 
//   NumericMatrix Cmat(nRow,nCol) ;
// 
//   for (int i=0; i<nRow; i++) {
// 
//     for (int j=0; j<nCol; j++){
// 
//       Cmat(i,j) = X(i,j) / Y(i,j) ;
// 
//     }
// 
//   }
// 
//   return(Cmat);
// }
// 
// // Column Raw Summation
// // [[Rcpp::export]]
// NumericVector apply_sum( NumericMatrix X, int direction){
// 
//   if((direction == 2)){
// 
//     NumericVector X_bar(X.ncol());
// 
//     for(int i=0; i<X.ncol(); i++){
// 
//       X_bar(i) = sum(X(_,i));
// 
//     }
// 
//     return(X_bar);
// 
//   }else if((direction == 1)){
// 
//     NumericVector X_bar(X.nrow());
// 
//     for(int i=0; i<X.nrow(); i++){
// 
//       X_bar(i) = sum(X(i,_));
// 
//     }
// 
//     return(X_bar);
// 
//   }
// 
//   return(0);
// 
// }
// 
// // Matrix Element Wise Square Root
// // [[Rcpp::export]]
// NumericMatrix MatEleSqrt( NumericMatrix X ){
// 
//   NumericMatrix cMat(X.nrow(), X.ncol());
// 
//   for(int i=0; i<X.nrow(); i++){
// 
//     for(int j=0; j<X.ncol(); j++){
// 
//       cMat(i,j) = sqrt(X(i,j));
// 
//     }
// 
//   }
// 
//   return(cMat);
// 
// }

// Main Function
//[[Rcpp::export]]
arma::vec ADMM_Lasso( arma::mat X, arma::colvec Y, double lambda, double eta){

  int nVar  = X.n_cols;

  int marker = 1;
  double tol   = 0.01;
  double tol_1 = 0.1;
  double tol_2 = 0.1;

  int terminate = 0;
  int converge  = 0;

  // Initialization
  arma::vec b_gd    = rnorm(X.n_cols, 0, 1) ;
  arma::vec z_admm  = rnorm(X.n_cols, 0, 1) ;
  arma::vec z_old = z_admm;
  arma::vec mu_admm = rnorm(X.n_cols, 0, 1) ;
  
  // Using ADAM GD method
  NumericVector m_temp;
  NumericVector v_temp;
  
  arma::vec m;
  arma::vec v;
  
  arma::vec steps;
  
  while(!terminate){
    
    m_temp = rep(0,X.n_cols);
    m = as<arma::vec>(m_temp); 
    v_temp = rep(0,X.n_cols);
    v = as<arma::vec>(v_temp);
    
    converge  = 0;
    int marker_inner = 0;

    // Optimize Beta
    while(!converge){
      
      arma::vec G_1(X.n_cols);
      arma::vec G_2(X.n_cols);
      arma::vec G_3(X.n_cols);
      arma::vec G(X.n_cols);

      for(int i=0; i<nVar; i++){

        G_1(i) = sum(X.col(i) % Y);
        G_2(i) = sum(exp(X * b_gd) % X.col(i));

      }

      G_3 = (b_gd - z_admm) + mu_admm;

      G = G_1 - G_2 + G_3;

      m = eta * 0.1*m + eta * 0.9*G;
      v = 0.1*v + 0.9*G % G;
      
      steps = m % pow(v, -0.5);
      b_gd  = b_gd  +  steps;
      
      Rcout << "b_gd : " << trans(b_gd) << std::endl;
      
      marker_inner = marker_inner + 1;

      if( max(steps) < tol || marker_inner > 20 ){

        converge = 1;
        break;

      }

    }

    // Soft Threshhold Z
    for(int i=0; i<nVar; i++){

      if(b_gd(i)+mu_admm(i) > lambda){

        z_admm(i) = b_gd(i) + mu_admm(i) - lambda;

      }
      else if(b_gd(i)+mu_admm(i) < -lambda){

        z_admm(i) = b_gd(i) + mu_admm(i) + lambda;

      }
      else{

        z_admm(i) = 0;

      }

    }
    
    // For monitor reason
    Rcout << "z_admm : "  << trans(z_admm) << endl;
   
    // Updating augmented mu
    mu_admm = b_gd - z_admm + mu_admm;
    
    // For monitor reason
    Rcout << "mu_admm : " << trans(mu_admm) << endl;

    // Check Convergence
    int cond_1 = (max(abs(b_gd - z_admm))  < tol_1);
    int cond_2 = (max(abs(z_admm - z_old)) < tol_2);

    if((cond_1==1) && (cond_2==1)){

      terminate = 1;
      break;

    }

    z_old = z_admm;

    marker = marker + 1;

    Rcout << "Iteration : " << marker << std::endl;

  }

  Rcpp::Rcout << trans(b_gd)    << std::endl ;
  Rcpp::Rcout << trans(z_admm)  << std::endl ;
  Rcpp::Rcout << trans(mu_admm) << std::endl ;

  return(b_gd);

}







