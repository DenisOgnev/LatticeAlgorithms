#ifndef ALGOTITHMS_HPP
#define ALGOTITHMS_HPP

#include <Eigen/Dense>
#include <boost/multiprecision/cpp_int.hpp>
#ifdef GMP
#include <boost/multiprecision/gmp.hpp>
#endif

namespace Algorithms
{
    namespace HNF
    {
        Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF_full_row_rank(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &B);
        Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &B);
        
        #ifdef GMP
        Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> HNF_full_row_rank_GMP(const Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> &B);
        Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> HNF_GMP(const Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> &B);
        #endif
    }
    namespace CVP
    {
        Eigen::VectorXd greedy_recursive(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
        Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
        
        #ifdef PARALLEL
        Eigen::VectorXd branch_and_bound_parallel(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
        #endif
    }
    #ifdef PARALLEL
    Eigen::MatrixXd gram_schmidt_parallel(const Eigen::MatrixXd &matrix, bool delete_zero_rows = true);
    #endif
    Eigen::MatrixXd gram_schmidt_sequential(const Eigen::MatrixXd &matrix, bool delete_zero_rows = true);
}

#endif