#ifndef ALGOTITHMS_HPP
#define ALGOTITHMS_HPP

#include <Eigen/Dense>

namespace Algorithms
{
    namespace HNF
    {
        Eigen::Matrix<unsigned long long, Eigen::Dynamic, Eigen::Dynamic> HNF_full_row_rank(const Eigen::Matrix<unsigned long long, Eigen::Dynamic, Eigen::Dynamic> &B);
        Eigen::MatrixXd HNF(const Eigen::MatrixXd &B);
    }
    namespace CVP
    {
        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
        Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
    }
    Eigen::MatrixXd gram_schmidt(const Eigen::Matrix<unsigned long long, Eigen::Dynamic, Eigen::Dynamic> &matrix, bool delete_zero_rows = true);
}

#endif