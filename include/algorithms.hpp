#ifndef ALGOTITHMS_HPP
#define ALGOTITHMS_HPP

#include <Eigen/Dense>

namespace Algorithms
{
    namespace HNF
    {
        Eigen::MatrixXd HNF_full_row_rank(const Eigen::MatrixXd &B);
        Eigen::MatrixXd HNF(const Eigen::MatrixXd &B);
    }
    namespace CVP
    {
        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
        Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target);
    }
    Eigen::MatrixXd gram_schmidt(const Eigen::MatrixXd &matrix, bool normalize = false, bool delete_zero_rows = true);
}

#endif