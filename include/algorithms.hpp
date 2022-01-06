#ifndef ALGOTITHMS_HPP
#define ALGOTITHMS_HPP

#include <Eigen/Dense>

namespace Algorithms
{
    namespace HNF
    {
        Eigen::MatrixXd HNF_full_row_rank(Eigen::MatrixXd B);
        Eigen::MatrixXd HNF(Eigen::MatrixXd B);
    }
    namespace CVP
    {
        Eigen::VectorXd greedy(Eigen::MatrixXd matrix, Eigen::VectorXd target);
        Eigen::VectorXd branch_and_bound(Eigen::MatrixXd matrix, Eigen::VectorXd target);
    }
    Eigen::MatrixXd gram_schmidt(Eigen::MatrixXd matrix, bool normalize = false, bool delete_zero_rows = true);
}

#endif