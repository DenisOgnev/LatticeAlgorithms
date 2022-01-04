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
    Eigen::MatrixXd gram_schmidt(Eigen::MatrixXd matrix, bool normalize = false, bool delete_zero_rows = true);
}

#endif