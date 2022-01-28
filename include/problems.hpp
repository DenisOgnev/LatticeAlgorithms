#ifndef PROBLEMS_HPP
#define PROBLEMS_HPP

#include <Eigen/Dense>

namespace Solver
{
    Eigen::MatrixXd main_problem(const Eigen::MatrixXd &matrix);
    bool equivalence_problem(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2);
    Eigen::MatrixXd union_problem(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2);
    void containtment_problem(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2);
    void including_problem(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector);
}

#endif