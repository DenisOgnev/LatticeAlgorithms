#include "problems.hpp"
#include "algorithms.hpp"
#include <iostream>

namespace Solver
{
    // Eigen::MatrixXd main_problem(const Eigen::MatrixXd &matrix)
    // {
    //     return Algorithms::HNF::HNF(matrix);
    // }
    // bool equivalence_problem(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2)
    // {
    //     Eigen::MatrixXd H1 = Algorithms::HNF::HNF(matrix1);
    //     Eigen::MatrixXd H2 = Algorithms::HNF::HNF(matrix2);

    //     return H1 == H2;
    // }
    // Eigen::MatrixXd union_problem(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2)
    // {
    //     Eigen::MatrixXd res(matrix1.rows(), matrix1.cols() + matrix2.cols());
    //     res.block(0, 0, matrix1.rows(), matrix1.cols()) = matrix1;
    //     res.block(0, matrix1.cols(), matrix2.rows(), matrix2.cols()) = matrix2;

    //     return Algorithms::HNF::HNF(res);
    // }
    // void containtment_problem(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2)
    // {
    //     Eigen::MatrixXd res(matrix1.rows(), matrix1.cols() + matrix2.cols());
    //     res.block(0, 0, matrix1.rows(), matrix1.cols()) = matrix1;
    //     res.block(0, matrix1.cols(), matrix2.rows(), matrix2.cols()) = matrix2;

    //     std::cout << "HNF(B) = \n" << Algorithms::HNF::HNF(matrix1) << "\n\n";
    //     std::cout << "HNF(B|B') = \n" << Algorithms::HNF::HNF(res) << "\n\n";
    // }
    
    // void including_problem(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector)
    // {
    //     Eigen::MatrixXd res(matrix.rows(), matrix.cols() + vector.cols());
    //     res.block(0, 0, matrix.rows(), matrix.cols()) = matrix;
    //     res.block(0, matrix.cols(), vector.rows(), vector.cols()) = vector;
        
    //     std::cout << "HNF(B) = \n" << Algorithms::HNF::HNF(matrix) << "\n\n";
    //     std::cout << "HNF(B|v) = \n" << Algorithms::HNF::HNF(res) << "\n\n";
    // }
}