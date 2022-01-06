#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <vector>

namespace Utils
{
    Eigen::MatrixXd add_column(Eigen::MatrixXd H, Eigen::ArrayXd b_column);
    Eigen::ArrayXd reduce(Eigen::ArrayXd vector, Eigen::MatrixXd matrix);
    Eigen::MatrixXd generate_random_matrix_with_full_row_rank(const int m, const int n, double lowest, double highest);
    Eigen::MatrixXd generate_random_matrix(const int m, const int n, double lowest, double highest);
    Eigen::MatrixXd get_linearly_independent_columns_by_gram_schmidt(Eigen::MatrixXd matrix);
    std::tuple<Eigen::MatrixXd, std::vector<int>> get_linearly_independent_rows_by_gram_schmidt(Eigen::MatrixXd matrix);
    double det_by_gram_schmidt(Eigen::MatrixXd matrix);
    std::tuple<int, int, int> gcd_extended(int a, int b);
    bool check_linear_independency(Eigen::MatrixXd matrix);
    Eigen::MatrixXd generate_random_matrix_with_linearly_independent_rows(const int m, const int n, double lowest, double highest);
    Eigen::ArrayXd generate_random_array(const int m, double lowest, double highest);
    Eigen::VectorXd projection(Eigen::MatrixXd matrix, Eigen::VectorXd vector);
    double distance_between_two_vectors(Eigen::VectorXd vector1, Eigen::VectorXd vector2);
    Eigen::VectorXd closest_vector(std::vector<Eigen::VectorXd> matrix, Eigen::VectorXd vector);
}

#endif