#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <vector>

namespace Utils
{
    Eigen::MatrixXi add_column(const Eigen::MatrixXi &H, const Eigen::VectorXi &b_column);
    Eigen::VectorXi reduce(const Eigen::VectorXi &vector, const Eigen::MatrixXi &matrix);
    Eigen::MatrixXi generate_random_matrix_with_full_row_rank(const int m, const int n, int lowest, int highest);
    Eigen::MatrixXi generate_random_matrix(const int m, const int n, int lowest, int highest);
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXd, std::vector<int>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::MatrixXi &matrix);
    std::tuple<Eigen::MatrixXi, std::vector<int>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::MatrixXi &matrix);
    double det_by_gram_schmidt(const Eigen::MatrixXi &matrix);
    std::tuple<int, int, int> gcd_extended(int a, int b);
    std::string matrix_to_string(const Eigen::MatrixXi &matrix);
    bool check_linear_independency(const Eigen::MatrixXi &matrix);
    Eigen::MatrixXd generate_random_matrix_with_linearly_independent_rows(const int m, const int n, double lowest, double highest);
    Eigen::ArrayXd generate_random_array(const int m, double lowest, double highest);
    Eigen::VectorXd projection(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector);
    double distance_between_two_vectors(const Eigen::VectorXd &vector1, const Eigen::VectorXd &vector2);
    Eigen::VectorXd closest_vector(const std::vector<Eigen::VectorXd> &matrix, const Eigen::VectorXd &vector);
}

#endif