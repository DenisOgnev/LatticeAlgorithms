#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

namespace Utils
{
    Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> add_column(const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &H, const Eigen::Vector<int64_t, Eigen::Dynamic> &b_column);
    Eigen::Vector<int64_t, Eigen::Dynamic> reduce(const Eigen::Vector<int64_t, Eigen::Dynamic> &vector, const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &matrix);
    Eigen::Matrix<boost::multiprecision::cpp_int, Eigen::Dynamic, Eigen::Dynamic> generate_random_matrix_with_full_row_rank(const int m, const int n, int lowest, int highest);
    Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> generate_random_matrix(const int m, const int n, int lowest, int highest);
    std::tuple<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>, Eigen::MatrixXd, std::vector<int>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &matrix);
    std::tuple<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>, std::vector<int>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &matrix);
    double det_by_gram_schmidt(const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &matrix);
    std::tuple<int64_t, int64_t, int64_t> gcd_extended(int64_t a, int64_t b);
    std::string matrix_to_string(const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &matrix);
    bool check_linear_independency(const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> &matrix);
    Eigen::MatrixXd generate_random_matrix_with_linearly_independent_rows(const int m, const int n, double lowest, double highest);
    Eigen::ArrayXd generate_random_array(const int m, double lowest, double highest);
    Eigen::VectorXd projection(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector);
    double distance_between_two_vectors(const Eigen::VectorXd &vector1, const Eigen::VectorXd &vector2);
    Eigen::VectorXd closest_vector(const std::vector<Eigen::VectorXd> &matrix, const Eigen::VectorXd &vector);
}

#endif