#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

namespace Utils
{
    Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> add_column(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &H, const Eigen::Vector<boost::multiprecision::cpp_int, -1> &b_column);
    Eigen::Vector<boost::multiprecision::cpp_int, -1> reduce(const Eigen::Vector<boost::multiprecision::cpp_int, -1> &vector, const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> generate_random_matrix_with_full_row_rank(const int m, const int n, int lowest, int highest);
    Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> generate_random_matrix(const int m, const int n, int lowest, int highest);
    std::tuple<Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>, Eigen::Matrix<boost::multiprecision::cpp_rational, -1, -1>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    std::tuple<Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<boost::multiprecision::cpp_rational, -1, -1>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    double det_by_gram_schmidt(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    std::tuple<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int, boost::multiprecision::cpp_int> gcd_extended(boost::multiprecision::cpp_int a, boost::multiprecision::cpp_int b);
    std::string matrix_to_string(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    Eigen::MatrixXd generate_random_matrix_with_linearly_independent_rows(const int m, const int n, double lowest, double highest);
    Eigen::ArrayXd generate_random_array(const int m, double lowest, double highest);
    Eigen::VectorXd projection(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector);
    Eigen::VectorXd closest_vector(const std::vector<Eigen::VectorXd> &matrix, const Eigen::VectorXd &vector);
}

#endif