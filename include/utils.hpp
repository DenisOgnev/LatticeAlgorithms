#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef GMP
#include <boost/multiprecision/gmp.hpp>
#endif

namespace Utils
{
    Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> add_column(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &H, const Eigen::Vector<boost::multiprecision::cpp_int, -1> &b_column);
    Eigen::Vector<boost::multiprecision::cpp_int, -1> reduce(const Eigen::Vector<boost::multiprecision::cpp_int, -1> &vector, const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> generate_random_matrix_with_full_row_rank(const int m, const int n, int lowest, int highest);
    Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> generate_random_matrix(const int m, const int n, int lowest, int highest);
    std::tuple<Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>, Eigen::Matrix<boost::multiprecision::cpp_rational, -1, -1>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    std::tuple<Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<boost::multiprecision::cpp_rational, -1, -1>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix);
    std::tuple<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int, boost::multiprecision::cpp_int> gcd_extended(boost::multiprecision::cpp_int a, boost::multiprecision::cpp_int b);

    #ifdef GMP
    Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> add_column_GMP(const Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> &H, const Eigen::Vector<boost::multiprecision::mpz_int, -1> &b_column);
    Eigen::Vector<boost::multiprecision::mpz_int, -1> reduce_GMP(const Eigen::Vector<boost::multiprecision::mpz_int, -1> &vector, const Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> &matrix);
    Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> generate_random_matrix_with_full_row_rank_GMP(const int m, const int n, int lowest, int highest);
    Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> generate_random_matrix_GMP(const int m, const int n, int lowest, int highest);
    std::tuple<Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>, Eigen::Matrix<boost::multiprecision::mpq_rational, -1, -1>> get_linearly_independent_columns_by_gram_schmidt_GMP(const Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> &matrix);
    std::tuple<Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<boost::multiprecision::mpq_rational, -1, -1>> get_linearly_independent_rows_by_gram_schmidt_GMP(const Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1> &matrix);
    std::tuple<boost::multiprecision::mpz_int, boost::multiprecision::mpz_int, boost::multiprecision::mpz_int> gcd_extended_GMP(boost::multiprecision::mpz_int a, boost::multiprecision::mpz_int b);
    #endif

    Eigen::MatrixXd generate_random_matrix_with_full_column_rank(const int m, const int n, int lowest, int highest);
    Eigen::VectorXd generate_random_vector(const int m, double lowest, double highest);
    Eigen::VectorXd projection(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector);
    Eigen::VectorXd closest_vector(const std::vector<Eigen::VectorXd> &matrix, const Eigen::VectorXd &vector);
}

#endif