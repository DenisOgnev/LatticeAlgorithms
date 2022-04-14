#include "utils.hpp"
#include <iostream>
#include <random>
#include <functional>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <string> 
#include <chrono>
#include <thread>
#include "algorithms.hpp"

namespace mp = boost::multiprecision;
typedef mp::number<mp::cpp_bin_float_100::backend_type, mp::et_off> cpp_bin_float_100_et_off;

namespace Utils
{
    // Function for computing HNF of full row rank matrix
    // @return Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic>
    // @param H HNF
    // @param b column to be added
    Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> add_column(const Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> &H, const Eigen::Vector<mp::cpp_int, Eigen::Dynamic> &b_column)
    {
        if (H.rows() == 0)
        {
            return H;
        }

        Eigen::Vector<mp::cpp_int, Eigen::Dynamic> H_first_col = H.col(0);

        mp::cpp_int a = H_first_col(0);
        Eigen::Vector<mp::cpp_int, Eigen::Dynamic> h = H_first_col.tail(H_first_col.rows() - 1);
        Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> H_stroke = H.block(1, 1, H.rows() - 1, H.cols() - 1);
        mp::cpp_int b = b_column(0);
        Eigen::Vector<mp::cpp_int, Eigen::Dynamic> b_stroke = b_column.tail(b_column.rows() - 1);

        std::tuple<mp::cpp_int, mp::cpp_int, mp::cpp_int> gcd_result = gcd_extended(a, b);
        mp::cpp_int g, x, y;
        std::tie(g, x, y) = gcd_result;

        Eigen::Matrix<mp::cpp_int, 2, 2> U;
        //U << static_cast<mp::cpp_bin_float_100>(x), static_cast<mp::cpp_bin_float_100>(-b) / static_cast<mp::cpp_bin_float_100>(g), static_cast<mp::cpp_bin_float_100>(y), static_cast<mp::cpp_bin_float_100>(a) / static_cast<mp::cpp_bin_float_100>(g);
        U << x, -b / g, y, a / g;
        

        Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> temp_matrix(H.rows(), 2);
        temp_matrix.col(0) = H_first_col;
        temp_matrix.col(1) = b_column;
        Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> temp_result = temp_matrix * U;

        Eigen::Vector<mp::cpp_int, Eigen::Dynamic> h_stroke = temp_result.col(0).tail(temp_result.rows() - 1);
        Eigen::Vector<mp::cpp_int, Eigen::Dynamic> b_double_stroke = temp_result.col(1).tail(temp_result.rows() - 1);

        b_double_stroke = reduce(b_double_stroke, H_stroke);

        Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> H_double_stroke = add_column(H_stroke, b_double_stroke);

        h_stroke = reduce(h_stroke, H_double_stroke);

        Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> result(H.rows(), H.cols());

        result(0, 0) = g;
        result.col(0).tail(result.cols() - 1) = h_stroke;
        result.row(0).tail(result.rows() - 1).setZero();
        result.block(1, 1, H_double_stroke.rows(), H_double_stroke.cols()) = H_double_stroke;

        return result;
    }

    // Function for computing HNF, reduces elements of vector modulo diagonal elements of matrix
    // @return Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic>
    // @param vector vector to be reduced
    // @param matrix input matrix
    Eigen::Vector<mp::cpp_int, Eigen::Dynamic> reduce(const Eigen::Vector<mp::cpp_int, Eigen::Dynamic> &vector, const Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
    {
        Eigen::Vector<mp::cpp_int, Eigen::Dynamic> result = vector;
        for (int i = 0; i < result.rows(); i++)
        {
            // Eigen::Vector<cpp_int, Eigen::Dynamic> matrix_column = matrix.col(i);
            // double vec_elem = static_cast<double>(result(i));
            // double matrix_elem = static_cast<double>(matrix(i, i));
            // cpp_int x;
            // if (vec_elem < 0)
            // {
            //     x = static_cast<cpp_int>(std::ceil(std::abs(vec_elem) / matrix_elem));
            //     result += matrix_column * x;
            //     vec_elem = static_cast<double>(result(i));
            // }
            // if (vec_elem >= matrix_elem)
            // {
            //     x = static_cast<cpp_int>(std::floor(vec_elem / matrix_elem));
            //     result -= matrix_column * x;
            // }
            Eigen::Vector<mp::cpp_int, Eigen::Dynamic> matrix_column = matrix.col(i);
            mp::cpp_int t_vec_elem = result(i);
            mp::cpp_int t_matrix_elem = matrix(i, i);

            boost::multiprecision::cpp_int x;
            if (t_vec_elem >= 0)
            {
                x = (t_vec_elem / t_matrix_elem);
            }
            else
            {
                x = (t_vec_elem - (t_matrix_elem - 1)) / t_matrix_elem;
            }

            result.noalias() -= matrix_column * x;
        }
        return result;
    }
    
    // Generates random matrix with full row rank (or with all rows linearly independent)
    // @return Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic>
    // @param m number of rows, must be greater than one and less than or equal to the parameter n
    // @param n number of columns, must be greater than one and greater than or equal to the parameter m 
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> generate_random_matrix_with_full_row_rank(const int m, const int n, int lowest, int highest)
    {
        if (m > n)
        {
            throw std::invalid_argument("m must be less than or equal n");
        }
        if (m < 1 || n < 1)
        {
            throw std::invalid_argument("Number of rows or columns should be greater than one");
        }
        if (highest - lowest < 1)
        {
            throw std::invalid_argument("highest parameter must be greater than lowest parameter by at least one");
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis (lowest, highest);

        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(matrix.cast<double>());
        auto rank = lu_decomp.rank();

        while (rank != m)
        {
            matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(m, n, [&]()
                                                  { return dis(gen); });

            lu_decomp.compute(matrix.cast<double>());
            rank = lu_decomp.rank();
        }

        return matrix.cast<mp::cpp_int>();
    }

    // Generates random matrix
    // @return Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic>
    // @param m number of rows, must be greater than one
    // @param n number of columns, must be greater than one
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> generate_random_matrix(const int m, const int n, int lowest, int highest)
    {
        if (m < 1 || n < 1)
        {
            throw std::invalid_argument("Number of rows or columns should be greater than one");
        }
        if (highest - lowest < 1)
        {
            throw std::invalid_argument("highest parameter must be greater than lowest parameter by at least one");
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis (lowest, highest);

        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        return matrix.cast<mp::cpp_int>();
    }

    // Returns matrix that consist of linearly independent columns of input matrix, othogonalized matrix and indexes of that columns in input matrix
    // @param matrix input matrix
    // @return std::tuple<Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<cpp_bin_float_100, Eigen::Dynamic, Eigen::Dynamic>, std::vector<int>>
    std::tuple<Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<mp::cpp_rational, Eigen::Dynamic, Eigen::Dynamic>, std::vector<int>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
    {
        std::vector<Eigen::Vector<mp::cpp_rational, Eigen::Dynamic>> basis;
        std::vector<int> indexes;

        int counter = 0;
        for (const Eigen::Vector<mp::cpp_rational, Eigen::Dynamic> &vec : matrix.cast<mp::cpp_rational>().colwise())
        {
            Eigen::Vector<mp::cpp_rational, Eigen::Dynamic> projections = Eigen::Vector<mp::cpp_rational, Eigen::Dynamic>::Zero(vec.size());

            //#pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                mp::cpp_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis[i].data(), mp::cpp_rational(0.0));
                mp::cpp_rational inner2 = std::inner_product(basis[i].data(), basis[i].data() + basis[i].size(), basis[i].data(), mp::cpp_rational(0.0));
                //#pragma omp critical
                projections.noalias() += (inner1 / inner2) * basis[i];
            }

            Eigen::Vector<mp::cpp_rational, Eigen::Dynamic> result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                basis.push_back(result);
                indexes.push_back(counter);
            }
            counter++;
        }

        Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> result(matrix.rows(), indexes.size());
        Eigen::Matrix<mp::cpp_rational, Eigen::Dynamic, Eigen::Dynamic> gram_schmidt(matrix.rows(), basis.size());
        //#pragma omp parallel for
        for (int i = 0; i < indexes.size(); i++)
        {
            result.col(i) = matrix.col(indexes[i]);
            gram_schmidt.col(i) = basis[i];
        }
        return std::make_tuple(result, gram_schmidt, indexes);
    }

    // Returns matrix that consist of linearly independent rows of input matrix and indexes of that rows in input matrix
    // @param matrix input matrix
    // @return std::tuple<Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic>, std::vector<int>>
    // std::tuple<Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic>, std::vector<int>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
    // {
    //     std::vector<Eigen::Vector<mp::cpp_bin_float_100, Eigen::Dynamic>> basis;
    //     std::vector<int> indexes;
    //     std::vector<mp::cpp_bin_float_100> uji;

    //     int counter = 0;
    //     for (const Eigen::Vector<mp::cpp_int, Eigen::Dynamic> &vec : matrix.rowwise())
    //     {
    //         Eigen::Vector<mp::cpp_bin_float_100, Eigen::Dynamic> projections = Eigen::Vector<mp::cpp_bin_float_100, Eigen::Dynamic>::Zero(vec.size());

    //         //#pragma omp parallel for
    //         for (int i = 0; i < basis.size(); i++)
    //         {
    //             mp::cpp_bin_float_100 inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis[i].data(), 0.0);
    //             mp::cpp_bin_float_100 inner2 = std::inner_product(basis[i].data(), basis[i].data() + basis[i].size(), basis[i].data(), 0.0);
    //             uji.push_back(inner1 / inner2);
    //             projections.noalias() += (inner1 / inner2) * basis[i];
    //         }
    //         // NO PARALLEL
    //         // for (const auto &b : basis)
    //         // {
    //         //     double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), b.data(), 0.0);
    //         //     double inner2 = std::inner_product(b.data(), b.data() + b.size(), b.data(), 0.0);
    //         //     projections.noalias() += (inner1 / inner2) * b;
    //         // }

    //         Eigen::Vector<mp::cpp_bin_float_100, Eigen::Dynamic> result = vec.cast<mp::cpp_bin_float_100>() - projections;

    //         bool is_all_zero = result.isZero(1e-3);
    //         if (!is_all_zero)
    //         {
    //             basis.push_back(result);
    //             indexes.push_back(counter);
    //         }
    //         counter++;
    //     }

    //     Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> result(indexes.size(), matrix.cols());
    //     //#pragma omp parallel for
    //     for (int i = 0; i < indexes.size(); i++)
    //     {
    //         result.row(i) = matrix.row(indexes[i]);
    //     }
    //     return std::make_tuple(result, indexes);
    // }

    // Computes determinant by using Gram Schmidt orthogonalization
    // @return double
    // @param matrix input matrix
    // double det_by_gram_schmidt(const Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
    // {
    //     double result = 1.0;
    //     Eigen::MatrixXd gs = Algorithms::gram_schmidt(matrix);
    //     // for (const auto &vec : gs.colwise())
    //     // {
    //     //     result *= vec.norm();
    //     // }
    //     for (int i = 0; i < gs.cols(); i++)
    //     {
    //         Eigen::VectorXd vec = gs.col(i);
    //         result *= vec.norm();
    //     }
        
    //     return result;
    // }

    // Extended GCD algorithm, returns tuple of g, x, y such that xa + yb = g
    // @return std::tuple<cpp_int, cpp_int, cpp_int>
    // @param a first number
    // @param b second number
    std::tuple<mp::cpp_int, mp::cpp_int, mp::cpp_int> gcd_extended(mp::cpp_int a, mp::cpp_int b)
    {
        if (a == 0)
        {
            return std::make_tuple(b, 0, 1);
        }
        mp::cpp_int gcd, x1, y1;
        std::tie(gcd, x1, y1) = gcd_extended(b % a, a);

        mp::cpp_int x = y1 - (b / a) * x1;
        mp::cpp_int y = x1;

        return std::make_tuple(gcd, x, y);
    }

    // Function that translates Eigen Matrix to std::string for WolframAlpha checking
    // @return std::string
    // @param matrix input matrix
    std::string matrix_to_string(const Eigen::Matrix<mp::cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
    {
        int m = static_cast<int>(matrix.rows());
        int n = static_cast<int>(matrix.cols());
        if (m < 1 || n < 1)
        {
            throw std::invalid_argument("Matrix is not initialized");
        }
        if (matrix.isZero())
        {
            throw std::exception("Matrix is empty");
        }

        std::string result = "{";
        for (const auto &vec : matrix.colwise())
        {
            result += "{";
            for (const auto &elem : vec)
            {
                //result += std::to_string(static_cast<cpp_int>(elem)) + ", ";
                result += elem.str() + ", ";
            }
            result.pop_back();
            if (!result.empty())
            {
                result.pop_back();
                result += "}, ";
            }
        }
        result.pop_back();
        if (!result.empty())
        {
            result.pop_back();
            result += "}";
        }
        return result;
    }

    // bool check_linear_independency(const Eigen::Matrix<cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
    // {
    //     std::vector<int> inds = std::get<2>(get_linearly_independent_columns_by_gram_schmidt(matrix));

    //     if (inds.size() != matrix.rows())
    //     {
    //         return false;
    //     }
    //     return true;
    // }

    // Generates random array
    // @return Eigen::VectorXd
    // @param m number of rows, must be greater than one
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::ArrayXd generate_random_array(const int m, double lowest, double highest)
    {
        if (m < 1)
        {
            throw std::invalid_argument("Number of rows or columns should be greater than one");
        }
        if (highest - lowest < 1)
        {
            throw std::invalid_argument("highest parameter must be greater than lowest parameter by at least one");
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(lowest, highest + 1);

        Eigen::ArrayXd array = Eigen::ArrayXd::NullaryExpr(m, [&]()
                                                           { return dis(gen); });

        return array;
    }

    Eigen::VectorXd projection(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector)
    {
        Eigen::VectorXd projection = Eigen::VectorXd::Zero(vector.rows());

        for (const Eigen::VectorXd &matrix_row : matrix.rowwise())
        {
            projection += (vector.dot(matrix_row) / matrix_row.dot(matrix_row)) * matrix_row;
        }

        Eigen::VectorXd result = vector - projection;

        return result;
    }

    double distance_between_two_vectors(const Eigen::VectorXd &vector1, const Eigen::VectorXd &vector2)
    {
        return (vector1 - vector2).norm();
    }

    Eigen::VectorXd closest_vector(const std::vector<Eigen::VectorXd> &matrix, const Eigen::VectorXd &vector)
    {
        Eigen::VectorXd closest = matrix[0];
        for (auto const &v : matrix)
        {
            if (distance_between_two_vectors(vector, v) <= distance_between_two_vectors(vector, closest))
            {
                closest = v;
            }
        }
        return closest;
    }
}
