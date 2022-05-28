#include "utils.hpp"
#include <iostream>
#include <random>
#include <functional>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <string> 
#include <chrono>
#include <algorithm>
#include <thread>
#include "algorithms.hpp"

namespace mp = boost::multiprecision;

namespace Utils
{
    // Function for computing HNF of full row rank matrix
    // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
    // @param H HNF
    // @param b column to be added
    Eigen::Matrix<mp::cpp_int, -1, -1> add_column(const Eigen::Matrix<mp::cpp_int, -1, -1> &H, const Eigen::Vector<mp::cpp_int, -1> &b_column)
    {
        if (H.rows() == 0)
        {
            return H;
        }

        Eigen::Vector<mp::cpp_int, -1> H_first_col = H.col(0);

        mp::cpp_int a = H_first_col(0);
        Eigen::Vector<mp::cpp_int, -1> h = H_first_col.tail(H_first_col.rows() - 1);
        Eigen::Matrix<mp::cpp_int, -1, -1> H_stroke = H.block(1, 1, H.rows() - 1, H.cols() - 1);
        mp::cpp_int b = b_column(0);
        Eigen::Vector<mp::cpp_int, -1> b_stroke = b_column.tail(b_column.rows() - 1);

        std::tuple<mp::cpp_int, mp::cpp_int, mp::cpp_int> gcd_result = gcd_extended(a, b);
        mp::cpp_int g, x, y;
        std::tie(g, x, y) = gcd_result;

        Eigen::Matrix<mp::cpp_int, 2, 2> U;
        U << x, -b / g, y, a / g;
        

        Eigen::Matrix<mp::cpp_int, -1, 2> temp_matrix(H.rows(), 2);
        temp_matrix.col(0) = H_first_col;
        temp_matrix.col(1) = b_column;
        Eigen::Matrix<mp::cpp_int, -1, 2> temp_result = temp_matrix * U;

        Eigen::Vector<mp::cpp_int, -1> h_stroke = temp_result.col(0).tail(temp_result.rows() - 1);
        Eigen::Vector<mp::cpp_int, -1> b_double_stroke = temp_result.col(1).tail(temp_result.rows() - 1);

        b_double_stroke = reduce(b_double_stroke, H_stroke);

        Eigen::Matrix<mp::cpp_int, -1, -1> H_double_stroke = add_column(H_stroke, b_double_stroke);

        h_stroke = reduce(h_stroke, H_double_stroke);

        Eigen::Matrix<mp::cpp_int, -1, -1> result(H.rows(), H.cols());

        result(0, 0) = g;
        result.col(0).tail(result.cols() - 1) = h_stroke;
        result.row(0).tail(result.rows() - 1).setZero();
        result.block(1, 1, H_double_stroke.rows(), H_double_stroke.cols()) = H_double_stroke;

        return result;
    }

    
    // Function for computing HNF, reduces elements of vector modulo diagonal elements of matrix
    // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
    // @param vector vector to be reduced
    // @param matrix input matrix
    Eigen::Vector<mp::cpp_int, -1> reduce(const Eigen::Vector<mp::cpp_int, -1> &vector, const Eigen::Matrix<mp::cpp_int, -1, -1> &matrix)
    {
        Eigen::Vector<mp::cpp_int, -1> result = vector;
        for (int i = 0; i < result.rows(); i++)
        {
            Eigen::Vector<mp::cpp_int, -1> matrix_column = matrix.col(i);
            mp::cpp_int t_vec_elem = result(i);
            mp::cpp_int t_matrix_elem = matrix(i, i);

            mp::cpp_int x;
            if (t_vec_elem >= 0)
            {
                x = (t_vec_elem / t_matrix_elem);
            }
            else
            {
                x = (t_vec_elem - (t_matrix_elem - 1)) / t_matrix_elem;
            }

            result -= matrix_column * x;
        }
        return result;
    }

    
    // Generates random matrix with full row rank
    // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
    // @param m number of rows, must be greater than one and less than or equal to the parameter n
    // @param n number of columns, must be greater than one and greater than or equal to the parameter m 
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::Matrix<mp::cpp_int, -1, -1> generate_random_matrix_with_full_row_rank(const int m, const int n, int lowest, int highest)
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

        Eigen::Matrix<int, -1, -1> matrix = Eigen::Matrix<int, -1, -1>::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(matrix.cast<double>());
        auto rank = lu_decomp.rank();

        while (rank != m)
        {
            matrix = Eigen::Matrix<int, -1, -1>::NullaryExpr(m, n, [&]()
                                                  { return dis(gen); });

            lu_decomp.compute(matrix.cast<double>());
            rank = lu_decomp.rank();
        }

        return matrix.cast<mp::cpp_int>();
    }


    // Generates random matrix
    // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
    // @param m number of rows, must be greater than one
    // @param n number of columns, must be greater than one
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::Matrix<mp::cpp_int, -1, -1> generate_random_matrix(const int m, const int n, int lowest, int highest)
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

        Eigen::Matrix<int, -1, -1> matrix = Eigen::Matrix<int, -1, -1>::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        return matrix.cast<mp::cpp_int>();
    }


    // Returns matrix that consist of linearly independent columns of input matrix and othogonalized matrix
    // @param matrix input matrix
    // @return std::tuple<Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>, Eigen::Matrix<boost::multiprecision::cpp_rational, -1, -1>>
    std::tuple<Eigen::Matrix<mp::cpp_int, -1, -1>, Eigen::Matrix<mp::cpp_rational, -1, -1>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::Matrix<mp::cpp_int, -1, -1> &matrix)
    {
        std::vector<Eigen::Vector<mp::cpp_rational, -1>> basis;
        std::vector<int> indexes;

        int counter = 0;
        for (const Eigen::Vector<mp::cpp_rational, -1> &vec : matrix.cast<mp::cpp_rational>().colwise())
        {
            Eigen::Vector<mp::cpp_rational, -1> projections = Eigen::Vector<mp::cpp_rational, -1>::Zero(vec.size());

            for (int i = 0; i < basis.size(); i++)
            {
                Eigen::Vector<mp::cpp_rational, -1> basis_vector = basis[i];
                mp::cpp_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), mp::cpp_rational(0.0));
                mp::cpp_rational inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), mp::cpp_rational(0.0));
                    
                mp::cpp_rational coef = inner1 / inner2;

                projections += basis_vector * coef;
            }

            Eigen::Vector<mp::cpp_rational, -1> result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                basis.push_back(result);
                indexes.push_back(counter);
            }
            counter++;
        }

        Eigen::Matrix<mp::cpp_int, -1, -1> result(matrix.rows(), indexes.size());
        Eigen::Matrix<mp::cpp_rational, -1, -1> gram_schmidt(matrix.rows(), basis.size());
        
        for (int i = 0; i < indexes.size(); i++)
        {
            result.col(i) = matrix.col(indexes[i]);
            gram_schmidt.col(i) = basis[i];
        }
        return std::make_tuple(result, gram_schmidt);
    }

    
    // Returns matrix that consist of linearly independent rows of input matrix, indicies of that rows in input matrix, indices of deleted rows and martix T, that consists of Gram Schmidt coefficients
    // @param matrix input matrix
    // @return std::tuple<Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>>
    std::tuple<Eigen::Matrix<mp::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<mp::cpp_rational, -1, -1>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::Matrix<mp::cpp_int, -1, -1> &matrix)
    {
        std::vector<Eigen::Vector<mp::cpp_rational, -1>> basis;
        std::vector<int> indicies;
	    std::vector<int> deleted_indicies;
        Eigen::Matrix<mp::cpp_rational, -1, -1> T = Eigen::Matrix<mp::cpp_rational, -1, -1>::Identity(matrix.rows(), matrix.rows());

        int counter = 0;
        for (const Eigen::Vector<mp::cpp_rational, -1> &vec : matrix.cast<mp::cpp_rational>().rowwise())
        {
            Eigen::Vector<mp::cpp_rational, -1> projections = Eigen::Vector<mp::cpp_rational, -1>::Zero(vec.size());

            for (int i = 0; i < basis.size(); i++)
            {
                Eigen::Vector<mp::cpp_rational, -1> basis_vector = basis[i];
                mp::cpp_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), mp::cpp_rational(0.0));
                mp::cpp_rational inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), mp::cpp_rational(0.0));
        
                mp::cpp_rational u_ij = 0;
                if (!inner1.is_zero())
                {
                    u_ij = inner1 / inner2;

                    projections += u_ij * basis_vector;
                    T(counter, i) = u_ij;
                }
            }

            Eigen::Vector<mp::cpp_rational, -1> result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                indicies.push_back(counter);
            }
            else
            {
                deleted_indicies.push_back(counter);
            }
            basis.push_back(result);
            counter++;
        }

        Eigen::Matrix<mp::cpp_int, -1, -1> result(indicies.size(), matrix.cols());
        for (int i = 0; i < indicies.size(); i++)
        {
            result.row(i) = matrix.row(indicies[i]);
        }
        return std::make_tuple(result, indicies, deleted_indicies, T);
    }


    // Extended GCD algorithm, returns tuple of g, x, y such that xa + yb = g
    // @return std::tuple<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int, boost::multiprecision::cpp_int>
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

    #ifdef GMP
    // Function for computing HNF of full row rank matrix
    // @return Eigen::Matrix<boost::multiprecision::cpp_mpz, -1, -1>
    // @param H HNF
    // @param b column to be added
    Eigen::Matrix<mp::mpz_int, -1, -1> add_column_GMP(const Eigen::Matrix<mp::mpz_int, -1, -1> &H, const Eigen::Vector<mp::mpz_int, -1> &b_column)
    {
        if (H.rows() == 0)
        {
            return H;
        }

        Eigen::Vector<mp::mpz_int, -1> H_first_col = H.col(0);

        mp::mpz_int a = H_first_col(0);
        Eigen::Vector<mp::mpz_int, -1> h = H_first_col.tail(H_first_col.rows() - 1);
        Eigen::Matrix<mp::mpz_int, -1, -1> H_stroke = H.block(1, 1, H.rows() - 1, H.cols() - 1);
        mp::mpz_int b = b_column(0);
        Eigen::Vector<mp::mpz_int, -1> b_stroke = b_column.tail(b_column.rows() - 1);

        std::tuple<mp::mpz_int, mp::mpz_int, mp::mpz_int> gcd_result = gcd_extended_GMP(a, b);
        mp::mpz_int g, x, y;
        std::tie(g, x, y) = gcd_result;

        Eigen::Matrix<mp::mpz_int, 2, 2> U;
        U << x, -b / g, y, a / g;
        

        Eigen::Matrix<mp::mpz_int, -1, 2> temp_matrix(H.rows(), 2);
        temp_matrix.col(0) = H_first_col;
        temp_matrix.col(1) = b_column;
        Eigen::Matrix<mp::mpz_int, -1, 2> temp_result = temp_matrix * U;

        Eigen::Vector<mp::mpz_int, -1> h_stroke = temp_result.col(0).tail(temp_result.rows() - 1);
        Eigen::Vector<mp::mpz_int, -1> b_double_stroke = temp_result.col(1).tail(temp_result.rows() - 1);

        b_double_stroke = reduce_GMP(b_double_stroke, H_stroke);

        Eigen::Matrix<mp::mpz_int, -1, -1> H_double_stroke = add_column_GMP(H_stroke, b_double_stroke);

        h_stroke = reduce_GMP(h_stroke, H_double_stroke);

        Eigen::Matrix<mp::mpz_int, -1, -1> result(H.rows(), H.cols());

        result(0, 0) = g;
        result.col(0).tail(result.cols() - 1) = h_stroke;
        result.row(0).tail(result.rows() - 1).setZero();
        result.block(1, 1, H_double_stroke.rows(), H_double_stroke.cols()) = H_double_stroke;

        return result;
    }


    // Function for computing HNF, reduces elements of vector modulo diagonal elements of matrix
    // @return Eigen::Matrix<boost::multiprecision::cpp_mpz, -1, -1>
    // @param vector vector to be reduced
    // @param matrix input matrix
    Eigen::Vector<mp::mpz_int, -1> reduce_GMP(const Eigen::Vector<mp::mpz_int, -1> &vector, const Eigen::Matrix<mp::mpz_int, -1, -1> &matrix)
    {
        Eigen::Vector<mp::mpz_int, -1> result = vector;
        for (int i = 0; i < result.rows(); i++)
        {
            Eigen::Vector<mp::mpz_int, -1> matrix_column = matrix.col(i);
            mp::mpz_int t_vec_elem = result(i);
            mp::mpz_int t_matrix_elem = matrix(i, i);

            mp::mpz_int x;
            if (t_vec_elem >= 0)
            {
                x = (t_vec_elem / t_matrix_elem);
            }
            else
            {
                x = (t_vec_elem - (t_matrix_elem - 1)) / t_matrix_elem;
            }

            result -= matrix_column * x;
        }
        return result;
    }


    // Generates random matrix with full row rank (all rows are linearly independent)
    // @return Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>
    // @param m number of rows, must be greater than one and less than or equal to the parameter n
    // @param n number of columns, must be greater than one and greater than or equal to the parameter m 
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::Matrix<mp::mpz_int, -1, -1> generate_random_matrix_with_full_row_rank_GMP(const int m, const int n, int lowest, int highest)
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

        Eigen::Matrix<int, -1, -1> matrix = Eigen::Matrix<int, -1, -1>::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(matrix.cast<double>());
        auto rank = lu_decomp.rank();

        while (rank != m)
        {
            matrix = Eigen::Matrix<int, -1, -1>::NullaryExpr(m, n, [&]()
                                                  { return dis(gen); });

            lu_decomp.compute(matrix.cast<double>());
            rank = lu_decomp.rank();
        }

        return matrix.cast<mp::mpz_int>();
    }


    // Generates random matrix
    // @return Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>
    // @param m number of rows, must be greater than one
    // @param n number of columns, must be greater than one
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::Matrix<mp::mpz_int, -1, -1> generate_random_matrix_GMP(const int m, const int n, int lowest, int highest)
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

        Eigen::Matrix<int, -1, -1> matrix = Eigen::Matrix<int, -1, -1>::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        return matrix.cast<mp::mpz_int>();
    }


    // Returns matrix that consist of linearly independent columns of input matrix and othogonalized matrix
    // @param matrix input matrix
    // @return std::tuple<Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>, Eigen::Matrix<boost::multiprecision::mpq_rational, -1, -1>>
    std::tuple<Eigen::Matrix<mp::mpz_int, -1, -1>, Eigen::Matrix<mp::mpq_rational, -1, -1>> get_linearly_independent_columns_by_gram_schmidt_GMP(const Eigen::Matrix<mp::mpz_int, -1, -1> &matrix)
    {
        std::vector<Eigen::Vector<mp::mpq_rational, -1>> basis;
        std::vector<int> indexes;

        int counter = 0;
        for (const Eigen::Vector<mp::mpq_rational, -1> &vec : matrix.cast<mp::mpq_rational>().colwise())
        {
            Eigen::Vector<mp::mpq_rational, -1> projections = Eigen::Vector<mp::mpq_rational, -1>::Zero(vec.size());

            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                Eigen::Vector<mp::mpq_rational, -1> basis_vector = basis[i];
                mp::mpq_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), mp::mpq_rational(0.0));
                mp::mpq_rational inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), mp::mpq_rational(0.0));
                    
                mp::mpq_rational coef = inner1 / inner2;   
                #pragma omp critical
                projections += basis_vector * coef;
            }

            Eigen::Vector<mp::mpq_rational, -1> result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                basis.push_back(result);
                indexes.push_back(counter);
            }
            counter++;
        }

        Eigen::Matrix<mp::mpz_int, -1, -1> result(matrix.rows(), indexes.size());
        Eigen::Matrix<mp::mpq_rational, -1, -1> gram_schmidt(matrix.rows(), basis.size());
        
        for (int i = 0; i < indexes.size(); i++)
        {
            result.col(i) = matrix.col(indexes[i]);
            gram_schmidt.col(i) = basis[i];
        }
        return std::make_tuple(result, gram_schmidt);
    }


    // Returns matrix that consist of linearly independent rows of input matrix, indicies of that rows in input matrix, indices of deleted rows and martix T, that consists of Gram Schmidt coefficients
    // @param matrix input matrix
    // @return std::tuple<Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>>
    std::tuple<Eigen::Matrix<mp::mpz_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<mp::mpq_rational, -1, -1>> get_linearly_independent_rows_by_gram_schmidt_GMP(const Eigen::Matrix<mp::mpz_int, -1, -1> &matrix)
    {
        std::vector<Eigen::Vector<mp::mpq_rational, -1>> basis;
        std::vector<int> indicies;
	    std::vector<int> deleted_indicies;
        Eigen::Matrix<mp::mpq_rational, -1, -1> T = Eigen::Matrix<mp::mpq_rational, -1, -1>::Identity(matrix.rows(), matrix.rows());

        int counter = 0;
        for (const Eigen::Vector<mp::mpq_rational, -1> &vec : matrix.cast<mp::mpq_rational>().rowwise())
        {
            Eigen::Vector<mp::mpq_rational, -1> projections = Eigen::Vector<mp::mpq_rational, -1>::Zero(vec.size());

            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                Eigen::Vector<mp::mpq_rational, -1> basis_vector = basis[i];
                mp::mpq_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), mp::mpq_rational(0.0));
                mp::mpq_rational inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), mp::mpq_rational(0.0));
        
                mp::mpq_rational u_ij = 0;
                if (!inner1.is_zero())
                {
                    u_ij = inner1 / inner2;
                    #pragma omp critical
                    {
                        projections += u_ij * basis_vector;
                        T(counter, i) = u_ij;
                    }
                }
            }

            Eigen::Vector<mp::mpq_rational, -1> result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                indicies.push_back(counter);
            }
            else
            {
                deleted_indicies.push_back(counter);
            }
            basis.push_back(result);
            counter++;
        }

        Eigen::Matrix<mp::mpz_int, -1, -1> result(indicies.size(), matrix.cols());
        for (int i = 0; i < indicies.size(); i++)
        {
            result.row(i) = matrix.row(indicies[i]);
        }
        return std::make_tuple(result, indicies, deleted_indicies, T);
    }


    // Extended GCD algorithm, returns tuple of g, x, y such that xa + yb = g
    // @return std::tuple<boost::multiprecision::mpz_int, boost::multiprecision::mpz_int, boost::multiprecision::mpz_int>
    // @param a first number
    // @param b second number
    std::tuple<mp::mpz_int, mp::mpz_int, mp::mpz_int> gcd_extended_GMP(mp::mpz_int a, mp::mpz_int b)
    {
        if (a == 0)
        {
            return std::make_tuple(b, 0, 1);
        }
        mp::mpz_int gcd, x1, y1;
        std::tie(gcd, x1, y1) = gcd_extended_GMP(b % a, a);

        mp::mpz_int x = y1 - (b / a) * x1;
        mp::mpz_int y = x1;

        return std::make_tuple(gcd, x, y);
    }
    #endif


    // Generates random matrix with full column rank
    // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
    // @param m number of rows, must be greater than one and greater than or equal to the parameter n
    // @param n number of columns, must be greater than one and lower than or equal to the parameter m 
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::MatrixXd generate_random_matrix_with_full_column_rank(const int m, const int n, int lowest, int highest)
    {
        if (m < n)
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

        Eigen::MatrixXd matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
                                                              { return dis(gen); });

        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(matrix);
        auto rank = lu_decomp.rank();

        while (rank != n)
        {
            matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
                                                  { return dis(gen); });

            lu_decomp.compute(matrix);
            rank = lu_decomp.rank();
        }

        return matrix;
    }   


    // Generates random vector
    // @return Eigen::VectorXd
    // @param m number of rows, must be greater than one
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::VectorXd generate_random_vector(const int m, double lowest, double highest)
    {
        if (m < 1)
        {
            throw std::invalid_argument("Number of rows or columns should be greater than one");
        }
        if (highest - lowest < 1)
        {
            throw std::invalid_argument("highest parameter must be greater than lowest parameter by at least one");
        }
        std::mt19937 gen(std::random_device{}());
        
        std::uniform_real_distribution<double> dis(lowest, highest);

        Eigen::VectorXd vector = Eigen::VectorXd::NullaryExpr(m, [&]()
                                                           { return dis(gen); });

        return vector;
    }


    // Computes component of a vector perpendicular to a matrix using equations from Gram Schmidt computing
    // @return Eigen::VectorXd
    // @param matrix input matrix
    // @param vector input vector
    Eigen::VectorXd projection(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &vector)
    {
        Eigen::MatrixXd t_matrix(matrix.rows(), matrix.cols() + 1);
        t_matrix << matrix, vector;
        std::vector<Eigen::VectorXd> basis;

        for (const Eigen::VectorXd &vec : t_matrix.colwise())
        {
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());

            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                Eigen::VectorXd basis_vector = basis[i];
                double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), 0.0);
                double inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), 0.0);
                
                double coef = inner1 / inner2;
                #pragma omp critical
                projections += basis_vector * coef;
            }

            Eigen::VectorXd t_result = vec - projections;

            basis.push_back(t_result);
        }

        Eigen::VectorXd result = basis[basis.size() - 1];

        return result;
    }


    // Finds vector that is closest to other vectors in matrix
    // @return Eigen::VectorXd
    // @param matrix input matrix
    // @param vector input vector
    Eigen::VectorXd closest_vector(const std::vector<Eigen::VectorXd> &matrix, const Eigen::VectorXd &vector)
    {
        Eigen::VectorXd closest = matrix[0];
        for (const auto &v : matrix)
        {
            if ((vector - v).norm() <= (vector - closest).norm())
            {
                closest = v;
            }
        }
        return closest;
    }
}
