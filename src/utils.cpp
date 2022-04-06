#include "utils.hpp"
#include <iostream>
#include <random>
#include <functional>
#include <numeric>
#include <vector>
#include <stdexcept>
#include "algorithms.hpp"

namespace Utils
{
    // Function for computing HNF of full row rank matrix
    // @return Eigen::MatrixXd
    // @param H HNF
    // @param b column to be added
    Eigen::MatrixXd add_column(const Eigen::MatrixXd &H, const Eigen::ArrayXd &b_column)
    {
        if (H.rows() == 0)
        {
            return H;
        }

        double a = H(0, 0);
        Eigen::ArrayXd h = H.block(1, 0, H.rows() - 1, 1);
        Eigen::MatrixXd H_stroke = H.block(1, 1, H.rows() - 1, H.cols() - 1);
        double b = b_column(0);
        Eigen::ArrayXd b_stroke = b_column.tail(b_column.rows() - 1);
        std::tuple<int, int, int> gcd_result = gcd_extended(static_cast<int>(a), static_cast<int>(b));
        double g = static_cast<double>(std::get<0>(gcd_result));
        double x = static_cast<double>(std::get<1>(gcd_result));
        double y = static_cast<double>(std::get<2>(gcd_result));
        Eigen::Matrix2d U;
        U << x, -b / g, y, a / g;
        Eigen::MatrixXd temp_matrix(H.rows(), 2);
        temp_matrix.col(0) = Eigen::ArrayXd(H.col(0));
        temp_matrix.col(1) = b_column;
        Eigen::MatrixXd temp_result = temp_matrix * U;

        Eigen::ArrayXd h_stroke = temp_result.block(1, 0, temp_result.rows() - 1, 1);
        Eigen::ArrayXd b_double_stroke = temp_result.block(1, 1, temp_result.rows() - 1, 1);

        b_double_stroke = reduce(b_double_stroke, H_stroke);

        Eigen::MatrixXd H_double_stroke = add_column(H_stroke, b_double_stroke);

        h_stroke = reduce(h_stroke, H_double_stroke);

        Eigen::MatrixXd result(H.rows(), H.cols());
        result(0, 0) = g;
        result.block(1, 0, h_stroke.rows(), 1) = h_stroke;
        result.block(0, 1, 1, H_double_stroke.rows()).setZero();
        result.block(1, 1, H_double_stroke.rows(), H_double_stroke.cols()) = H_double_stroke;

        return result;
    }

    Eigen::ArrayXd reduce(const Eigen::ArrayXd &vector, const Eigen::MatrixXd &matrix)
    {
        Eigen::ArrayXd result = vector;
        for (int i = 0; i < result.rows(); i++)
        {
            Eigen::ArrayXd matrix_column = matrix.col(i);
            if (result(i) < 0)
            {
                double x = abs(ceil(result(i) / matrix(i, i))) + 1;
                result += matrix_column * x;
            }
            if (result(i) >= matrix(i, i))
            {
                double x = floor(result(i) / matrix(i, i));
                result -= matrix_column * x;
            }

        }
        return result;
    }
    
    // Generates random matrix with full row rank (or with all rows linearly independent)
    // @return Eigen::MatrixXd
    // @param m number of rows, must be greater than one and less than or equal to the parameter n
    // @param n number of columns, must be greater than one and greater than or equal to the parameter m 
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::MatrixXd generate_random_matrix_with_full_row_rank(const int m, const int n, double lowest, double highest)
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
        std::uniform_real_distribution<double> dis(lowest, highest + 1);

        Eigen::MatrixXd matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
                                                              { return double(int(dis(gen))); });

        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(matrix);
        auto rank = lu_decomp.rank();

        while (rank != m)
        {
            matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
                                                  { return double(int(dis(gen))); });

            lu_decomp.compute(matrix);
            rank = lu_decomp.rank();
        }

        return matrix;
    }

    // Generates random matrix
    // @return Eigen::MatrixXd
    // @param m number of rows, must be greater than one
    // @param n number of columns, must be greater than one
    // @param lowest lowest generated number, must be lower than lowest parameter by at least one
    // @param highest highest generated number, must be greater than lowest parameter by at least one
    Eigen::MatrixXd generate_random_matrix(const int m, const int n, double lowest, double highest)
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
        std::uniform_real_distribution<double> dis(lowest, highest + 1);

        Eigen::MatrixXd matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
                                                              { return double(int(dis(gen))); });

        return matrix;
    }

    // Returns matrix that consist of linearly independent columns of input matrix, othogonalized matrix and indexes of that columns in input matrix
    // @param matrix input matrix
    // @return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>>
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>> get_linearly_independent_columns_by_gram_schmidt(const Eigen::MatrixXd &matrix)
    {
        std::vector<Eigen::VectorXd> basis;
        std::vector<int> indexes;

        int counter = 0;
        for (const Eigen::VectorXd &vec : matrix.colwise())
        {
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());
            
            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis[i].data(), 0.0);
                double inner2 = std::inner_product(basis[i].data(), basis[i].data() + basis[i].size(), basis[i].data(), 0.0);
                projections.noalias() += (inner1 / inner2) * basis[i];
            }
            // NO PARALLEL
            // for (const auto &b : basis)
            // {
            //     double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), b.data(), 0.0);
            //     double inner2 = std::inner_product(b.data(), b.data() + b.size(), b.data(), 0.0);
            //     projections.noalias() += (inner1 / inner2) * b;
            // }

            Eigen::VectorXd result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                basis.push_back(result);
                indexes.push_back(counter);
            }
            counter++;
        }

        Eigen::MatrixXd result(matrix.rows(), indexes.size());
        Eigen::MatrixXd gram_schmidt(matrix.rows(), basis.size());
        #pragma omp parallel for
        for (int i = 0; i < indexes.size(); i++)
        {
            result.col(i) = matrix.col(indexes[i]);
            gram_schmidt.col(i) = basis[i];
        }
        return std::make_tuple(result, gram_schmidt, indexes);
    }

    // Returns matrix that consist of linearly independent rows of input matrix and indexes of that rows in input matrix
    // @param matrix input matrix
    // @return std::tuple<Eigen::MatrixXd, std::vector<int>>
    std::tuple<Eigen::MatrixXd, std::vector<int>> get_linearly_independent_rows_by_gram_schmidt(const Eigen::MatrixXd &matrix)
    {
        std::vector<Eigen::VectorXd> basis;
        std::vector<int> indexes;

        int counter = 0;
        for (const Eigen::VectorXd &vec : matrix.rowwise())
        {
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());

            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis[i].data(), 0.0);
                double inner2 = std::inner_product(basis[i].data(), basis[i].data() + basis[i].size(), basis[i].data(), 0.0);
                projections.noalias() += (inner1 / inner2) * basis[i];
            }
            // NO PARALLEL
            // for (const auto &b : basis)
            // {
            //     double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), b.data(), 0.0);
            //     double inner2 = std::inner_product(b.data(), b.data() + b.size(), b.data(), 0.0);
            //     projections.noalias() += (inner1 / inner2) * b;
            // }

            Eigen::VectorXd result = vec - projections;

            bool is_all_zero = result.isZero(1e-3);
            if (!is_all_zero)
            {
                basis.push_back(result);
                indexes.push_back(counter);
            }
            counter++;
        }

        Eigen::MatrixXd result(indexes.size(), matrix.cols());
        #pragma omp parallel for
        for (int i = 0; i < indexes.size(); i++)
        {
            result.row(i) = matrix.row(indexes[i]);
        }
        return std::make_tuple(result, indexes);
    }

    // Computes determinant by using Gram Schmidt orthogonalization
    // @return double
    // @param matrix input matrix
    double det_by_gram_schmidt(const Eigen::MatrixXd &matrix)
    {
        double result = 1.0;
        Eigen::MatrixXd gs = Algorithms::gram_schmidt(matrix);
        // for (const auto &vec : gs.colwise())
        // {
        //     result *= vec.norm();
        // }
        for (int i = 0; i < gs.cols(); i++)
        {
            Eigen::VectorXd vec = gs.col(i);
            result *= vec.norm();
        }
        
        return result;
    }

    // Extended GCD algorithm, returns tuple of g, x, y such that xa + yb = g
    // @return std::tuple<int, int, int>
    // @param a first number
    // @param b second number
    std::tuple<int, int, int> gcd_extended(int a, int b)
    {
        if (a == 0)
        {
            return std::make_tuple(b, 0, 1);
        }
        std::tuple<int, int, int> tuple = gcd_extended(b % a, a);
        int gcd = std::get<0>(tuple);
        int x1 = std::get<1>(tuple);
        int y1 = std::get<2>(tuple);

        int x = y1 - (b / a) * x1;
        int y = x1;

        return std::make_tuple(gcd, x, y);
    }

    bool check_linear_independency(const Eigen::MatrixXd &matrix)
    {
        std::vector<int> inds = std::get<2>(get_linearly_independent_columns_by_gram_schmidt(matrix));

        if (inds.size() != matrix.rows())
        {
            return false;
        }
        return true;
    }

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
