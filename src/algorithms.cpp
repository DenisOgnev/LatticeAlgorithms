#include "algorithms.hpp"
#include <iostream>
#include "utils.hpp"
#include <vector>
#include <numeric>

namespace Algorithms
{
    namespace HNF
    {
        // Computes HNF of a matrix that is full row rank
        // @return Eigen::MatrixXd
        // @param B full row rank matrix
        Eigen::MatrixXd HNF_full_row_rank(const Eigen::MatrixXd &B)
        {
            int m = static_cast<int>(B.rows());
            int n = static_cast<int>(B.cols());

            if (m > n)
            {
                throw std::invalid_argument("m must be less than or equal n");
            }
            if (m < 1 || n < 1)
            {
                throw std::invalid_argument("Matrix is not initialized");
            }
            if (B.isZero(1e-3))
            {
                throw std::exception("Matrix is empty");
            }

            std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>> result_of_gs = Utils::get_linearly_independent_columns_by_gram_schmidt(B);
            Eigen::MatrixXd B_stroke = std::get<0>(result_of_gs);
            Eigen::MatrixXd ortogonalized = std::get<1>(result_of_gs);
            
            double det = 1.0;
            for (const auto &vec : ortogonalized.colwise())
            {
                det *= vec.norm();
            }

            Eigen::MatrixXd H_temp = Eigen::MatrixXd::Identity(m, m) * det;

            for (int i = 0; i < n; i++)
            {
                H_temp = Utils::add_column(H_temp, B.col(i));
            }

            Eigen::MatrixXd H(m, n);
            H.block(0, 0, H_temp.rows(), H_temp.cols()) = H_temp;
            if (n > m)
            {
                H.block(0, H_temp.cols(), H_temp.rows(), n - m) = Eigen::MatrixXd::Zero(H_temp.rows(), n - m);
            }

            return H;
        }

        // Computes HNF of an arbitrary matrix
        // @return Eigen::MatrixXd
        // @param B arbitrary matrix
        Eigen::MatrixXd HNF(const Eigen::MatrixXd &B)
        {
            int m = static_cast<int>(B.rows());
            int n = static_cast<int>(B.cols());

            if (m < 1 || n < 1)
            {
                throw std::invalid_argument("Matrix is not initialized");
            }
            if (B.isZero(1e-3))
            {
                throw std::exception("Matrix is empty");
            }

            std::tuple<Eigen::MatrixXd, std::vector<int>> projection = Utils::get_linearly_independent_rows_by_gram_schmidt(B);
            Eigen::MatrixXd B_stroke = std::get<0>(projection);
            std::vector<int> inds = std::get<1>(projection);

            Eigen::MatrixXd B_stroke_transposed = B_stroke.transpose();

            Eigen::MatrixXd B_double_stroke = HNF_full_row_rank(B_stroke);

            std::vector<Eigen::VectorXd> basis;
            for (const Eigen::VectorXd &vec : B_double_stroke.rowwise())
            {
                basis.push_back(vec);
            }

            int counter = 0;
            for (const Eigen::VectorXd &vec : B.rowwise())
            {
                if (std::find(inds.begin(), inds.end(), counter) == inds.end())
                {
                    Eigen::VectorXd x = B_stroke_transposed.colPivHouseholderQr().solve(vec);

                    Eigen::VectorXd result = Eigen::VectorXd::Zero(x.rows());
                    int second_counter = 0;
                    for (const Eigen::VectorXd &HNF_vec : B_double_stroke.rowwise())
                    {
                        result += HNF_vec * x(second_counter);
                        second_counter++;
                    }
                    basis.push_back(result);
                }
                counter++;
            }
            Eigen::MatrixXd result(basis.size(), basis[0].rows());
            for (int i = 0; i < basis.size(); i++)
            {
                result.row(i) = basis[i];
            }
            return result;
        }
    }
    namespace CVP
    {
        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            if (matrix.rows() == 0)
            {
                return Eigen::VectorXd::Zero(matrix.cols());
            }
            Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
            Eigen::MatrixXd mat = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
            Eigen::VectorXd b_star = Utils::projection(mat, b);
            double x = target.dot(b_star) / b_star.dot(b_star);
            double c = round(x);

            return c * b + Algorithms::CVP::greedy(mat, target - c * b);
        }

        Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            if (matrix.rows() == 0)
            {
                return Eigen::VectorXd::Zero(matrix.cols());
            }
            Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
            Eigen::MatrixXd mat = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
            Eigen::VectorXd b_star = Utils::projection(mat, b);
            Eigen::VectorXd v = Algorithms::CVP::greedy(mat, target);

            double upper_bound = std::ceil((target - v).norm());
            double x_middle = std::floor(target.dot(b_star) / b_star.dot(b_star));
            double lower_bound = Utils::projection(mat, target - x_middle * b).norm();

            double x = x_middle;
            double temp_lower_bound = lower_bound;
            while (temp_lower_bound <= upper_bound)
            {
                x += 1;
                temp_lower_bound = Utils::projection(mat, target - x * b).norm();
            }
            double x_highest = x;

            x = x_middle;
            temp_lower_bound = lower_bound;
            while (temp_lower_bound <= upper_bound)
            {
                x -= 1;
                temp_lower_bound = Utils::projection(mat, target - x * b).norm();
            }
            double x_lowest = x + 1;

            std::vector<int> x_array;
            for (int i =  static_cast<int>(x_lowest); i < x_highest; i++)
            {
                x_array.push_back(i);
            }
            if (x_array.size() == 0)
            {
                x_array.push_back(static_cast<int>(x_middle));
            }
            std::vector<Eigen::VectorXd> v_array;
            for (auto const &elem : x_array)
            {
                Eigen::VectorXd res = elem * b + Algorithms::CVP::branch_and_bound(mat, target - elem * b);
                v_array.push_back(res);
            }
            return Utils::closest_vector(v_array, target);
        }
    }
    // Computes Gram Schmidt orthogonalization
    // @return Eigen::MatrixXd
    // @param matrix input matrix
    // @param normalize indicates that should we or not normalize output values
    // @param delete_zero_rows indicates that should we or not delete zero rows
    Eigen::MatrixXd gram_schmidt(const Eigen::MatrixXd &matrix, bool delete_zero_rows)
    {
        std::vector<Eigen::VectorXd> basis;

        for (const auto &vec : matrix.colwise())
        {
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());

            //#pragma omp parallel for
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

            if (delete_zero_rows)
            {
                bool is_all_zero = result.isZero(1e-3);
                if (!is_all_zero)
                {
                    basis.push_back(result);
                }
            }
            else
            {
                basis.push_back(result);
            }
        }

        Eigen::MatrixXd result(matrix.rows(), basis.size());
        #pragma omp parallel for
        for (int i = 0; i < basis.size(); i++)
        {
            result.col(i) = basis[i];
        }
        return result;
    }
}