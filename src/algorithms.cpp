#include "algorithms.hpp"
#include <iostream>
#include "utils.hpp"
#include <vector>

namespace Algorithms
{
    namespace HNF
    {
        Eigen::MatrixXd HNF_full_row_rank(Eigen::MatrixXd B)
        {
            int m = B.rows();
            int n = B.cols();
            Eigen::MatrixXd B_stroke = Utils::get_linearly_independent_columns_by_gram_schmidt(B);

            double det = round(Utils::det_by_gram_schmidt(B_stroke));

            Eigen::MatrixXd H = Eigen::MatrixXd::Identity(m, m) * det;

            for (size_t i = 0; i < n; i++)
            {
                H = Utils::add_column(H, B_stroke.col(i));
            }

            return H;
        }

        Eigen::MatrixXd HNF(Eigen::MatrixXd B)
        {
            std::tuple<Eigen::MatrixXd, std::vector<int>> projection = Utils::get_linearly_independent_rows_by_gram_schmidt(B);
            Eigen::MatrixXd B_stroke = std::get<0>(projection);
            std::vector<int> inds = std::get<1>(projection);

            Eigen::MatrixXd B_stroke_transposed = B_stroke.transpose();

            Eigen::MatrixXd B_double_stroke = HNF_full_row_rank(B_stroke);

            std::vector<Eigen::VectorXd> basis;
            for (size_t i = 0; i < B_double_stroke.rows(); i++)
            {
                Eigen::VectorXd vec = B_double_stroke.row(i);
                basis.push_back(vec);
            }

            for (size_t i = 0; i < B.rows(); i++)
            {
                if (std::find(inds.begin(), inds.end(), i) == inds.end())
                {
                    Eigen::VectorXd vec = B.row(i);
                    Eigen::VectorXd x = B_stroke_transposed.colPivHouseholderQr().solve(vec);

                    Eigen::VectorXd result = Eigen::VectorXd::Zero(x.rows());
                    for (size_t j = 0; j < B_double_stroke.rows(); j++)
                    {
                        Eigen::VectorXd HNF_vec = B_double_stroke.row(j);
                        result += HNF_vec * x(j);
                    }
                    basis.push_back(result);
                }
            }
            Eigen::MatrixXd result(basis.size(), basis[0].rows());
            for (size_t i = 0; i < basis.size(); i++)
            {
                result.row(i) = basis[i];
            }
            return result;
        }
    }
    namespace CVP
    {
        Eigen::VectorXd greedy(Eigen::MatrixXd matrix, Eigen::VectorXd target)
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

        Eigen::VectorXd branch_and_bound(Eigen::MatrixXd matrix, Eigen::VectorXd target)
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
            for (size_t i = x_lowest; i <= x_highest; i++)
            {
                x_array.push_back(i);
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
    Eigen::MatrixXd gram_schmidt(Eigen::MatrixXd matrix, bool normalize, bool delete_zero_rows)
    {
        std::vector<Eigen::VectorXd> basis;
        Eigen::MatrixXd result(matrix.rows(), matrix.rows());

        for (const auto &vec : matrix.colwise())
        {
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());
            for (const auto &b : basis)
            {
                projections += (vec.dot(b) / b.dot(b)) * b;
            }
            Eigen::VectorXd result = vec - projections;
            if (delete_zero_rows)
            {
                bool is_all_zero = result.isZero(1e-3);
                if (!is_all_zero)
                {
                    if (normalize)
                    {
                        basis.push_back(result / result.norm());
                    }
                    else
                    {
                        basis.push_back(result);
                    }
                }
            }
            else
            {
                if (normalize)
                {
                    basis.push_back(result / result.norm());
                }
                else
                {
                    basis.push_back(result);
                }
            }
        }
        for (size_t i = 0; i < basis.size(); i++)
        {
            result.col(i) = basis[i];
        }
        return result;
    }
}