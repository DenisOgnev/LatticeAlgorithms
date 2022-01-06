#include "algorithms.hpp"
#include <iostream>
#include "utils.hpp"
#include <vector>

namespace Algorithms
{
    namespace HNF
    {
        Eigen::MatrixXd HNF_full_row_rank(const Eigen::MatrixXd &B)
        {
            int m = B.rows();
            int n = B.cols();
            Eigen::MatrixXd B_stroke = Utils::get_linearly_independent_columns_by_gram_schmidt(B);

            double det = round(Utils::det_by_gram_schmidt(B_stroke));

            Eigen::MatrixXd H_temp = Eigen::MatrixXd::Identity(m, m) * det;

            for (size_t i = 0; i < n; i++)
            {
                H_temp = Utils::add_column(H_temp, B.col(i));
            }

            if (n > m)
            {
                Eigen::MatrixXd H(m, n);
                H.block(0, 0, H_temp.rows(), H_temp.cols()) = H_temp;
                H.block(0, H_temp.cols(), H_temp.rows(), n - m) = Eigen::MatrixXd::Zero(H_temp.rows(), n - m);

                return H;
            }

            return H_temp;
        }

        Eigen::MatrixXd HNF(const Eigen::MatrixXd &B)
        {
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
    Eigen::MatrixXd gram_schmidt(const Eigen::MatrixXd &matrix, bool normalize, bool delete_zero_rows)
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