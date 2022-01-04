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