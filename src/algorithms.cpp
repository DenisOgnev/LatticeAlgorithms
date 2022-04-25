#include "algorithms.hpp"
#include <iostream>
#include "utils.hpp"
#include <vector>
#include <numeric>

namespace mp = boost::multiprecision;
typedef mp::number<mp::cpp_bin_float_100::backend_type, mp::et_off> cpp_bin_float_100_et_off;

namespace Algorithms
{
    namespace HNF
    {
        // Computes HNF of a matrix that is full row rank
        // @return Eigen::Matrix<cpp_int, -1, -1>
        // @param B full row rank matrix
        Eigen::Matrix<mp::cpp_int, -1, -1> HNF_full_row_rank(const Eigen::Matrix<mp::cpp_int, -1, -1> &B)
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
            if (B.isZero())
            {
                throw std::exception("Matrix is empty");
            }

            Eigen::Matrix<mp::cpp_int, -1, -1> B_stroke;
            Eigen::Matrix<mp::cpp_rational, -1, -1> ortogonalized;

            std::tuple<Eigen::Matrix<mp::cpp_int, -1, -1>, Eigen::Matrix<mp::cpp_rational, -1, -1>> result_of_gs = Utils::get_linearly_independent_columns_by_gram_schmidt(B);
            
            std::tie(B_stroke, ortogonalized) = result_of_gs;

            mp::cpp_rational t_det = 1.0;
            for (const Eigen::Vector<mp::cpp_rational, -1> &vec : ortogonalized.colwise())
            {
                t_det *= vec.squaredNorm();
            }
            mp::cpp_int det = mp::sqrt(mp::numerator(t_det));

            Eigen::Matrix<mp::cpp_int, -1, -1> H_temp = Eigen::Matrix<mp::cpp_int, -1, -1>::Identity(m, m) * det;

            for (int i = 0; i < n; i++)
            {
                H_temp = Utils::add_column(H_temp, B.col(i));
            }

            Eigen::Matrix<mp::cpp_int, -1, -1> H(m, n);
            H.block(0, 0, H_temp.rows(), H_temp.cols()) = H_temp;
            if (n > m)
            {
                H.block(0, H_temp.cols(), H_temp.rows(), n - m) = Eigen::Matrix<mp::cpp_int, -1, -1>::Zero(H_temp.rows(), n - m);
            }

            return H;
        }

        // Computes HNF of an arbitrary matrix
        // @return Eigen::Matrix<cpp_int, -1, -1>
        // @param B arbitrary matrix
        Eigen::Matrix<mp::cpp_int, -1, -1> HNF(const Eigen::Matrix<mp::cpp_int, -1, -1> &B)
        {
            int m = static_cast<int>(B.rows());
            int n = static_cast<int>(B.cols());

            if (m < 1 || n < 1)
            {
                throw std::invalid_argument("Matrix is not initialized");
            }
            if (B.isZero())
            {
                throw std::exception("Matrix is empty");
            }

            Eigen::Matrix<mp::cpp_int, -1, -1> B_stroke;
            std::vector<int> indicies;
            std::vector<int> deleted_indicies;
            Eigen::Matrix<mp::cpp_rational, -1, -1> T;
            std::tuple<Eigen::Matrix<mp::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<mp::cpp_rational, -1, -1>> projection = Utils::get_linearly_independent_rows_by_gram_schmidt(B);
            std::tie(B_stroke, indicies, deleted_indicies, T) = projection;

            Eigen::Matrix<mp::cpp_int, -1, -1> B_double_stroke = HNF_full_row_rank(B_stroke);
            
            Eigen::Matrix<mp::cpp_int, -1, -1> HNF(B.rows(), B.cols());

            for (int i = 0; i < indicies.size(); i++)
            {
                HNF.row(indicies[i]) = B_double_stroke.row(i);
            }

            Eigen::Matrix<mp::cpp_bin_float_double, -1, -1> t_HNF = HNF.cast<mp::cpp_bin_float_double>(); // for other way

            // First way, just find linear combinations of deleted rows, more accurate
            // Eigen::Matrix<mp::cpp_bin_float_double, -1, -1> B_stroke_transposed = B_stroke.transpose().cast<mp::cpp_bin_float_double>();
            // auto QR = B_stroke.cast<mp::cpp_bin_float_double>().colPivHouseholderQr().transpose();

            // for (const auto &indx : deleted_indicies)
            // {
            //     Eigen::Vector<mp::cpp_bin_float_double, -1> vec = B.row(indx).cast<mp::cpp_bin_float_double>();
            //     Eigen::RowVector<mp::cpp_bin_float_double, -1> x = QR.solve(vec);

            //     Eigen::Vector<mp::cpp_bin_float_double, -1> res = x * HNF.cast<mp::cpp_bin_float_double>();
            //     for (mp::cpp_bin_float_double &elem : res) 
            //     {
            //         elem = mp::round(elem);
            //     }
            //     HNF.row(indx) = res.cast<mp::cpp_int>();
            // }
            // return HNF;

            // Other, the "right" way, numerical errors +-2 
            for (const auto &indx : deleted_indicies)
            {
                Eigen::Vector<mp::cpp_bin_float_double, -1> res = Eigen::Vector<mp::cpp_bin_float_double, -1>::Zero(B.cols());
                for (int i = 0; i < indx; i++)
                {
                    res += T(indx, i).convert_to<mp::cpp_bin_float_double>() * t_HNF.row(i);
                }
                
                t_HNF.row(indx) = res;
            }

            return t_HNF.cast<mp::cpp_int>();
        }
    }
    namespace CVP
    {
        // bool first_time = true;
        // Eigen::MatrixXd gram_schmidt;
        // int index;
        // Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        // {
        //     if (matrix.rows() == 0)
        //     {
        //         first_time = true;
        //         return Eigen::VectorXd::Zero(matrix.cols());
        //     }
        //     if (first_time)
        //     {
        //         first_time = false;
        //         gram_schmidt = Algorithms::gram_schmidt(matrix, false);
        //         index = matrix.cols() - 1;
        //     }
        //     Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
        //     Eigen::MatrixXd B = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
        //     //Eigen::VectorXd b_star = Utils::projection(B, b);
        //     Eigen::VectorXd b_star = gram_schmidt.col(index);
        //     index--;
        //     double x = target.dot(b_star) / b_star.dot(b_star);
        //     double c = round(x);

        //     return c * b + Algorithms::CVP::greedy(B, target - c * b);
        // }
        Eigen::MatrixXd gram_schmidt;
        Eigen::MatrixXd B;
        int index;

        Eigen::VectorXd greedy_recursive(const Eigen::VectorXd &target)
        {
            if (index == 0)
            {
                return Eigen::VectorXd::Zero(target.rows());
            }
            index--;
            Eigen::VectorXd b = B.col(index);
            Eigen::VectorXd b_star = gram_schmidt.col(index);
            double x = target.dot(b_star) / b_star.dot(b_star);
            if (b_star.isZero(1e-3))
            {
                b_star = b;
                x = target.dot(b_star);
            }
            double c = round(x);

            return c * b + Algorithms::CVP::greedy_recursive(target - c * b);
        }

        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            B = matrix;
            gram_schmidt = Algorithms::gram_schmidt(matrix, false);
            index = static_cast<int>(matrix.cols());
            
            Eigen::VectorXd result = greedy_recursive(target);

            return result;
        }

        Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            if (matrix.rows() == 0)
            {
                return Eigen::VectorXd::Zero(matrix.cols());
            }
            Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
            Eigen::MatrixXd B = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
            Eigen::VectorXd b_star = Utils::projection(B, b);
            Eigen::VectorXd v = Algorithms::CVP::greedy(B, target);

            double upper_bound = (target - v).norm();
            double x_middle = std::floor(target.dot(b_star) / b_star.dot(b_star));
            double lower_bound = Utils::projection(B, target - x_middle * b).norm();

            double x = x_middle;
            double temp_lower_bound = lower_bound;
            while (temp_lower_bound <= upper_bound)
            {
                x += 1;
                temp_lower_bound = Utils::projection(B, target - x * b).norm();
            }
            double x_highest = x;

            x = x_middle;
            temp_lower_bound = lower_bound;
            while (temp_lower_bound <= upper_bound)
            {
                x -= 1;
                temp_lower_bound = Utils::projection(B, target - x * b).norm();
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
                Eigen::VectorXd res = elem * b + Algorithms::CVP::branch_and_bound(B, target - elem * b);
                v_array.push_back(res);
            }
            return Utils::closest_vector(v_array, target);
        }

        // Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        // {
        //     if (matrix.rows() == 0)
        //     {
        //         return Eigen::VectorXd::Zero(matrix.cols());
        //     }
        //     Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
        //     Eigen::MatrixXd B = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
        //     Eigen::VectorXd b_star = Utils::projection(B, b);
        //     Eigen::VectorXd v = Algorithms::CVP::greedy(B, target);

        //     double upper_bound = std::ceil((target - v).norm());
        //     double x_middle = std::floor(target.dot(b_star) / b_star.dot(b_star));
        //     double lower_bound = Utils::projection(B, target - x_middle * b).norm();

        //     double x = x_middle;
        //     double temp_lower_bound = lower_bound;
        //     while (temp_lower_bound <= upper_bound)
        //     {
        //         x += 1;
        //         temp_lower_bound = Utils::projection(B, target - x * b).norm();
        //     }
        //     double x_highest = x;

        //     x = x_middle;
        //     temp_lower_bound = lower_bound;
        //     while (temp_lower_bound <= upper_bound)
        //     {
        //         x -= 1;
        //         temp_lower_bound = Utils::projection(B, target - x * b).norm();
        //     }
        //     double x_lowest = x + 1;

        //     std::vector<int> x_array;
        //     for (int i =  static_cast<int>(x_lowest); i < x_highest; i++)
        //     {
        //         x_array.push_back(i);
        //     }
        //     if (x_array.size() == 0)
        //     {
        //         x_array.push_back(static_cast<int>(x_middle));
        //     }
        //     std::vector<Eigen::VectorXd> v_array;
        //     for (auto const &elem : x_array)
        //     {
        //         Eigen::VectorXd res = elem * b + Algorithms::CVP::branch_and_bound(B, target - elem * b);
        //         v_array.push_back(res);
        //     }
        //     return Utils::closest_vector(v_array, target);
        // }
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

            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                double inner1;
                double inner2;
                Eigen::MatrixXd basis_vector = basis[i];
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), 0.0);
                    }
                    #pragma omp section
                    {
                        inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), 0.0);
                    }
                }
                projections.noalias() += (inner1 / inner2) * basis_vector;
            }

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

        for (int i = 0; i < basis.size(); i++)
        {
            result.col(i) = basis[i];
        }

        return result;
    }
}