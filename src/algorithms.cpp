#include "algorithms.hpp"
#include <iostream>
#include "utils.hpp"
#include <vector>
#include <numeric>

namespace mp = boost::multiprecision;

namespace Algorithms
{
    namespace HNF
    {
        // Computes HNF of a integer matrix that is full row rank
        // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
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

        // Computes HNF of an arbitrary integer matrix
        // @return Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1>
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

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // First way: just find linear combinations of deleted rows. More accurate

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
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Other, the "right" way that is desribed in algorithm. Have small numerical errors 
            Eigen::Matrix<mp::cpp_bin_float_double, -1, -1> t_HNF = HNF.cast<mp::cpp_bin_float_double>();
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
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
    }
    namespace CVP
    {
        Eigen::MatrixXd gram_schmidt_greedy;
        Eigen::MatrixXd B_greedy;
        int index_greedy;

        Eigen::MatrixXd gram_schmidt_bb;
        
        // Recursive body of greedy algorithm
        // @return Eigen::VectorXd
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd greedy_recursive(const Eigen::VectorXd &target)
        {
            if (index_greedy == 0)
            {
                return Eigen::VectorXd::Zero(target.rows());
            }
            index_greedy--;
            Eigen::VectorXd b = B_greedy.col(index_greedy);
            Eigen::VectorXd b_star = gram_schmidt_greedy.col(index_greedy);
            double x = target.dot(b_star) / b_star.dot(b_star);
            double c = std::round(x);

            return c * b + Algorithms::CVP::greedy_recursive(target - c * b);
        }


        // Solves CVP using a greedy algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            B_greedy = matrix;
            gram_schmidt_greedy = Algorithms::gram_schmidt(matrix, false);
            index_greedy = static_cast<int>(matrix.cols());

            return greedy_recursive(target);
        }

        // Recursive body of branch and bound algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd branch_and_bound_recursive(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            if (matrix.cols() == 0)
            {
                return Eigen::VectorXd::Zero(target.rows());
            }
            Eigen::MatrixXd B = matrix.block(0, 0, matrix.rows(), matrix.cols() - 1);
            Eigen::VectorXd b = matrix.col(matrix.cols() - 1);
            Eigen::VectorXd b_star = gram_schmidt_bb.col(matrix.cols() - 1);

            Eigen::VectorXd v = Algorithms::CVP::greedy(B, target);
            double upper_bound = (target - v).norm();

            double x_middle = std::round(target.dot(b_star) / b_star.dot(b_star));

            std::vector<int> X;
            X.push_back(static_cast<int>(x_middle));

            bool flag1 = true;
            bool flag2 = true;

            double x1 = x_middle + 1;
            double x2 = x_middle - 1;
            while (flag1 || flag2)
            {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        if (flag1 && Utils::projection(B, target - x1 * b).norm() <= upper_bound)
                        {
                            #pragma omp critical
                            X.push_back(static_cast<int>(x1));
                            x1++;
                        } 
                        else
                        {
                            flag1 = false;
                        }
                    }
                    #pragma omp section
                    {
                        if (flag2 && Utils::projection(B, target - x2 * b).norm() <= upper_bound)
                        {
                            #pragma omp critical
                            X.push_back(static_cast<int>(x2));
                            x2--;
                        } 
                        else 
                        {
                            flag2 = false;
                        }
                    }
                }
            }

            // #pragma omp parallel sections 
            // {
            //     #pragma omp section
            //     {
            //         double x = x_middle + 1;
            //         while (Utils::projection(B, target - x * b).norm() <= upper_bound)
            //         {
            //             #pragma omp critical
            //             X.push_back(static_cast<int>(x));
            //             x++;
            //         }   
            //     }
            //     #pragma omp section
            //     {
            //         double x = x_middle - 1;
            //         while (Utils::projection(B, target - x * b).norm() <= upper_bound)
            //         {
            //             #pragma omp critical
            //             X.push_back(static_cast<int>(x));
            //             x--;
            //         }
            //     }
            // }

            // double x = x_middle + 1;
            // while (Utils::projection(B, target - x * b).norm() <= upper_bound)
            // {
            //     X.push_back(static_cast<int>(x));
            //     x++;
            // }   
            // x = x_middle - 1;
            // while (Utils::projection(B, target - x * b).norm() <= upper_bound)
            // {
            //     X.push_back(static_cast<int>(x));
            //     x--;
            // }

            std::vector<Eigen::VectorXd> V;
            for (const int &x : X)
            {
                Eigen::VectorXd res = x * b + Algorithms::CVP::branch_and_bound(B, target - x * b);
                V.push_back(res);
            }
            return Utils::closest_vector(V, target);
        }

        // Solves CVP using a branch and bound algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd branch_and_bound(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            gram_schmidt_bb = Algorithms::gram_schmidt(matrix, false);

            return branch_and_bound_recursive(matrix, target);
        }
    }
    // Computes Gram Schmidt orthogonalization
    // @return Eigen::MatrixXd
    // @param matrix input matrix
    // @param normalize indicates whether to normalize output vectors
    // @param delete_zero_rows indicates whether to delete zero rows
    Eigen::MatrixXd gram_schmidt(const Eigen::MatrixXd &matrix, bool delete_zero_rows)
    {
        std::vector<Eigen::VectorXd> basis;

        for (const auto &vec : matrix.colwise())
        {
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());

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
                projections += (inner1 / inner2) * basis_vector;
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