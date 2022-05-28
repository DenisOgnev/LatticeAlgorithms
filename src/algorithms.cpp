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
                throw std::runtime_error("Matrix is empty");
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
                throw std::runtime_error("Matrix is empty");
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
            // Other, the "right" way that is desribed in algorithm.
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

        #ifdef GMP
        // Computes HNF of a integer matrix that is full row rank
        // @return Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>
        // @param B full row rank matrix
        Eigen::Matrix<mp::mpz_int, -1, -1> HNF_full_row_rank_GMP(const Eigen::Matrix<mp::mpz_int, -1, -1> &B)
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
                throw std::runtime_error("Matrix is empty");
            }

            Eigen::Matrix<mp::mpz_int, -1, -1> B_stroke;
            Eigen::Matrix<mp::mpq_rational, -1, -1> ortogonalized;

            std::tuple<Eigen::Matrix<mp::mpz_int, -1, -1>, Eigen::Matrix<mp::mpq_rational, -1, -1>> result_of_gs = Utils::get_linearly_independent_columns_by_gram_schmidt_GMP(B);
            
            std::tie(B_stroke, ortogonalized) = result_of_gs;

            mp::mpq_rational t_det = 1.0;
            for (const Eigen::Vector<mp::mpq_rational, -1> &vec : ortogonalized.colwise())
            {
                t_det *= vec.squaredNorm();
            }
            mp::mpz_int det = mp::sqrt(mp::numerator(t_det));

            Eigen::Matrix<mp::mpz_int, -1, -1> H_temp = Eigen::Matrix<mp::mpz_int, -1, -1>::Identity(m, m) * det;

            for (int i = 0; i < n; i++)
            {
                H_temp = Utils::add_column_GMP(H_temp, B.col(i));
            }

            Eigen::Matrix<mp::mpz_int, -1, -1> H(m, n);
            H.block(0, 0, H_temp.rows(), H_temp.cols()) = H_temp;
            if (n > m)
            {
                H.block(0, H_temp.cols(), H_temp.rows(), n - m) = Eigen::Matrix<mp::mpz_int, -1, -1>::Zero(H_temp.rows(), n - m);
            }

            return H;
        }


        // Computes HNF of an arbitrary integer matrix
        // @return Eigen::Matrix<boost::multiprecision::mpz_int, -1, -1>
        // @param B arbitrary matrix
        Eigen::Matrix<mp::mpz_int, -1, -1> HNF_GMP(const Eigen::Matrix<mp::mpz_int, -1, -1> &B)
        {
            int m = static_cast<int>(B.rows());
            int n = static_cast<int>(B.cols());

            if (m < 1 || n < 1)
            {
                throw std::invalid_argument("Matrix is not initialized");
            }
            if (B.isZero())
            {
                throw std::runtime_error("Matrix is empty");
            }

            Eigen::Matrix<mp::mpz_int, -1, -1> B_stroke;
            std::vector<int> indicies;
            std::vector<int> deleted_indicies;
            Eigen::Matrix<mp::mpq_rational, -1, -1> T;
            std::tuple<Eigen::Matrix<mp::mpz_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<mp::mpq_rational, -1, -1>> projection = Utils::get_linearly_independent_rows_by_gram_schmidt_GMP(B);
            std::tie(B_stroke, indicies, deleted_indicies, T) = projection;

            Eigen::Matrix<mp::mpz_int, -1, -1> B_double_stroke = HNF_full_row_rank_GMP(B_stroke);
            
            Eigen::Matrix<mp::mpz_int, -1, -1> HNF(B.rows(), B.cols());

            for (int i = 0; i < indicies.size(); i++)
            {
                HNF.row(indicies[i]) = B_double_stroke.row(i);
            }

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // First way: just find linear combinations of deleted rows. More accurate

            // Eigen::Matrix<mp::mpf_float_100, -1, -1> B_stroke_transposed = B_stroke.transpose().cast<mp::mpf_float_100>();
            // auto QR = B_stroke.cast<mp::mpf_float_100>().colPivHouseholderQr().transpose();

            // for (const auto &indx : deleted_indicies)
            // {
            //     Eigen::Vector<mp::mpf_float_100, -1> vec = B.row(indx).cast<mp::mpf_float_100>();
            //     Eigen::RowVector<mp::mpf_float_100, -1> x = QR.solve(vec);

            //     Eigen::Vector<mp::mpf_float_100, -1> res = x * HNF.cast<mp::mpf_float_100>();
            //     for (mp::mpf_float_100 &elem : res) 
            //     {
            //         elem = mp::round(elem);
            //     }
            //     HNF.row(indx) = res.cast<mp::mpz_int>();
            // }
            // return HNF;
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Other, the "right" way that is desribed in algorithm.
            Eigen::Matrix<mp::mpf_float_50, -1, -1> t_HNF = HNF.cast<mp::mpf_float_50>();
            for (const auto &indx : deleted_indicies)
            {
                Eigen::Vector<mp::mpf_float_50, -1> res = Eigen::Vector<mp::mpf_float_50, -1>::Zero(B.cols());
                for (int i = 0; i < indx; i++)
                {
                    res += T(indx, i).convert_to<mp::mpf_float_50>() * t_HNF.row(i);
                }
                
                t_HNF.row(indx) = res;
            }

            return t_HNF.cast<mp::mpz_int>();
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
        #endif
    }


    namespace CVP
    {
        Eigen::MatrixXd gram_schmidt_greedy;
        Eigen::MatrixXd B_greedy;
        int index_greedy;

        Eigen::MatrixXd gram_schmidt_bb;
        Eigen::MatrixXd gram_schmidt_bb_parallel;
        

        // Recursive body of greedy algorithm
        // @return Eigen::VectorXd
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd greedy_recursive_part(const Eigen::VectorXd &target)
        {
            if (index_greedy == 0)
            {
                return Eigen::VectorXd::Zero(target.rows());
            }
            index_greedy--;
            Eigen::VectorXd b = B_greedy.col(index_greedy);
            Eigen::VectorXd b_star = gram_schmidt_greedy.col(index_greedy);
            double inner1 = std::inner_product(target.data(), target.data() + target.size(), b_star.data(), 0.0);
            double inner2 = std::inner_product(b_star.data(), b_star.data() + b_star.size(), b_star.data(), 0.0);
                
            double x = inner1 / inner2;
            double c = std::round(x);

            Eigen::VectorXd t_res = c * b;

            return t_res + Algorithms::CVP::greedy_recursive_part(target - t_res);
        }


        // Solves CVP using a recursive greedy algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd greedy_recursive(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            B_greedy = matrix;
            gram_schmidt_greedy = Algorithms::gram_schmidt(matrix, false);
            index_greedy = static_cast<int>(matrix.cols());

            return greedy_recursive_part(target);
        }


        // Solves CVP using a non recursive greedy algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd greedy(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            Eigen::MatrixXd gram_schmidt = Algorithms::gram_schmidt(matrix, false);

            Eigen::VectorXd result = Eigen::VectorXd::Zero(target.rows());

            Eigen::VectorXd t_target = target;

            int n = static_cast<int>(matrix.cols());
            for (int i = 0; i < matrix.cols(); i++)
            {
                int index = n - i - 1;
                Eigen::VectorXd b = matrix.col(index);
                Eigen::VectorXd b_star = gram_schmidt.col(index);
                double inner1 = std::inner_product(t_target.data(), t_target.data() + t_target.size(), b_star.data(), 0.0);
                double inner2 = std::inner_product(b_star.data(), b_star.data() + b_star.size(), b_star.data(), 0.0);
                    
                double x = inner1 / inner2;
                double c = std::round(x);
                Eigen::VectorXd t_res = c * b;

                t_target -= t_res;
                result += t_res;
            }

            return result;
        }


        // Recursive body of branch and bound algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd branch_and_bound_recursive_part(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            if (matrix.cols() == 0)
            {
                return Eigen::VectorXd::Zero(target.rows());
            }
            Eigen::MatrixXd B = matrix.block(0, 0, matrix.rows(), matrix.cols() - 1);
            Eigen::VectorXd b = matrix.col(B.cols());
            Eigen::VectorXd b_star = gram_schmidt_bb.col(B.cols());

            Eigen::VectorXd v = Algorithms::CVP::greedy(matrix, target);
            
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
                if (flag1 && Utils::projection(B, target - x1 * b).norm() <= upper_bound)
                {
                    X.push_back(static_cast<int>(x1));
                    x1++;
                } 
                else
                {
                    flag1 = false;
                }
                    
                if (flag2 && Utils::projection(B, target - x2 * b).norm() <= upper_bound)
                {
                    X.push_back(static_cast<int>(x2));
                    x2--;
                } 
                else 
                {
                    flag2 = false;
                }
            }

            std::vector<Eigen::VectorXd> V;

            
            Eigen::VectorXd t_res;
            for (const int &x : X)
            {
                t_res = x * b + Algorithms::CVP::branch_and_bound_recursive_part(B, target - x * b);
                V.push_back(t_res);
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

            return branch_and_bound_recursive_part(matrix, target);
        }

        #ifdef PARALLEL_BB
        // Recursive parallel body of branch and bound algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd branch_and_bound_recursive_part_parallel(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            if (matrix.cols() == 0)
            {
                return Eigen::VectorXd::Zero(target.rows());
            }
            Eigen::MatrixXd B = matrix.block(0, 0, matrix.rows(), matrix.cols() - 1);
            Eigen::VectorXd b = matrix.col(B.cols());
            Eigen::VectorXd b_star = gram_schmidt_bb_parallel.col(B.cols());

            Eigen::VectorXd v = Algorithms::CVP::greedy(matrix, target);
            
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
                if (flag1 && Utils::projection(B, target - x1 * b).norm() <= upper_bound)
                {
                    X.push_back(static_cast<int>(x1));
                    x1++;
                } 
                else
                {
                    flag1 = false;
                }
                    
                if (flag2 && Utils::projection(B, target - x2 * b).norm() <= upper_bound)
                {
                    X.push_back(static_cast<int>(x2));
                    x2--;
                } 
                else 
                {
                    flag2 = false;
                }
            }

            std::vector<Eigen::VectorXd> V;

                        
            Eigen::VectorXd result;
            Eigen::VectorXd res;
            #pragma omp parallel 
            {
                #pragma omp single nowait
                {
                    for (const int &x : X)
                    {
                        #pragma omp task
                        {
                            res = x * b + Algorithms::CVP::branch_and_bound_recursive_part_parallel(B, target - x * b);
                            #pragma omp critical
                            V.push_back(res);
                        }
                    }
                    #pragma omp taskwait
                    result = Utils::closest_vector(V, target);
                }
            }
            
            return result;
        }


        // Solves CVP using a branch and bound parallel algorithm
        // @return Eigen::VectorXd
        // @param matrix input rational lattice basis that is linearly independent
        // @param target vector for which lattice point is being searched for
        Eigen::VectorXd branch_and_bound_parallel(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &target)
        {
            gram_schmidt_bb_parallel = Algorithms::gram_schmidt(matrix, false);

            return branch_and_bound_recursive_part_parallel(matrix, target);
        }
        #endif
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

            #pragma omp parallel for
            for (int i = 0; i < basis.size(); i++)
            {
                Eigen::MatrixXd basis_vector = basis[i];
                double inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis_vector.data(), 0.0);
                double inner2 = std::inner_product(basis_vector.data(), basis_vector.data() + basis_vector.size(), basis_vector.data(), 0.0);

                #pragma omp critical
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