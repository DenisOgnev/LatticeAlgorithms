#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

void print_HNF_last_col(const Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> &matrix)
{
	Eigen::Vector<boost::multiprecision::cpp_int, -1> column = matrix.col(matrix.cols() - 1);
	for (const auto &elem : column)
	{
		std::cout << elem << "\n";
	}
}

namespace mp = boost::multiprecision;


int main()
{
	int m = 10;
	int n = 8;
	int lowest = 1;
	int highest = 5;
	Eigen::MatrixXd mat = Utils::generate_random_matrix_with_full_column_rank(m, n, lowest, highest);
	Eigen::VectorXd vec = Utils::generate_random_vector(m, lowest, highest);
	
	double start = omp_get_wtime();
	Eigen::VectorXd greedy = Algorithms::CVP::greedy(mat, vec);
	double end = omp_get_wtime();
	std::cout << "Greedy: " << end - start << "\n";

	start = omp_get_wtime();
	Eigen::VectorXd bb = Algorithms::CVP::branch_and_bound(mat, vec);
	end = omp_get_wtime();
	std::cout << "B&b: " << end - start << "\n";

	// std::cout << "mat = \n" << mat << "\n" << "vec = \n" << vec << "\n\n";
	// std::cout << greedy << "\n\n" << bb << "\n\n";

	return 0;
}

// int main()
// {
// 	Eigen::Matrix<mp::cpp_int, -1, -1> mat = Utils::generate_random_matrix(5, 6, 1, 10);
// 	Eigen::Matrix<mp::cpp_int, -1, -1> mat2 = Utils::generate_random_matrix_with_full_row_rank(5, 5, 1, 10);
// 	// mat << 2, 1, 2, 2, 2,
// 	// 	2, 1, 1, 2, 1,
// 	// 	2, 1, 1, 2, 2,
// 	// 	1, 2, 2, 1, 1,
// 	// 	2, 2, 1, 2, 2;
// 	std::cout << mat << "\n\n";
// 	double start_time = omp_get_wtime();
// 	Eigen::Matrix<mp::cpp_int, -1, -1> HNF = Algorithms::HNF::HNF(mat);
// 	double end_time = omp_get_wtime();
// 	std::cout << end_time - start_time << "\n";

// 	start_time = omp_get_wtime();
// 	Eigen::Matrix<mp::cpp_int, -1, -1> HNF2 = Algorithms::HNF::HNF_full_row_rank(mat);
// 	end_time = omp_get_wtime();
// 	std::cout << end_time - start_time << "\n";

// 	std::cout << HNF << "\n\n" <<  HNF2 << "\n\n";
// 	return 0;
// }
