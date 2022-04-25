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
	Eigen::MatrixXd mat(2, 2);
	mat << 1, 0, 0, 1;
	//mat << 1, 0, 0, 1;
	Eigen::VectorXd vec(2);
	vec << 1.6, 1;
	
	std::cout << mat << "\n\n";
	std::cout << vec << "\n\n";
	std::cout << "Answer=\n" << Algorithms::CVP::greedy(mat, vec) << "\n\n";
	std::cout << Algorithms::CVP::branch_and_bound(mat, vec) << "\n\n";


	return 0;
}


	// Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> mat = Utils::generate_random_matrix(25, 25, 1, 10);
	// Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> mat2 = Utils::generate_random_matrix_with_full_row_rank(25, 25, 1, 10);
	// // mat << 2, 1, 2, 2, 2,
	// // 	2, 1, 1, 2, 1,
	// // 	2, 1, 1, 2, 2,
	// // 	1, 2, 2, 1, 1,
	// // 	2, 2, 1, 2, 2;
	// double start_time = omp_get_wtime();
	// Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF = Algorithms::HNF::HNF(mat);
	// double end_time = omp_get_wtime();
	// std::cout << end_time - start_time << "\n";

	// start_time = omp_get_wtime();
	// Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF2 = Algorithms::HNF::HNF_full_row_rank(mat2);
	// end_time = omp_get_wtime();
	// std::cout << end_time - start_time << "\n";