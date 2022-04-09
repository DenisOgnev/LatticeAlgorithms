#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

void print_HNF_last_col(const Eigen::Matrix<boost::multiprecision::cpp_int, Eigen::Dynamic, Eigen::Dynamic> &matrix)
{
	Eigen::Vector<boost::multiprecision::cpp_int, Eigen::Dynamic> column = matrix.col(matrix.cols() - 1);
	for (const auto &elem : column)
	{
		std::cout << elem << "\n";
	}
}

namespace mp = boost::multiprecision;

int main()
{
	Eigen::Matrix<boost::multiprecision::cpp_int, Eigen::Dynamic, Eigen::Dynamic> mat = Utils::generate_random_matrix_with_full_row_rank(25, 45, 1, 100);
	//mat << 1, 1, -1, 2;
	//mat << 6, 10, 6, 9, 1, 6, 5, 9, 5;
// 	mat << 7, 10,  4,  6,  2, 10, 4,  6,  6,  9,  5,  9,
//  5,  4,  1,  3,  9,  4,
//  2,  3,  5,  6,  2,  4,
//  9,  9,  6,  4,  5,  3,
//  5,  9,  1,  2,  2,  4;
	std::cout << Utils::matrix_to_string(mat) << "\n";
	double start_time = omp_get_wtime();
	Eigen::Matrix<boost::multiprecision::cpp_int, Eigen::Dynamic, Eigen::Dynamic> HNF = Algorithms::HNF::HNF_full_row_rank(mat);
	double end_time = omp_get_wtime();
	std::cout << end_time - start_time << "\n";
	std::cout << mat.transpose() << "\n\n";
	//print_HNF_last_col(HNF);

	std::cout << HNF.transpose() << "\n\n";


	// std::cout << c / d << " " << (-c - (d - 1)) / d << "\n";	
	// std::cout << mp::floor(e / f) << " " << mp::floor(-e / f) << "\n";	


	//std::cout << Utils::det_by_gram_schmidt(mat) << "\n";

	// Eigen::MatrixXd mat = Utils::generate_random_matrix_with_full_row_rank(3, 3, 1, 5);
	// // double start_time = omp_get_wtime();
	// // Utils::get_linearly_independent_columns_by_gram_schmidt(mat);
	// // double end_time = omp_get_wtime();
	// // std::cout << end_time - start_time << "\n";

	// // start_time = omp_get_wtime();	
	// // Utils::get_linearly_independent_rows_by_gram_schmidt(mat);
	// // end_time = omp_get_wtime();
	// // std::cout << end_time - start_time << "\n";

	return 0;
}
