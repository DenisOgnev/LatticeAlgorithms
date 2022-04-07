#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "algorithms.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <random>


int main()
{
	Eigen::MatrixXd mat = Utils::generate_random_matrix_with_full_row_rank(3, 3, 1, 10);
	//mat << 1, 1, -1, 2;
	mat << 6, 10, 6, 9, 1, 6, 5, 9, 5;
	std::cout << Utils::matrix_to_string(mat) << "\n";
	double start_time = omp_get_wtime();
	Eigen::MatrixXd HNF = Algorithms::HNF::HNF_full_row_rank(mat);
	double end_time = omp_get_wtime();
	std::cout << end_time - start_time << "\n";
	std::cout << mat << "\n\n";
	std::cout << HNF << "\n\n";
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
