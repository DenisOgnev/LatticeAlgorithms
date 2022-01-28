#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include "SFML/Graphics.hpp"
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>
#include "problems.hpp"

int main()
{
	// const int m = 500; // size of vector
	// const int n = 500; // number of vectors
	// const double lowest = 1;
	// const double highest = 15;

	// Eigen::MatrixXd B = Utils::generate_random_matrix(m, n, lowest, highest);
	// //std::cout << B.transpose() << "\n\n";
	// double start = omp_get_wtime();
	// Eigen::MatrixXd H = Algorithms::HNF::HNF(B);
	// double end = omp_get_wtime();
	// std::cout << end - start << "\n\n";
	// //std::cout << H << "\n\n";


	const int m = 75; // size of vector
	const int n = 75; // number of vectors
	const double lowest = 1;
	const double highest = 25;
	const double arr_lowest = 1;
	const double arr_highest = 25;

	Eigen::MatrixXd B = Utils::generate_random_matrix_with_linearly_independent_rows(m, n, lowest, highest);
	//std::cout << "B = \n" << B << "\n\n";
	Eigen::VectorXd t = Utils::generate_random_array(n, arr_lowest, arr_highest);
	//std::cout << "t = " << t.transpose() << "\n\n";


	double start = omp_get_wtime();
	Eigen::VectorXd res = Algorithms::CVP::branch_and_bound(B, t);
	double end = omp_get_wtime();
	std::cout << end - start << "\n";
	//std::cout << "Result of b&b = " << res.transpose() << "\n\n";

	return 0;
}
