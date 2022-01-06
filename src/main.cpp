#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include "SFML/Graphics.hpp"
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>

int main()
{
	const int m = 5; // size of vector
	const int n = 3; // number of vectors
	const double lowest = 0;
	const double highest = 5;

	Eigen::MatrixXd B = Utils::generate_random_matrix(m, n, lowest, highest);

	std::cout << B << "\n\n";
	Eigen::MatrixXd H = Algorithms::HNF::HNF(B);
	std::cout << H << "\n\n";

	// const int m = 2; // size of vector
	// const int n = 2; // number of vectors
	// const double lowest = 0;
	// const double highest = 5;
	// const double arr_lowest = 0;
	// const double arr_highest = 5;

	// Eigen::MatrixXd B = Utils::generate_random_matrix_with_linearly_independent_rows(m, n, lowest, highest);
	// std::cout << B << "\n\n";
	// Eigen::VectorXd t = Utils::generate_random_array(n, arr_lowest, arr_highest);
	// std::cout << t << "\n\n";

	// Eigen::VectorXd res = Algorithms::CVP::branch_and_bound(B, t);
	// std::cout << res << "\n\n";

	return 0;
}
