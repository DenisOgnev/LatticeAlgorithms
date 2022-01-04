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
	const int m = 3; // size of vector
	const int n = 3; // number of vectors
	const double lowest = 0;
	const double highest = 5;

	Eigen::MatrixXd B = Utils::generate_random_matrix_with_full_row_rank(m, n, lowest, highest);

	std::cout << B << "\n\n";
	Eigen::MatrixXd H = Algorithms::HNF::HNF(B);
	std::cout << H << "\n\n";

	return 0;
}
