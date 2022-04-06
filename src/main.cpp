#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>

int main()
{
	Eigen::Matrix<int, 4, 4> mat1;
	Eigen::Matrix<double, 4, 4> mat2;
	mat1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
	mat2 << 1.1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
	Eigen::Matrix<double, 4, 4> mat3;
	std::cout << mat1 << "\n";
	std::cout << mat2 << "\n";
	std::cout << mat3 << "\n";

	return 0;
}
