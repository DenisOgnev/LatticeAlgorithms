#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <stack>
#include <boost/multiprecision/gmp.hpp>

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
	int m = 7;
	int n = 7;
	int lowest = 1;
	int highest = 25;
	
	Eigen::MatrixXd mat = Utils::generate_random_matrix_with_full_column_rank(m, n, lowest, highest);
	Eigen::VectorXd vec = Utils::generate_random_vector(m, lowest, highest);

	// Eigen::MatrixXd mat(2, 2);
	// Eigen::VectorXd vec(2);
	// mat << 1, 0, 0, 1;
	// vec << 1.6, 1.6;
	
	double start = omp_get_wtime();
	Eigen::VectorXd greedy = Algorithms::CVP::greedy_recursive(mat, vec);
	double end = omp_get_wtime();
	std::cout << end - start << "\n\n";

	start = omp_get_wtime();
	Eigen::VectorXd greedy2 = Algorithms::CVP::greedy(mat, vec);
	end = omp_get_wtime();
	std::cout << end - start << "\n\n";
	

	start = omp_get_wtime();
	Eigen::VectorXd bb = Algorithms::CVP::branch_and_bound(mat, vec);
	end = omp_get_wtime();
	std::cout << end - start << "\n\n";

	
	start = omp_get_wtime();
	Eigen::VectorXd bb1 = Algorithms::CVP::branch_and_bound_parallel(mat, vec);
	end = omp_get_wtime();
	std::cout << end - start << "\n\n";
	
	// std::cout << "Greedy:\n" << greedy << "\n\n";
	// std::cout << "Greedy:\n" << greedy2 << "\n\n";
	// std::cout << "BB:\n" << bb << "\n\n";
	// std::cout << "BB:\n" << bb1 << "\n\n";


	boost::multiprecision::mpz_int a = 10;
	std::cout << a << "\n\n";

	// std::cout << "mat = \n" << mat << "\n" << "vec = \n" << vec << "\n\n";
	// std::cout << greedy << "\n\n" << bb << "\n\n";

	return 0;
}

// int main()
// {
// 	int m = 150;
// 	int n = 150;
// 	int lowest = 1;
// 	int highest = 10;

// 	Eigen::Matrix<mp::cpp_int, -1, -1> mat = Utils::generate_random_matrix(m, n, lowest, highest);

// 	double start = omp_get_wtime();
// 	Eigen::Matrix<mp::cpp_int, -1, -1> HNF = Algorithms::HNF::HNF(mat);
// 	double end = omp_get_wtime();

// 	std::cout << end - start << "\n\n";

// 	return 0;
// }
