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

Eigen::Matrix<mp::cpp_rational, -1, -1> gs(const Eigen::Matrix<mp::cpp_int, -1, -1> &mat)
{
	std::vector<Eigen::Vector<mp::cpp_rational, -1>> basis;
	std::vector<int> indexes;

	int counter = 0;
	for (const Eigen::Vector<mp::cpp_rational, -1> &vec : mat.cast<mp::cpp_rational>().colwise())
	{
		Eigen::Vector<mp::cpp_rational, -1> projections = Eigen::Vector<mp::cpp_rational, -1>::Zero(vec.size());
		//#pragma omp parallel for
		for (int i = 0; i < basis.size(); i++)
		{
			mp::cpp_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis[i].data(), mp::cpp_rational(0.0));
			mp::cpp_rational inner2 = std::inner_product(basis[i].data(), basis[i].data() + basis[i].size(), basis[i].data(), mp::cpp_rational(0.0));
			//#pragma omp critical
			projections.noalias() += (inner1 / inner2) * basis[i];
		}

		Eigen::Vector<mp::cpp_rational, -1> result = vec - projections;

		bool is_all_zero = result.isZero(1e-3);
		if (!is_all_zero)
		{
			basis.push_back(result);
			indexes.push_back(counter);
		}
		counter++;
	}

	Eigen::Matrix<mp::cpp_int, -1, -1> result(mat.rows(), indexes.size());
	Eigen::Matrix<mp::cpp_rational, -1, -1> gram_schmidt(mat.rows(), basis.size());
	//#pragma omp parallel for
	for (int i = 0; i < indexes.size(); i++)
	{
		result.col(i) = mat.col(indexes[i]);
		gram_schmidt.col(i) = basis[i];
	}

	return gram_schmidt;
}


int main()
{
	Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> mat = Utils::generate_random_matrix_with_full_row_rank(50, 50, 1, 20);
	// std::cout << Utils::matrix_to_string(mat) << "\n";
	double start_time = omp_get_wtime();
	Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF = Algorithms::HNF::HNF_full_row_rank(mat);
	double end_time = omp_get_wtime();
	std::cout << end_time - start_time << "\n";
	//std::cout << mat.transpose() << "\n\n";
	// std::cout << HNF.transpose() << "\n\n";
	//print_HNF_last_col(HNF.transpose());
	return 0;
}
