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

std::tuple<Eigen::Matrix<mp::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<mp::cpp_rational, -1, -1>> gs(const Eigen::Matrix<mp::cpp_int, -1, -1> &mat)
{
	std::vector<Eigen::Vector<mp::cpp_rational, -1>> basis;
	std::vector<int> indexes;
	std::vector<int> deleted_indexes;

	Eigen::Matrix<mp::cpp_rational, -1, -1> T = Eigen::Matrix<mp::cpp_rational, -1, -1>::Identity(mat.rows(), mat.rows());

	int counter = 0;
	for (const Eigen::Vector<mp::cpp_rational, -1> &vec : mat.cast<mp::cpp_rational>().rowwise())
	{
		Eigen::Vector<mp::cpp_rational, -1> projections = Eigen::Vector<mp::cpp_rational, -1>::Zero(vec.size());

		for (int i = 0; i < basis.size(); i++)
		{
			mp::cpp_rational inner1 = std::inner_product(vec.data(), vec.data() + vec.size(), basis[i].data(), mp::cpp_rational(0.0));
			mp::cpp_rational inner2 = std::inner_product(basis[i].data(), basis[i].data() + basis[i].size(), basis[i].data(), mp::cpp_rational(0.0));
			mp::cpp_rational u_ij = 0;
			if (!inner1.is_zero())
			{
				u_ij = inner1 / inner2;
				projections.noalias() += (u_ij)*basis[i];
				T(counter, i) = u_ij;
			}
		}

		Eigen::Vector<mp::cpp_rational, -1> result = vec - projections;

		bool is_all_zero = result.isZero(1e-3);
		if (!is_all_zero)
		{
			indexes.push_back(counter);
		}
		else
		{
			deleted_indexes.push_back(counter);
		}
		basis.push_back(result);
		counter++;
	}

	Eigen::Matrix<mp::cpp_int, -1, -1> result(indexes.size(), mat.cols());
	for (int i = 0; i < indexes.size(); i++)
	{
		result.row(i) = mat.row(indexes[i]);
	}

	return std::make_tuple(result, deleted_indexes, indexes, T);
}

int main()
{
	Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> mat = Utils::generate_random_matrix(50, 50, 1, 10);
	Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> mat2 = Utils::generate_random_matrix_with_full_row_rank(50, 50, 1, 10);
	// mat << 2, 1, 2, 2, 2,
	// 	2, 1, 1, 2, 1,
	// 	2, 1, 1, 2, 2,
	// 	1, 2, 2, 1, 1,
	// 	2, 2, 1, 2, 2;
	double start_time = omp_get_wtime();
	Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF = Algorithms::HNF::HNF(mat);
	double end_time = omp_get_wtime();
	std::cout << end_time - start_time << "\n";

	start_time = omp_get_wtime();
	Eigen::Matrix<boost::multiprecision::cpp_int, -1, -1> HNF2 = Algorithms::HNF::HNF_full_row_rank(mat2);
	end_time = omp_get_wtime();
	std::cout << end_time - start_time << "\n";
	// std::cout << HNF.transpose() << "\n\n";

	// std::tuple<Eigen::Matrix<mp::cpp_int, -1, -1>, std::vector<int>, std::vector<int>, Eigen::Matrix<mp::cpp_rational, -1, -1>> gs_res = gs(mat);
	// Eigen::Matrix<mp::cpp_int, -1, -1> B_stroke = std::get<0>(gs_res);
	// std::vector<int> deleted_inds = std::get<1>(gs_res);
	// std::vector<int> inds = std::get<2>(gs_res);
	// Eigen::Matrix<mp::cpp_rational, -1, -1> T = std::get<3>(gs_res);

	// Eigen::Matrix<mp::cpp_int, -1, -1> HNF = Algorithms::HNF::HNF(B_stroke);

	// std::cout << mat.transpose() << "\n\n";

	// std::cout << HNF << "\n\n";

	// Eigen::Matrix<mp::cpp_int, -1, -1> res_HNF(mat.rows(), mat.cols());

	// for (int i = 0; i < inds.size(); i++)
	// {
	// 	res_HNF.row(inds[i]) = HNF.row(i);
	// }

	// Eigen::Matrix<mp::cpp_bin_float_double, -1, -1> t_res_HNF = res_HNF.cast<mp::cpp_bin_float_double>();

	// Eigen::Matrix<mp::cpp_bin_float_double, -1, -1> B_stroke_transposed = B_stroke.transpose().cast<mp::cpp_bin_float_double>();
	// auto QR = B_stroke.cast<mp::cpp_bin_float_double>().colPivHouseholderQr().transpose();

	// for (const auto &indx : deleted_inds)
	// {
	// 	Eigen::Vector<mp::cpp_bin_float_double, -1> vec = mat.row(indx).cast<mp::cpp_bin_float_double>();
	// 	Eigen::RowVector<mp::cpp_bin_float_double, -1> x = QR.solve(vec);

	// 	Eigen::Vector<mp::cpp_bin_float_double, -1> res = x * HNF.cast<mp::cpp_bin_float_double>();
	// 	for (mp::cpp_bin_float_double &elem : res)
	// 	{
	// 		elem = mp::round(elem);
	// 	}
	// 	res_HNF.row(indx) = res.cast<mp::cpp_int>();
	// }

	// std::cout << res_HNF.transpose() << "\n\n";

	// for (const auto &indx : deleted_inds)
	// {
	// 	Eigen::Vector<mp::cpp_bin_float_double, -1> res = Eigen::Vector<mp::cpp_bin_float_double, -1>::Zero(mat.cols());
	// 	for (int i = 0; i < indx; i++)
	// 	{
	// 		res += T(indx, i).convert_to<mp::cpp_bin_float_double>() * t_res_HNF.row(i);
	// 	}

	// 	t_res_HNF.row(indx) = res;
	// }

	// std::cout << t_res_HNF.transpose() << "\n\n";

	return 0;
}
