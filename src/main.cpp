#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include "SFML/Graphics.hpp"
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>

Eigen::MatrixXd generate_random_matrix_with_full_row_rank(const int m, const int n, double lowest, double highest)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(lowest, highest);

	Eigen::MatrixXd matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
														  { return double(int(dis(gen))); });

	Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(matrix);
	auto rank = lu_decomp.rank();

	while (rank != m)
	{
		matrix = Eigen::MatrixXd::NullaryExpr(m, n, [&]()
											  { return double(int(dis(gen))); });

		lu_decomp.compute(matrix);
		rank = lu_decomp.rank();
	}

	return matrix;
}

Eigen::MatrixXd gram_schmidt(Eigen::MatrixXd matrix, bool normalize = false, bool delete_zero_rows = true)
{
	std::vector<Eigen::VectorXd> basis;
	Eigen::MatrixXd result(matrix.rows(), matrix.rows());

	for (const auto &vec : matrix.colwise())
	{
		Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());
		for (const auto &b : basis)
		{
			projections += (vec.dot(b) / b.dot(b)) * b;
		}
		Eigen::VectorXd result = vec - projections;
		if (delete_zero_rows)
		{
			bool is_all_zero = result.isZero(1e-3);
			if (!is_all_zero)
			{
				if (normalize)
				{
					basis.push_back(result / result.norm());
				}
				else
				{
					basis.push_back(result);
				}
			}
		}
		else
		{
			if (normalize)
			{
				basis.push_back(result / result.norm());
			}
			else
			{
				basis.push_back(result);
			}
		}
	}
	for (size_t i = 0; i < basis.size(); i++)
	{
		result.col(i) = basis[i];
	}
	return result;
}

Eigen::MatrixXd get_linearly_independent_columns_by_gram_schmidt(Eigen::MatrixXd matrix)
{
	std::vector<Eigen::VectorXd> basis;
	std::vector<int> indexes;
	Eigen::MatrixXd result(matrix.rows(), matrix.rows());
	for (int i = 0; i < matrix.cols(); i++)
	{
		Eigen::VectorXd vec = matrix.col(i);
		Eigen::VectorXd projections = Eigen::VectorXd::Zero(vec.size());
		for (const auto &b : basis)
		{
			projections += (vec.dot(b) / b.dot(b)) * b;
		}
		Eigen::VectorXd result = vec - projections;
		bool is_all_zero = result.isZero(1e-3);
		if (!is_all_zero)
		{
			basis.push_back(result);
			indexes.push_back(i);
		}
	}

	for (size_t i = 0; i < indexes.size(); i++)
	{
		result.col(i) = matrix.col(indexes[i]);
	}
	return result;
}

double det_by_gram_schmidt(Eigen::MatrixXd matrix)
{
	double result = 1.0;
	Eigen::MatrixXd gs = gram_schmidt(matrix);
	for (const auto &vec : gs.colwise())
	{
		result *= vec.norm();
	}
	return result;
}

std::tuple<int, int, int> gcd_extended(int a, int b)
{
	if (a == 0)
	{
		return std::make_tuple(b, 0, 1);
	}
	std::tuple<int, int, int> tuple = gcd_extended(b % a, a);
	int gcd = std::get<0>(tuple);
	int x1 = std::get<1>(tuple);
	int y1 = std::get<2>(tuple);

	int x = y1 - (b / a) * x1;
	int y = x1;

	return std::make_tuple(gcd, x, y);
}

Eigen::ArrayXd reduce(Eigen::ArrayXd vector, Eigen::MatrixXd matrix)
{
	for (size_t i = 0; i < vector.rows(); i++)
	{
		Eigen::ArrayXd matrix_column = matrix.col(i);
		while (vector(i) < 0)
		{
			vector += matrix_column;
		}
		while (vector(i) >= matrix(i, i))
		{
			vector -= matrix_column;
		}
	}
	return vector;
}

Eigen::MatrixXd add_column(Eigen::MatrixXd H, Eigen::ArrayXd b_column)
{
	if (H.cols() == 0)
	{
		return H;
	}

	double a = H(0, 0);
	Eigen::ArrayXd h = H.block(1, 0, H.rows() - 1, 1);
	Eigen::MatrixXd H_stroke = H.block(1, 1, H.rows() - 1, H.cols() - 1);
	double b = b_column(0);
	Eigen::ArrayXd b_stroke = b_column.tail(b_column.rows() - 1);
	std::tuple<int, int, int> gcd_result = gcd_extended(static_cast<int>(a), static_cast<int>(b));
	double g = static_cast<double>(std::get<0>(gcd_result));
	double x = static_cast<double>(std::get<1>(gcd_result));
	double y = static_cast<double>(std::get<2>(gcd_result));
	Eigen::MatrixXd U(2, 2);
	U << x, -b / g, y, a / g;
	Eigen::MatrixXd temp_matrix(H.rows(), 2);
	temp_matrix.col(0) = Eigen::ArrayXd(H.col(0));
	temp_matrix.col(1) = b_column;
	Eigen::MatrixXd temp_result = temp_matrix * U;

	Eigen::ArrayXd h_stroke = temp_result.block(1, 0, temp_result.rows() - 1, 1);
	Eigen::ArrayXd b_double_stroke = temp_result.block(1, 1, temp_result.rows() - 1, 1);

	b_double_stroke = reduce(b_double_stroke, H_stroke);

	Eigen::MatrixXd H_double_stroke = add_column(H_stroke, b_double_stroke);

	h_stroke = reduce(h_stroke, H_double_stroke);

	int length_of_zeros = H_double_stroke.rows();

	Eigen::MatrixXd result(H.rows(), H.cols());
	result(0, 0) = g;
	result.block(1, 0, h_stroke.rows(), 1) = h_stroke;
	result.block(0, 1, 1, length_of_zeros).setZero();
	result.block(1, 1, H_double_stroke.rows(), H_double_stroke.cols()) = H_double_stroke;

	return result;
}

int main()
{
	const int m = 3; // size of vector
	const int n = 3; // number of vectors
	const double lowest = 0;
	const double highest = 5;

	Eigen::MatrixXd B = generate_random_matrix_with_full_row_rank(m, n, lowest, highest);

	Eigen::MatrixXd B_stroke = get_linearly_independent_columns_by_gram_schmidt(B);
	std::cout << B_stroke << "\n\n";

	double det = round(det_by_gram_schmidt(B_stroke));

	Eigen::MatrixXd H = Eigen::MatrixXd::Identity(m, m) * det;

	for (size_t i = 0; i < n; i++)
	{
		H = add_column(H, B_stroke.col(i));
	}
	std::cout << H << "\n\n";

	return 0;
}
