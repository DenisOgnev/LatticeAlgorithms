#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include "SFML/Graphics.hpp"
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>

Eigen::VectorXd projection(Eigen::MatrixXd matrix, Eigen::VectorXd vector)
{
	Eigen::VectorXd projection = Eigen::VectorXd::Zero(vector.rows());

	for (size_t i = 0; i < matrix.rows(); i++)
	{
		Eigen::VectorXd matrix_row = matrix.row(i);
		projection += (vector.dot(matrix_row) / matrix_row.dot(matrix_row)) * matrix_row;
	}
	Eigen::VectorXd result = vector - projection;

	return result;
}

Eigen::VectorXd greedy(Eigen::MatrixXd matrix, Eigen::VectorXd target)
{
	if (matrix.rows() == 0)
	{
		return Eigen::VectorXd::Zero(matrix.cols());
	}
	Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
	Eigen::MatrixXd mat = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
	Eigen::VectorXd b_star = projection(mat, b);
	double x = target.dot(b_star) / b_star.dot(b_star);
	double c = round(x);

	return c * b + greedy(mat, target - c * b);
}

double distance_between_two_vectors(Eigen::VectorXd vector1, Eigen::VectorXd vector2)
{
	return (vector1 - vector2).norm();
}

Eigen::VectorXd closest_vector(std::vector<Eigen::VectorXd> matrix, Eigen::VectorXd vector)
{
	if (matrix.size() == 0)
	{
		return vector;
	}
	Eigen::VectorXd closest = matrix[0];
	for (auto const &v : matrix)
	{
		if (distance_between_two_vectors(vector, v) <= distance_between_two_vectors(vector, closest))
		{
			closest = v;
		}
	}
	return closest;
}

Eigen::VectorXd branch_and_bound(Eigen::MatrixXd matrix, Eigen::VectorXd target)
{
	if (matrix.rows() == 0)
	{
		return Eigen::VectorXd::Zero(matrix.cols());
	}
	Eigen::VectorXd b = matrix.row(matrix.rows() - 1);
	Eigen::MatrixXd mat = matrix.block(0, 0, matrix.rows() - 1, matrix.cols());
	Eigen::VectorXd b_star = projection(mat, b);
	Eigen::VectorXd v = greedy(mat, target);

	double upper_bound = std::ceil((target - v).norm());
	double x_middle = std::floor(target.dot(b_star) / b_star.dot(b_star));
	double lower_bound = projection(mat, target - x_middle * b).norm();

	double x = x_middle;
	double temp_lower_bound = lower_bound;
	while (temp_lower_bound <= upper_bound)
	{
		x += 1;
		temp_lower_bound = projection(mat, target - x * b).norm();
	}
	double x_highest = x;

	x = x_middle;
	temp_lower_bound = lower_bound;
	while (temp_lower_bound <= upper_bound)
	{
		x -= 1;
		temp_lower_bound = projection(mat, target - x * b).norm();
	}
	double x_lowest = x + 1;

	std::vector<int> x_array;
	for (size_t i = x_lowest; i <= x_highest; i++)
	{
		x_array.push_back(i);
	}
	std::vector<Eigen::VectorXd> v_array;
	for (auto const &elem : x_array)
	{
		Eigen::VectorXd res = elem * b + branch_and_bound(mat, target - elem * b);
		v_array.push_back(res);
	}
	return closest_vector(v_array, target);
}

int main()
{
	// const int m = 3; // size of vector
	// const int n = 3; // number of vectors
	// const double lowest = 0;
	// const double highest = 5;

	// Eigen::MatrixXd B = Utils::generate_random_matrix_with_full_row_rank(m, n, lowest, highest);

	// std::cout << B << "\n\n";
	// Eigen::MatrixXd H = Algorithms::HNF::HNF(B);
	// std::cout << H << "\n\n";

	const int m = 2; // size of vector
	const int n = 2; // number of vectors
	const double lowest = 0;
	const double highest = 5;
	const double arr_lowest = 0;
	const double arr_highest = 5;

	Eigen::MatrixXd B = Utils::generate_random_matrix_with_linearly_independent_rows(m, n, lowest, highest);
	std::cout << B << "\n\n";
	Eigen::VectorXd t = Utils::generate_random_array(n, arr_lowest, arr_highest);
	std::cout << t << "\n\n";

	Eigen::VectorXd res = branch_and_bound(B, t);
	std::cout << res << "\n\n";

	return 0;
}
