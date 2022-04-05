#include <iostream>
#include <omp.h>
#include "utils.hpp"
#include "algorithms.hpp"
#include <Eigen/Dense>
#include <random>
#include <map>

std::map<std::string, int> action_mapping;

void compute_HNF();

int main()
{
    bool flag = true;

    action_mapping["1"] = 1;
    action_mapping["2"] = 2;
    action_mapping["3"] = 3;
    action_mapping["4"] = 4;


    while (flag)
    {
        std::cout << "Select an action (write only a number):\n";
        std::cout << "1. Compute HNF\n";
        std::cout << "2. Solve CVP problem\n";
        std::cout << "3. CVP visualization\n";
        std::cout << "4. Exit\n";
        std::string action;
        std::cin >> action;
        switch (action_mapping[action])
        {
        case 1:
        std::cout << "\n";
            compute_HNF();
            break;
        case 4:
            std::cout << "Exiting\n";
            flag = false;
            break;
        
        default:
            std::cout << "Wrong selection\n\n";
            break;
        }
    }

	return 0;
}

void HNF_particular_matrix()
{
    std::cout << "\n";
}
void compute_HNF()
{
    bool flag = true;


    std::cout << "Select an action (write only a number):\n";
    std::cout << "1. Compute HNF of a particular matrix\n";
    std::cout << "2. Compute HNF of a random matrix\n";
    std::cout << "3. Speed test\n";
    std::cout << "4. Menu\n";
    while (flag)
    {    
        std::string action;
        std::cin >> action;
        switch (action_mapping[action])
        {
        case 1:
            HNF_particular_matrix();
            flag = false;
            break;
        case 4:
            std::cout << "\n\n";
            flag = false;
            break;
            
        default:
            std::cout << "Wrong selection\n\n";
            break;
        }
    }
}


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


	// const int m = 75; // size of vector
	// const int n = 75; // number of vectors
	// const double lowest = 1;
	// const double highest = 25;
	// const double arr_lowest = 1;
	// const double arr_highest = 25;

	// Eigen::MatrixXd B = Utils::generate_random_matrix_with_linearly_independent_rows(m, n, lowest, highest);
	// std::cout << "B = \n" << B << "\n\n";
	// Eigen::VectorXd t = Utils::generate_random_array(n, arr_lowest, arr_highest);
	// std::cout << "t = " << t.transpose() << "\n\n";


	// double start = omp_get_wtime();
	// Eigen::VectorXd res = Algorithms::CVP::branch_and_bound(B, t);
	// double end = omp_get_wtime();
	// std::cout << end - start << "\n";
	// std::cout << "Result of b&b = " << res.transpose() << "\n\n";