#include <random>

class my_matrix
{
    private:
        int* array;
    public:
        int m, n;
    
    my_matrix(int rows, int columns)
    {
        m = rows;
        n = columns;
        array = new int[m * n];

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                array[i * n + j] = 0;
            }
        }
    };

    void fill_random(int lowest, int highest)
    {
        std::random_device rd;
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(lowest, highest);

        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                array[i * n + j] = dist(gen);
            }
        }
    };
};