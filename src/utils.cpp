#include "utils.h"
#include <iostream>
#include <random>

void print()
{
    std::cout << "Test" << std::endl;
}

void get_platforms(cl_platform_id *&platforms, cl_uint &platform_count)
{
    platform_count = 0;
    clGetPlatformIDs(0, nullptr, &platform_count);
    platforms = new cl_platform_id[platform_count];
    clGetPlatformIDs(platform_count, platforms, nullptr);
}

void get_devices_gpu(cl_device_id *&devices_gpu, cl_uint &device_count_gpu, cl_platform_id *&platforms, cl_uint &platform_count)
{
    device_count_gpu = 0;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count_gpu);
    devices_gpu = new cl_device_id[device_count_gpu];
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, device_count_gpu, devices_gpu, nullptr);
}

void get_devices_cpu(cl_device_id *&devices_cpu, cl_uint &device_count_cpu, cl_platform_id *&platforms, cl_uint &platform_count)
{
    device_count_cpu = 0;
    clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_CPU, 0, nullptr, &device_count_cpu);
    devices_cpu = new cl_device_id[device_count_cpu];
    clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_CPU, device_count_cpu, devices_cpu, nullptr);
}

void get_devices_info(cl_device_id *&devices, cl_uint &device_count)
{
    for (cl_uint i = 0; i < device_count; ++i)
    {
        char device_name[128];
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
                        128, device_name, nullptr);
        std::cout << device_name << std::endl;
    }
}

void get_platforms_info(cl_platform_id *&platforms, cl_uint &platform_count)
{
    for (cl_uint i = 0; i < platform_count; ++i)
    {
        char platform_name[128];
        char platform_version[128];
        char platform_vendor[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                          128, platform_name, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,
                          128, platform_version, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                          128, platform_vendor, nullptr);
        std::cout << platform_name << std::endl;
        std::cout << platform_version << std::endl;
        std::cout << platform_vendor << std::endl;
    }
}

void get_devices(cl_device_id **&devices, cl_uint &device_count, cl_uint *&device_count_per_platform, cl_platform_id *&platforms, cl_uint &platform_count)
{
    device_count = 0;

    device_count_per_platform = new cl_uint[platform_count];
    devices = new cl_device_id *[platform_count];
    for (cl_uint i = 0; i < platform_count; ++i)
    {
        device_count_per_platform[i] = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count_per_platform[i]);
        device_count += device_count_per_platform[i];
        devices[i] = new cl_device_id[device_count_per_platform[i]];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count_per_platform[i], devices[i], nullptr);
    }
}

void get_random_double_array(double* array, int size, double lower_bound, double upper_bound)
{
    std::mt19937 random(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        array[i] = std::round(10. * dist(random)) / 10.;
        //array[i] = dist(random);
    }
}

void get_random_float_array(float* array, int size, float lower_bound, float upper_bound)
{
    std::mt19937 random(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

    for (int i = 0; i < size; i++)
    {
        array[i] = std::round(10.f * dist(random)) / 10.f;
    }
}

void fill_array_with_number(double* array, int size, double number)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        array[i] = number;
    }
}

void fill_array_with_number(float* array, int size, float number)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        array[i] = number;
    }
}

void print_array(double* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

void print_array(float* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

void copy_double_array_to_float(double* double_array, float* float_array, int array_size)
{
#pragma omp parallel for
    for (int i = 0; i < array_size; i++)
    {
        float_array[i] = static_cast<float>(double_array[i]);
    }
}

bool check_equality_between_float_and_double_array(double* double_array, float* float_array, int array_size)
{
    double epsilon = 5e-3;
    bool flag = true;
#pragma omp parallel for
    for (int i = 0; i < array_size; ++i)
    {
        if (fabs(double_array[i] - float_array[i]) > epsilon)
        {
            flag = false;
        }
    }
    return flag;
}