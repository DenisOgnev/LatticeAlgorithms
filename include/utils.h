#ifndef __UTILS_H__
#define __UTILS_H__

#include <CL/cl.h>

void print();

void get_platforms(cl_platform_id *&platforms, cl_uint &platform_count);
void get_devices_gpu(cl_device_id *&devices_gpu, cl_uint &device_count_gpu, cl_platform_id *&platforms, cl_uint &platform_count);
void get_devices_cpu(cl_device_id *&devices_cpu, cl_uint &device_count_cpu, cl_platform_id *&platforms, cl_uint &platform_count);
void get_devices_info(cl_device_id *&devices, cl_uint &device_count);
void get_platforms_info(cl_platform_id *&platforms, cl_uint &platform_count);
void get_devices(cl_device_id **&devices, cl_uint &device_count, cl_uint *&device_count_per_platform, cl_platform_id *&platforms, cl_uint &platform_count);
void get_random_double_array(double *array, int size, double lower_bound, double upper_bound);
void get_random_float_array(float *array, int size, float lower_bound, float upper_bound);
void fill_array_with_number(double *array, int size, double number);
void fill_array_with_number(float *array, int size, float number);
void print_array(double *array, int size);
void print_array(float *array, int size);
void copy_double_array_to_float(double *double_array, float *float_array, int array_size);
bool check_equality_between_float_and_double_array(double *double_array, float *float_array, int array_size);
#endif