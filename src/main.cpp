#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <fstream>
#include <sstream>
#include <utils.h>
#include <omp.h>

void saxpy(int n, float a, float* x, int incx, float* y, int incy)
{
	for (int i = 0; i * incy < n && i * incx < n; i++)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void daxpy(int n, double a, double* x, int incx, double* y, int incy)
{
	for (int i = 0; i * incy < n && i * incx < n; i++)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy)
{
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		if (i * incy < n && i * incx < n)
		{
			y[i * incy] = y[i * incy] + a * x[i * incx];
		}
	}
}

void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy)
{
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		if (i * incy < n && i * incx < n)
		{
			y[i * incy] = y[i * incy] + a * x[i * incx];
		}
	}
}


void daxpy_time_test(cl_uint _array_size)
{
	//cl_int array_size = 850000000;
	cl_int array_size = _array_size;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	double start, end;

	fill_array_with_number(x_double, array_size, 1.0);
	fill_array_with_number(y_double, array_size, 1.0);

	start = omp_get_wtime();
	daxpy(array_size, a_double, x_double, incx, y_double, incy);
	end = omp_get_wtime();
	std::cout << "daxpy: " << end - start << std::endl;

	delete[] x_double;
	delete[] y_double;
}

void daxpy_time_test_omp(cl_uint _array_size)
{
	cl_int array_size = _array_size;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	double start, end;

	fill_array_with_number(x_double, array_size, 1.0);
	fill_array_with_number(y_double, array_size, 1.0);

	start = omp_get_wtime();
	daxpy_omp(array_size, a_double, x_double, incx, y_double, incy);
	end = omp_get_wtime();
	std::cout << "daxpy omp: " << end - start << std::endl;

	delete[] x_double;
	delete[] y_double;
}

void saxpy_time_test(cl_uint _array_size)
{
	cl_int array_size = _array_size;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;
	double start, end;

	fill_array_with_number(x_float, array_size, 1.0f);
	fill_array_with_number(y_float, array_size, 1.0f);

	start = omp_get_wtime();
	saxpy(array_size, a_float, x_float, incx, y_float, incy);
	end = omp_get_wtime();
	std::cout << "saxpy: " << end - start << std::endl;

	delete[] x_float;
	delete[] y_float;
}

void saxpy_time_test_omp(cl_uint _array_size)
{
	cl_int array_size = _array_size;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;
	double start, end;

	fill_array_with_number(x_float, array_size, 1.0f);
	fill_array_with_number(y_float, array_size, 1.0f);

	start = omp_get_wtime();
	saxpy_omp(array_size, a_float, x_float, incx, y_float, incy);
	end = omp_get_wtime();
	std::cout << "saxpy omp: " << end - start << std::endl;

	delete[] x_float;
	delete[] y_float;
}

void daxpy_test_with_print()
{
	cl_int array_size = 8;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	std::cout << "double arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_double, array_size);
	std::cout << "y: ";
	print_array(y_double, array_size);

	daxpy(array_size, a_double, x_double, incx, y_double, incy);

	std::cout << "daxpy result:" << std::endl;
	print_array(y_double, array_size);

	delete[] x_double;
	delete[] y_double;
}

void saxpy_test_with_print()
{
	cl_int array_size = 8;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	double a_float = 1.0f;

	get_random_float_array(x_float, array_size, 0, 10);
	get_random_float_array(y_float, array_size, 0, 10);
	std::cout << "float arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_float, array_size);
	std::cout << "y: ";
	print_array(y_float, array_size);

	saxpy(array_size, a_float, x_float, incx, y_float, incy);

	std::cout << "saxpy result:" << std::endl;
	print_array(y_float, array_size);

	delete[] x_float;
	delete[] y_float;
}

void daxpy_omp_test_with_print()
{
	cl_int array_size = 8;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	std::cout << "double arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_double, array_size);
	std::cout << "y: ";
	print_array(y_double, array_size);

	daxpy_omp(array_size, a_double, x_double, incx, y_double, incy);

	std::cout << "daxpy result omp:" << std::endl;
	print_array(y_double, array_size);

	delete[] x_double;
	delete[] y_double;
}

void saxpy_omp_test_with_print()
{
	cl_int array_size = 8;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	double a_float = 1.0f;

	get_random_float_array(x_float, array_size, 0, 10);
	get_random_float_array(y_float, array_size, 0, 10);
	std::cout << "float arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_float, array_size);
	std::cout << "y: ";
	print_array(y_float, array_size);

	saxpy_omp(array_size, a_float, x_float, incx, y_float, incy);

	std::cout << "saxpy result omp:" << std::endl;
	print_array(y_float, array_size);

	delete[] x_float;
	delete[] y_float;
}

void check_identity_daxpy_and_saxpy()
{
	cl_int array_size = 10000;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	float a_float = 1.0f;
	double start, end;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	copy_double_array_to_float(x_double, x_float, array_size);
	copy_double_array_to_float(y_double, y_float, array_size);

	daxpy(array_size, a_double, x_double, incx, y_double, incy);
	saxpy(array_size, a_float, x_float, incx, y_float, incy);

	std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << "\n\n";
}

void check_identity_daxpy_and_saxpy_omp()
{
	cl_int array_size = 10000;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	float a_float = 1.0f;
	double start, end;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	copy_double_array_to_float(x_double, x_float, array_size);
	copy_double_array_to_float(y_double, y_float, array_size);



	daxpy_omp(array_size, a_double, x_double, incx, y_double, incy);
	saxpy_omp(array_size, a_float, x_float, incx, y_float, incy);

	std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << "\n\n";
}

void get_device_info(cl_device_id device)
{
	char device_name[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME,
		128, device_name, nullptr);
	std::cout << device_name << std::endl;
}

void daxpy_time_test_ocl_gpu(cl_uint _array_size, size_t _work_size)
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[0][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_double_mem;
	cl_mem y_double_mem;

	//cl_uint array_size = 536870912;
	cl_int array_size = _array_size;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	double start, end;

	fill_array_with_number(x_double, array_size, 1);
	fill_array_with_number(y_double, array_size, 1);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = _work_size;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	y_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "daxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_double), &a_double);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_double_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_double_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	start = omp_get_wtime();
	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	end = omp_get_wtime();
	std::cout << "daxpy opencl gpu: " << end - start << std::endl;

	err_code = clEnqueueReadBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	clReleaseMemObject(x_double_mem);
	clReleaseMemObject(y_double_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_double;
	delete[] y_double;
}

void daxpy_time_test_ocl_cpu(cl_uint _array_size, size_t _work_size)
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[1][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_double_mem;
	cl_mem y_double_mem;

	//cl_uint array_size = 268435456;
	cl_int array_size = _array_size;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	double start, end;

	fill_array_with_number(x_double, array_size, 1);
	fill_array_with_number(y_double, array_size, 1);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = _work_size;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	y_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "daxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_double), &a_double);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_double_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_double_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	start = omp_get_wtime();
	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	end = omp_get_wtime();
	std::cout << "daxpy opencl cpu: " << end - start << std::endl;

	err_code = clEnqueueReadBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	clReleaseMemObject(x_double_mem);
	clReleaseMemObject(y_double_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_double;
	delete[] y_double;
}

void saxpy_time_test_ocl_gpu(cl_uint _array_size, size_t _work_size)
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[0][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_float_mem;
	cl_mem y_float_mem;

	//cl_uint array_size = 536870912;
	cl_int array_size = _array_size;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;
	float start, end;

	fill_array_with_number(x_float, array_size, 1);
	fill_array_with_number(y_float, array_size, 1);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = _work_size;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	y_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "saxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_float), &a_float);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_float_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_float_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	start = omp_get_wtime();
	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	end = omp_get_wtime();
	std::cout << "saxpy opencl gpu: " << end - start << std::endl;

	err_code = clEnqueueReadBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	clReleaseMemObject(x_float_mem);
	clReleaseMemObject(y_float_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_float;
	delete[] y_float;
}

void saxpy_time_test_ocl_cpu(cl_uint _array_size, size_t _work_size)
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[1][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_float_mem;
	cl_mem y_float_mem;

	//cl_uint array_size = 536870912;
	cl_int array_size = _array_size;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;
	float start, end;

	fill_array_with_number(x_float, array_size, 1);
	fill_array_with_number(y_float, array_size, 1);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = _work_size;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	y_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "saxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_float), &a_float);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_float_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_float_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	start = omp_get_wtime();
	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	end = omp_get_wtime();
	std::cout << "saxpy opencl gpu: " << end - start << std::endl;

	err_code = clEnqueueReadBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	clReleaseMemObject(x_float_mem);
	clReleaseMemObject(y_float_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_float;
	delete[] y_float;
}

void daxpy_ocl_gpu_test_with_print()
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[0][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_double_mem;
	cl_mem y_double_mem;

	cl_int array_size = 8;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	double start, end;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	std::cout << "double arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_double, array_size);
	std::cout << "y: ";
	print_array(y_double, array_size);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = 2;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	y_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "daxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_double), &a_double);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_double_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_double_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);

	err_code = clEnqueueReadBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	std::cout << "daxpy result ocl gpu:" << std::endl;
	print_array(y_double, array_size);

	clReleaseMemObject(x_double_mem);
	clReleaseMemObject(y_double_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_double;
	delete[] y_double;
}

void daxpy_ocl_cpu_test_with_print()
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[1][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_double_mem;
	cl_mem y_double_mem;

	cl_int array_size = 8;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	int incy = 1;
	int incx = 1;
	double a_double = 1.0;
	double start, end;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	std::cout << "double arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_double, array_size);
	std::cout << "y: ";
	print_array(y_double, array_size);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = 2;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	y_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "daxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_double), &a_double);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_double_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_double_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);

	err_code = clEnqueueReadBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	std::cout << "daxpy result ocl cpu:" << std::endl;
	print_array(y_double, array_size);

	clReleaseMemObject(x_double_mem);
	clReleaseMemObject(y_double_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_double;
	delete[] y_double;
}

void saxpy_ocl_gpu_test_with_print()
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[0][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_float_mem;
	cl_mem y_float_mem;

	cl_int array_size = 8;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;

	get_random_float_array(x_float, array_size, 0, 10);
	get_random_float_array(y_float, array_size, 0, 10);
	std::cout << "float arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_float, array_size);
	std::cout << "y: ";
	print_array(y_float, array_size);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = 2;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	y_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "saxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_float), &a_float);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_float_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_float_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);

	err_code = clEnqueueReadBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	std::cout << "saxpy result ocl gpu:" << std::endl;
	print_array(y_float, array_size);

	clReleaseMemObject(x_float_mem);
	clReleaseMemObject(y_float_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_float;
	delete[] y_float;
}

void saxpy_ocl_cpu_test_with_print()
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[1][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel;

	cl_mem x_float_mem;
	cl_mem y_float_mem;

	cl_int array_size = 8;

	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;

	get_random_float_array(x_float, array_size, 0, 10);
	get_random_float_array(y_float, array_size, 0, 10);
	std::cout << "float arrays:" << std::endl;
	std::cout << "x: ";
	print_array(x_float, array_size);
	std::cout << "y: ";
	print_array(y_float, array_size);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = 2;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	y_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	kernel = clCreateKernel(program, "saxpy", &ret);

	err_code = clSetKernelArg(kernel, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel, 1, sizeof(cl_float), &a_float);
	err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_float_mem);
	err_code = clSetKernelArg(kernel, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_float_mem);
	err_code = clSetKernelArg(kernel, 5, sizeof(cl_int), &incy);

	err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	clFinish(command_queue);

	err_code = clEnqueueReadBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);

	std::cout << "saxpy result ocl cpu:" << std::endl;
	print_array(y_float, array_size);

	clReleaseMemObject(x_float_mem);
	clReleaseMemObject(y_float_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_float;
	delete[] y_float;
}

void check_identity_daxpy_and_saxpy_ocl_gpu()
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[0][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel_daxpy, kernel_saxpy;

	cl_mem x_double_mem;
	cl_mem y_double_mem;
	cl_mem x_float_mem;
	cl_mem y_float_mem;

	cl_int array_size = 10000;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;
	double a_double = 1.0;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	copy_double_array_to_float(x_double, x_float, array_size);
	copy_double_array_to_float(y_double, y_float, array_size);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = 2;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	y_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	x_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	y_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	kernel_saxpy = clCreateKernel(program, "saxpy", &ret);
	kernel_daxpy = clCreateKernel(program, "daxpy", &ret);

	err_code = clSetKernelArg(kernel_saxpy, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel_saxpy, 1, sizeof(cl_float), &a_float);
	err_code = clSetKernelArg(kernel_saxpy, 2, sizeof(cl_mem), &x_float_mem);
	err_code = clSetKernelArg(kernel_saxpy, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel_saxpy, 4, sizeof(cl_mem), &y_float_mem);
	err_code = clSetKernelArg(kernel_saxpy, 5, sizeof(cl_int), &incy);

	err_code = clSetKernelArg(kernel_daxpy, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel_daxpy, 1, sizeof(cl_double), &a_double);
	err_code = clSetKernelArg(kernel_daxpy, 2, sizeof(cl_mem), &x_double_mem);
	err_code = clSetKernelArg(kernel_daxpy, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel_daxpy, 4, sizeof(cl_mem), &y_double_mem);
	err_code = clSetKernelArg(kernel_daxpy, 5, sizeof(cl_int), &incy);

	err_code = clEnqueueNDRangeKernel(command_queue, kernel_saxpy, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	err_code = clEnqueueNDRangeKernel(command_queue, kernel_daxpy, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);

	err_code = clEnqueueReadBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);
	err_code = clEnqueueReadBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << std::endl;

	clReleaseMemObject(x_float_mem);
	clReleaseMemObject(y_float_mem);
	clReleaseMemObject(x_double_mem);
	clReleaseMemObject(y_double_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel_saxpy);
	clReleaseKernel(kernel_daxpy);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_float;
	delete[] y_float;
}

void check_identity_daxpy_and_saxpy_ocl_cpu()
{
	cl_uint platform_count = 0;
	cl_platform_id* platforms;

	cl_device_id** devices;
	cl_uint* devices_count_per_platform;
	cl_uint device_count = 0;

	get_platforms(platforms, platform_count);
	get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);

	cl_device_id device = devices[1][0];

	//get_device_info(device);

	cl_int ret;
	int err_code;

	cl_context context;
	cl_command_queue command_queue;

	cl_program program;
	cl_kernel kernel_daxpy, kernel_saxpy;

	cl_mem x_double_mem;
	cl_mem y_double_mem;
	cl_mem x_float_mem;
	cl_mem y_float_mem;

	cl_int array_size = 10000;

	double* x_double = new double[array_size];
	double* y_double = new double[array_size];
	float* x_float = new float[array_size];
	float* y_float = new float[array_size];
	int incy = 1;
	int incx = 1;
	float a_float = 1.0f;
	double a_double = 1.0;

	get_random_double_array(x_double, array_size, 0, 10);
	get_random_double_array(y_double, array_size, 0, 10);
	copy_double_array_to_float(x_double, x_float, array_size);
	copy_double_array_to_float(y_double, y_float, array_size);

	std::ifstream f("axpy.cl");
	std::stringstream ss;
	ss << f.rdbuf();
	std::string str = ss.str();
	const char* source = str.c_str();
	size_t source_length = str.length();

	size_t group_size = array_size;
	size_t work_size = 2;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	command_queue = clCreateCommandQueue(context, device, 0, &ret);

	program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
	err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	x_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	y_float_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
	x_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	y_double_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
	err_code = clEnqueueWriteBuffer(command_queue, x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
	err_code = clEnqueueWriteBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	kernel_saxpy = clCreateKernel(program, "saxpy", &ret);
	kernel_daxpy = clCreateKernel(program, "daxpy", &ret);

	err_code = clSetKernelArg(kernel_saxpy, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel_saxpy, 1, sizeof(cl_float), &a_float);
	err_code = clSetKernelArg(kernel_saxpy, 2, sizeof(cl_mem), &x_float_mem);
	err_code = clSetKernelArg(kernel_saxpy, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel_saxpy, 4, sizeof(cl_mem), &y_float_mem);
	err_code = clSetKernelArg(kernel_saxpy, 5, sizeof(cl_int), &incy);

	err_code = clSetKernelArg(kernel_daxpy, 0, sizeof(cl_int), &array_size);
	err_code = clSetKernelArg(kernel_daxpy, 1, sizeof(cl_double), &a_double);
	err_code = clSetKernelArg(kernel_daxpy, 2, sizeof(cl_mem), &x_double_mem);
	err_code = clSetKernelArg(kernel_daxpy, 3, sizeof(cl_int), &incx);
	err_code = clSetKernelArg(kernel_daxpy, 4, sizeof(cl_mem), &y_double_mem);
	err_code = clSetKernelArg(kernel_daxpy, 5, sizeof(cl_int), &incy);

	err_code = clEnqueueNDRangeKernel(command_queue, kernel_saxpy, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);
	err_code = clEnqueueNDRangeKernel(command_queue, kernel_daxpy, 1, nullptr, &group_size, &work_size, 0, nullptr, nullptr);

	err_code = clEnqueueReadBuffer(command_queue, y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);
	err_code = clEnqueueReadBuffer(command_queue, y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);

	std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << std::endl;

	clReleaseMemObject(x_float_mem);
	clReleaseMemObject(y_float_mem);
	clReleaseMemObject(x_double_mem);
	clReleaseMemObject(y_double_mem);
	clReleaseProgram(program);
	clReleaseKernel(kernel_saxpy);
	clReleaseKernel(kernel_daxpy);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	delete[] platforms;
	for (cl_uint i = 0; i < platform_count; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;

	delete[] x_float;
	delete[] y_float;
}

int main()
{
	cl_int array_size = 134217728;

	daxpy_time_test(array_size);
	daxpy_time_test_omp(array_size);
	saxpy_time_test(array_size);
	saxpy_time_test_omp(array_size);

	daxpy_time_test_ocl_gpu(array_size, 2);
	daxpy_time_test_ocl_gpu(array_size, 4);
	daxpy_time_test_ocl_gpu(array_size, 8);
	daxpy_time_test_ocl_gpu(array_size, 16);
	daxpy_time_test_ocl_gpu(array_size, 32);
	daxpy_time_test_ocl_gpu(array_size, 64);
	daxpy_time_test_ocl_gpu(array_size, 128);
	daxpy_time_test_ocl_gpu(array_size, 256);

	daxpy_time_test_ocl_cpu(array_size, 2);
	saxpy_time_test_ocl_gpu(array_size, 2);
	saxpy_time_test_ocl_cpu(array_size, 2);

	daxpy_test_with_print();
	saxpy_test_with_print();
	daxpy_omp_test_with_print();
	saxpy_omp_test_with_print();

	daxpy_ocl_gpu_test_with_print();
	daxpy_ocl_cpu_test_with_print();
	saxpy_ocl_gpu_test_with_print();
	saxpy_ocl_cpu_test_with_print();

	check_identity_daxpy_and_saxpy();
	check_identity_daxpy_and_saxpy_omp();

	check_identity_daxpy_and_saxpy_ocl_gpu();
	check_identity_daxpy_and_saxpy_ocl_cpu();
}

//int main()
//{
//    cl_uint platform_count = 0;
//    cl_platform_id *platforms;
//
//    cl_device_id **devices;
//    cl_uint *devices_count_per_platform;
//    cl_uint device_count = 0;
//
//    get_platforms(platforms, platform_count);
//    get_devices(devices, device_count, devices_count_per_platform, platforms, platform_count);
//
//
//    for (cl_uint i = 0; i < platform_count; ++i)
//    {
//        get_devices_info(devices[i], devices_count_per_platform[i]);
//    }
//
//    cl_int ret;
//    int err_code;
//
//    cl_context *context = new cl_context[device_count];
//    cl_command_queue *command_queue = new cl_command_queue[device_count];
//
//    cl_program *program = new cl_program[device_count];
//    cl_kernel *kernel_daxpy = new cl_kernel[device_count];
//    cl_kernel *kernel_saxpy = new cl_kernel[device_count];
//
//    cl_mem x_double_mem;
//    cl_mem y_double_mem;
//    cl_mem x_float_mem;
//    cl_mem y_float_mem;
//
//    //cl_int array_size = 268435456;
//    //const cl_int array_size = 33554432;
//    //const cl_int array_size = 8;
//    //cl_int array_size = 400000000;
//    cl_int array_size = 268435456;
//
//    double *x_double = new double[array_size];
//    double *y_double = new double[array_size];
//    float *x_float = new float[array_size];
//    float *y_float = new float[array_size];
//    int incy = 1;
//    int incx = 1;
//    double a_double = 1.0;
//    float a_float = 1.0f;
//    double start, end;
//
//    start = omp_get_wtime();
//    get_random_double_array(x_double, array_size, 0, 10);
//    get_random_double_array(y_double, array_size, 0, 10);
//    copy_double_array_to_float(x_double, x_float, array_size);
//    copy_double_array_to_float(y_double, y_float, array_size);
//    end = omp_get_wtime();
//    std::cout << "creating: " << end - start << std::endl;
//
//    // print_array(x_double, array_size);
//    // print_array(y_double, array_size);
//    // print_array(x_float, array_size);
//    // print_array(y_float, array_size);
//
//    start = omp_get_wtime();
//    daxpy(array_size, a_double, x_double, incx, y_double, incy);
//    end = omp_get_wtime();
//    std::cout << "daxpy: " << end - start << std::endl;
//
//    start = omp_get_wtime();
//    saxpy(array_size, a_float, x_float, incx, y_float, incy);
//    end = omp_get_wtime();
//    std::cout << "saxpy: " << end - start << std::endl;
//
//    // std::cout << "Results" << std::endl;
//    // print_array(y_double, array_size);
//    // print_array(y_float, array_size);
//    std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << "\n\n";
//
//    // get_random_double_array(x_double, array_size, 0, 10);
//    // get_random_double_array(y_double, array_size, 0, 10);
//    // copy_double_array_to_float(x_double, x_float, array_size);
//    // copy_double_array_to_float(y_double, y_float, array_size);
//
//    //
//
//    start = omp_get_wtime();
//    daxpy_omp(array_size, a_double, x_double, incx, y_double, incy);
//    end = omp_get_wtime();
//    std::cout << "daxpy openmp: " << end - start << std::endl;
//
//    start = omp_get_wtime();
//    saxpy_omp(array_size, a_float, x_float, incx, y_float, incy);
//    end = omp_get_wtime();
//    std::cout << "saxpy openmp: " << end - start << std::endl;
//
//    // std::cout << "Results OMP" << std::endl;
//    // print_array(y_double, array_size);
//    // print_array(y_float, array_size);
//    std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << "\n\n";
//
//    // get_random_double_array(x_double, array_size, 0, 10);
//    // get_random_double_array(y_double, array_size, 0, 10);
//    // copy_double_array_to_float(x_double, x_float, array_size);
//    // copy_double_array_to_float(y_double, y_float, array_size);
//
//    //
//
//    std::ifstream f("axpy.cl");
//    std::stringstream ss;
//    ss << f.rdbuf();
//    std::string str = ss.str();
//    const char *source = str.c_str();
//    size_t source_length = str.length();
//
//    size_t group_size = array_size;
//    size_t* work_size = new size_t[device_count]{1, 2};
//
//    for (cl_uint i = 0, k = 0; i < platform_count; ++i)
//    {
//        for (cl_uint j = 0; j < devices_count_per_platform[i]; ++j, ++k)
//        {
//            context[k] = clCreateContext(nullptr, 1, &devices[i][j], nullptr, nullptr, &ret);
//            command_queue[k] = clCreateCommandQueue(context[k], devices[i][j], 0, &ret);
//
//            program[k] = clCreateProgramWithSource(context[k], 1, (const char **)&source, (const size_t *)&source_length, &ret);
//            err_code = clBuildProgram(program[k], 1, &devices[i][j], nullptr, nullptr, nullptr);
//
//            x_double_mem = clCreateBuffer(context[k], CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
//            y_double_mem = clCreateBuffer(context[k], CL_MEM_READ_WRITE, sizeof(cl_double) * array_size, nullptr, &ret);
//            x_float_mem = clCreateBuffer(context[k], CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
//            y_float_mem = clCreateBuffer(context[k], CL_MEM_READ_WRITE, sizeof(cl_float) * array_size, nullptr, &ret);
//            err_code = clEnqueueWriteBuffer(command_queue[k], x_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, x_double, 0, nullptr, nullptr);
//            err_code = clEnqueueWriteBuffer(command_queue[k], y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);
//            err_code = clEnqueueWriteBuffer(command_queue[k], x_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, x_float, 0, nullptr, nullptr);
//            err_code = clEnqueueWriteBuffer(command_queue[k], y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);
//
//            clReleaseMemObject(x_double_mem);
//            clReleaseMemObject(y_double_mem);
//            clReleaseMemObject(x_float_mem);
//            clReleaseMemObject(y_float_mem);
//
//            kernel_daxpy[k] = clCreateKernel(program[k], "daxpy", &ret);
//            kernel_saxpy[k] = clCreateKernel(program[k], "saxpy", &ret);
//
//            err_code = clSetKernelArg(kernel_daxpy[k], 0, sizeof(cl_int), &array_size);
//            err_code = clSetKernelArg(kernel_daxpy[k], 1, sizeof(cl_double), &a_double);
//            err_code = clSetKernelArg(kernel_daxpy[k], 2, sizeof(cl_mem), &x_double_mem);
//            err_code = clSetKernelArg(kernel_daxpy[k], 3, sizeof(cl_int), &incx);
//            err_code = clSetKernelArg(kernel_daxpy[k], 4, sizeof(cl_mem), &y_double_mem);
//            err_code = clSetKernelArg(kernel_daxpy[k], 5, sizeof(cl_int), &incy);
//
//            err_code = clSetKernelArg(kernel_saxpy[k], 0, sizeof(cl_int), &array_size);
//            err_code = clSetKernelArg(kernel_saxpy[k], 1, sizeof(cl_float), &a_float);
//            err_code = clSetKernelArg(kernel_saxpy[k], 2, sizeof(cl_mem), &x_float_mem);
//            err_code = clSetKernelArg(kernel_saxpy[k], 3, sizeof(cl_int), &incx);
//            err_code = clSetKernelArg(kernel_saxpy[k], 4, sizeof(cl_mem), &y_float_mem);
//            err_code = clSetKernelArg(kernel_saxpy[k], 5, sizeof(cl_int), &incy);
//
//            start = omp_get_wtime();
//            err_code = clEnqueueNDRangeKernel(command_queue[k], kernel_daxpy[k], 1, nullptr, &group_size, &work_size[k], 0, nullptr, nullptr);
//            end = omp_get_wtime();
//            std::cout << "daxpy opencl: " << end - start << std::endl;
//            start = omp_get_wtime();
//            err_code = clEnqueueNDRangeKernel(command_queue[k], kernel_saxpy[k], 1, nullptr, &group_size, &work_size[k], 0, nullptr, nullptr);
//            end = omp_get_wtime();
//            std::cout << "saxpy opencl: " << end - start << std::endl;
//
//            err_code = clEnqueueReadBuffer(command_queue[k], y_double_mem, CL_TRUE, 0, sizeof(cl_double) * array_size, y_double, 0, nullptr, nullptr);
//            err_code = clEnqueueReadBuffer(command_queue[k], y_float_mem, CL_TRUE, 0, sizeof(cl_float) * array_size, y_float, 0, nullptr, nullptr);
//
//            // std::cout << "Results OpenCL" << std::endl;
//            // print_array(y_double, array_size);
//            // print_array(y_float, array_size);
//            std::cout << check_equality_between_float_and_double_array(y_double, y_float, array_size) << std::endl;
//
//            // get_random_double_array(x_double, array_size, 0, 10);
//            // get_random_double_array(y_double, array_size, 0, 10);
//            // copy_double_array_to_float(x_double, x_float, array_size);
//            // copy_double_array_to_float(y_double, y_float, array_size);
//
//
//            clReleaseProgram(program[k]);
//            clReleaseKernel(kernel_daxpy[k]);
//            clReleaseKernel(kernel_saxpy[k]);
//            clReleaseCommandQueue(command_queue[k]);
//            clReleaseContext(context[k]);
//
//            std::cout << "\n";
//        }
//    }
//
//    delete[] platforms;
//    for (cl_uint i = 0; i < platform_count; ++i)
//    {
//        delete[] devices[i];
//    }
//    delete[] devices;
//
//    delete[] context;
//    delete[] command_queue;
//    delete[] program;
//    delete[] kernel_daxpy;
//    delete[] kernel_saxpy;
//
//    delete[] x_double;
//    delete[] y_double;
//    delete[] x_float;
//    delete[] y_float;
//
//    return 0;
//}