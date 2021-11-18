__kernel void daxpy(const int n, double a, __global double *x, const int incx,
                    __global double *y, const int incy) {
  int i = get_global_id(0);
  if (i * incy < n && i * incx < n) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
  }
}

__kernel void saxpy(const int n, float a, __global float *x, const int incx,
                    __global float *y, const int incy) {
  int i = get_global_id(0);
  if (i * incy < n && i * incx < n) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
  }
}