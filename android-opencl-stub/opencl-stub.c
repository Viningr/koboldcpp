/*
 * Link-only Android OpenCL import shim.
 *
 * Android NDK's libOpenCL stub adds OPENCL_* symbol versions that Qualcomm's
 * vendor driver does not export. This unversioned shim is used only while
 * linking; at runtime libOpenCL.so resolves to the copied Qualcomm driver.
 */
#define STUB(name) __attribute__((visibility("default"))) void name(void) {}

STUB(clGetPlatformIDs)
STUB(clGetPlatformInfo)
STUB(clGetDeviceIDs)
STUB(clGetDeviceInfo)
STUB(clCreateContext)
STUB(clCreateCommandQueue)
STUB(clSetKernelArg)
STUB(clGetKernelWorkGroupInfo)
STUB(clEnqueueNDRangeKernel)
STUB(clCreateBuffer)
STUB(clReleaseMemObject)
STUB(clCreateImage)
STUB(clCreateSubBuffer)
STUB(clCreateKernel)
STUB(clReleaseProgram)
STUB(clReleaseKernel)
STUB(clFinish)
STUB(clEnqueueBarrierWithWaitList)
STUB(clWaitForEvents)
STUB(clReleaseEvent)
STUB(clEnqueueCopyBuffer)
STUB(clEnqueueMarkerWithWaitList)
STUB(clFlush)
STUB(clCreateBufferWithProperties)
STUB(clCreateProgramWithSource)
STUB(clBuildProgram)
STUB(clGetProgramBuildInfo)
STUB(clEnqueueWriteBuffer)
STUB(clEnqueueReadBuffer)
STUB(clEnqueueFillBuffer)
STUB(clCreateProgramWithBinary)
STUB(clGetProgramInfo)
