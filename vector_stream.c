#include<wb.h>

#define SEG_SIZE 256*8
#define NUM_STREAMS 4
#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //printf("i: %d; in1: %f; in2: %f\n", i, in1[i], in2[i]);

  if ( i < len ) out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv) {
  wbArg_t args;
  int inputLength;
  float * hostInput1;
  float * hostInput2;
  float * hostOutput;
  //streaming variables
  float * deviceA0, *deviceB0, *deviceC0;
  float * deviceA1, *deviceB1, *deviceC1;
  float * deviceA2, *deviceB2, *deviceC2;
  float * deviceA3, *deviceB3, *deviceC3;
  //final, built result
  float * deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *) malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is", inputLength);

  wbTime_start(GPU, "Create some streams.");
  cudaStream_t stream0, stream1, stream2, stream3;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  wbTime_stop(GPU, "Create some streams.");

  wbTime_start(GPU, "Allocate GPU memory.");
  int size = SEG_SIZE * sizeof(float);
  //streaming alloc
  wbCheck(cudaMalloc((void **) &deviceA0, size));
  wbCheck(cudaMalloc((void **) &deviceA1, size));
  wbCheck(cudaMalloc((void **) &deviceA2, size));
  wbCheck(cudaMalloc((void **) &deviceA3, size));
  wbCheck(cudaMalloc((void **) &deviceB0, size));
  wbCheck(cudaMalloc((void **) &deviceB1, size));
  wbCheck(cudaMalloc((void **) &deviceB2, size));
  wbCheck(cudaMalloc((void **) &deviceB3, size));
  wbCheck(cudaMalloc((void **) &deviceC0, size));
  wbCheck(cudaMalloc((void **) &deviceC1, size));
  wbCheck(cudaMalloc((void **) &deviceC2, size));
  wbCheck(cudaMalloc((void **) &deviceC3, size));
  //final alloc
  wbCheck(cudaMalloc((void **) &deviceOutput, size));
  wbTime_stop(GPU, "Allocate GPU memory.");

  wbTime_start(GPU, "Main stream loop");
  for (int i = 0; i < inputLength; i+=SEG_SIZE*NUM_STREAMS){
    //interleaved HostToDevice loading
    printf("hostinput1 first: %f\n", *(hostInput1 + i));
    cudaMemcpyAsync(deviceA0, hostInput1 + i             , SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(deviceB0, hostInput2 + i             , SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(deviceA1, hostInput1 + i + SEG_SIZE  , SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceB1, hostInput2 + i + SEG_SIZE  , SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceA2, hostInput1 + i + SEG_SIZE*2, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(deviceB2, hostInput2 + i + SEG_SIZE*2, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(deviceA3, hostInput1 + i + SEG_SIZE*3, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(deviceB3, hostInput2 + i + SEG_SIZE*3, SEG_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream3);

    //vecAdd on each stream
    vecAdd<<<SEG_SIZE/256,256,0,stream0>>>(deviceA0, deviceB0, deviceC0, SEG_SIZE);
    vecAdd<<<SEG_SIZE/256,256,0,stream1>>>(deviceA1, deviceB1, deviceC1, SEG_SIZE);
    vecAdd<<<SEG_SIZE/256,256,0,stream2>>>(deviceA2, deviceB2, deviceC2, SEG_SIZE);
    vecAdd<<<SEG_SIZE/256,256,0,stream3>>>(deviceA3, deviceB3, deviceC3, SEG_SIZE);

    //DeviceToHost updates
    cudaMemcpyAsync(hostOutput + i             , deviceC0, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(hostOutput + i + SEG_SIZE  , deviceC1, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(hostOutput + i + SEG_SIZE*2, deviceC2, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(hostOutput + i + SEG_SIZE*3, deviceC3, SEG_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream3);
  }
  wbTime_stop(GPU, "Main stream loop");


  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}


