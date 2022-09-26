# EasyProf: Fast and scriptable CUDA/HIP kernels profiling

The absurd of vendor-provided unusable cyclopic GUI-based profiling tools ends here.

## Building

```
mkdir build
cmake .. -G Ninja
ninja
```

## Usage

Prepend the application command line with the following:

```
LD_PRELOAD=./libeasyprof_hip.so PROFILE_TIME=1
```

## Running simple examples

```
ninja easyprof_test
bilinear(int, int, RGBApixel*, RGBApixel*) (23 registers)
1 x <<<(3, 512, 1), (128, 1, 1)>>> min = 24.678ms, max = 24.678ms, avg = 24.678ms
```

```
ninja easyprof_test2
fluidsGL Starting...
CUDA device [NVIDIA T500] has 14 Multi-Processors
addForces_k(float2*, int, int, int, int, float, float, int, unsigned long) (18 registers)
86 x <<<(1, 1, 1), (9, 9, 1)>>> min = -1.66422e+15ms, max = 1936.65ms, avg = -1.3546e+14ms
advectParticles_k(float2*, float2*, int, int, float, int, unsigned long) (30 registers)
759 x <<<(8, 8, 1), (64, 4, 1)>>> min = 1355.37ms, max = 1866.31ms, avg = 1669.99ms
advectVelocity_k(float2*, float*, float*, int, int, int, float, int, unsigned long long) (48 registers)
759 x <<<(8, 8, 1), (64, 4, 1)>>> min = 1418.23ms, max = 1970.35ms, avg = 1757.82ms
diffuseProject_k(float2*, float2*, int, int, float, float, int) (35 registers)
759 x <<<(5, 8, 1), (64, 4, 1)>>> min = 1379.15ms, max = 1914.26ms, avg = 1709.44ms
updateVelocity_k(float2*, float*, float*, int, int, int, int, unsigned long) (32 registers)
759 x <<<(8, 8, 1), (64, 4, 1)>>> min = 1365.77ms, max = 1876.7ms, avg = 1679.18ms
```

