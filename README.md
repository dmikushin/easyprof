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

## Running simple example

```
ninja easyprof_test
bilinear(int, int, RGBApixel*, RGBApixel*) (23 registers)
1 x <<<(3, 512, 1), (128, 1, 1)>>> min = 24.678ms, max = 24.678ms, avg = 24.678ms
```


