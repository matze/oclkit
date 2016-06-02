## oclkit, plain and stupid OpenCL helper

_oclkit_ is a small set of C functions, to avoid writing the same OpenCL boiler
plate over and over again, yet keeping full control over executing OpenCL. It
also contains a set of binaries to check and test for the installed OpenCL
runtime.


### API

The API should be pretty self-explanatory by examining [ocl.h](https://github.com/matze/oclkit/blob/master/src/ocl.h).

### Binaries

Run `make` in the top-level directory and change into `build/examples`. Most
binaries are compiled with command-line flags to chose the OpenCL platform and
device type.


#### check-infrastructure-times

Measures the time for typical boilerplate operations such as `clCreateContext`,
`clBuildProgram` etc.

    $ ./check-infrastructure-times

    Create context: 0.150983 s
    Build program : 0.002060 s
    Create kernel : 0.000004 s
    Create buffer : 0.000003 s
    Cleanup       : 0.061502 s


#### check-launch-latencies

Runs a dummy kernel and measures the OpenCL profiling times and wall clock time
for submission and execution.

    $ ./check-launch-latencies

    GeForce GTX TITAN Black
      wait for submission:  2.88220 us
      wait for execution :  7.48316 us
      wall clock         : 22.12242 us

    GeForce GTX 580
      wait for submission:  2.38133 us
      wait for execution :  4.08007 us
      wall clock         : 16.33144 us


#### check-queue-impact

Uses a single blocking, an out-of-order, two or three queues to write data,
execute a kernel and read back data. The total


#### test-profile-timer

Outputs the queue profiling timer resolution for each device.

    $ ./test-profile-timer

    GeForce GTX TITAN Black       : 1000 ns
    GeForce GTX 580               : 1000 ns


#### test-double-flags

Outputs the double extension flags for each device.

    $ ./test-double-flags


    GeForce GTX TITAN Black
      cl_khr_fp64 = 1
      cl_amd_fp64 = 0

    GeForce GTX 580
      cl_khr_fp64 = 1
      cl_amd_fp64 = 0


#### dump-opencl-binary

Outputs the compiled binary (which might be PTX assembly for NVIDIA GPUs) for
each device by appending a counter, i.e.

  $ ./dump-opencl-binary test.cl

Generates output files `test.cl.0`, `.test.cl.1` etc.


### License

The code is licensed under GPL v3.
