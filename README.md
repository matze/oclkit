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


#### test-profile-timer

Outputs the queue profiling timer resolution for each device.

    $ ./test-profile-timer

    GeForce GTX TITAN Black
      wait for submission:  2.88220 us
      wait for execution :  7.48316 us
      wall clock         : 22.12242 us

    GeForce GTX 580
      wait for submission:  2.38133 us
      wait for execution :  4.08007 us
      wall clock         : 16.33144 us


### License

The code is licensed under GPL v3.
