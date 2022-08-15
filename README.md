# RAVE for SuperCollider

RAVE autoencoder UGens for SuperCollider.

### Requirements

- CMake >= 3.5
- SuperCollider source code
- libtorch distribution C++ from pytorch.org

scsynth plugin tested on macOS 12.4 (arm64) native and via Rosetta

supernova plugin untested

### Building

Clone the project:

    git clone https://github.com/victor-shepardson/rave-supercollider
    cd rave-supercollider
    mkdir build
    cd build

Then, use CMake to configure and build it:

    cmake .. -DSC_PATH=path/to/supercollider -DCMAKE_PREFIX_PATH="path/to/libtorch;/usr/local" -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    ln -s /path/to/rave-supercollider /path/to/sc/extensions/

Note for arm64 macs: supercollider and libtorch architecture need to match. Building and running with Rosetta is possible but native performance is much better.

`path/to/supercollider` is the checked out SuperCollider source from https://github.com/supercollider/supercollider at the same commit as the version you are running -- see the version number and `git checkout` that tag. At the time of writing you need to build SC from source for native build on arm64 mac.

`/path/to/libtorch` is the downloaded libtorch from https://pytorch.org/get-started/locally/ . At time of writing this only distributes the x86_64 version for Mac. But you can install pytorch for arm64 via conda, and then point to that; in my case it was `/Users/victor/mambaforge/pkgs/pytorch-1.12.0-py3.9_0/lib/python3.9/site-packages/torch`.

It's expected that the SuperCollider repo is cloned at `../supercollider` relative to this repo. If
it's not: add the option `-DSC_PATH=/path/to/sc/source`.

### Usage

see the `.schelp` files or `examples/test.scd`. You need a realtime RAVE model in a torchscript file, not provided here.

### Developing

Use the command in `regenerate` to update CMakeLists.txt when you add or remove files from the
project. You don't need to run it if you only change the contents of existing files. You may need to
edit the command if you add, remove, or rename plugins, to match the new plugin paths. Run the
script with `--help` to see all available options.

### Caveats

- must use sample rate 48000 and hardware block size 2048
- expensive, not a well behaved UGen. all kinds of memory allocation on the audio thread.
- model loading is done synchronously and can glitch the audio the first time each model is loaded

### Acknowledgements

Antoine Caillon's RAVE autoencoder: https://github.com/caillonantoine/RAVE

Andrew Fyfe and Bogdan Teleaga's RAVE-audition JUCE plugins: https://github.com/Fyfe93/RAVE-audition