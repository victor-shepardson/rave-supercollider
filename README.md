# RAVE for SuperCollider

Author: Victor Shepardson

SuperCollider plugin for RAVE autoencoder

### Requirements

- CMake >= 3.5
- SuperCollider source code
- libtorch distribution

tested on macOS 10.14.6

### Building

Clone the project:

    git clone https://github.com/victor-shepardson/rave-supercollider
    cd rave-supercollider
    mkdir build
    cd build

Then, use CMake to configure and build it:

    cmake .. -DSC_PATH=path/to/supercollider -DCMAKE_PREFIX_PATH="path/to/libtorch;/usr/local" -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cmake --build . --config Release --target install

You may want to manually specify the install location in the first step to point it at your
SuperCollider extensions directory: add the option `-DCMAKE_INSTALL_PREFIX=/path/to/extensions`.

It's expected that the SuperCollider repo is cloned at `../supercollider` relative to this repo. If
it's not: add the option `-DSC_PATH=/path/to/sc/source`.

### Developing

Use the command in `regenerate` to update CMakeLists.txt when you add or remove files from the
project. You don't need to run it if you only change the contents of existing files. You may need to
edit the command if you add, remove, or rename plugins, to match the new plugin paths. Run the
script with `--help` to see all available options.

### Caveats

- must use sample rate 48000 and block size 2048
- model loading interface is WIP
- not a well behaved UGen. all kinds of memory allocation on the audio thread.
- prior not yet implemented

### Acknowledgements

Antoine Caillon's RAVE autoencoder: https://github.com/caillonantoine/RAVE

Andrew Fyfe and Bogdan Teleaga's RAVE-audition JUCE plugins: https://github.com/Fyfe93/RAVE-audition