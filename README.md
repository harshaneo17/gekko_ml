# Gekko engine_ml

<img src="logo_gekko.png" width="250">


A framework to develop neural networks on CPP. 
This is for educational purpose to understand how neural networks work from scratch. 
his framework uses xtensor library.


to compile this program use

    cmake -S . -B build
    make -C build/

# Getting started
This assumes you have xtensor and xtl libraries installed on your machine.

* Clone the repo. Make changes to the find package in cmakelists file.
* Ensure it points to the right directory for xtensor and xtl
* Change values of the objects in main.cpp
* Build using ```make -C build/ ```
* Run the program with ```./build/main ```


# Notes:
This model is not saved anywhere. 
It currently only uses stack of Linear Layers.
Other layers are still pending.
https://www.sharpsightlabs.com/blog/numpy-axes-explained/#numpy-axes-quick-explanation



