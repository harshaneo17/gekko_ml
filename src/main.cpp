#include <iostream>
#include "tensor_load.hpp"
#include "layers.hpp"
#include "neuralnetwork.hpp"

int main(int argc, char* argv[])
{
    Tensor arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    Tensor arr2
      {{5.0, 6.0, 7.0}};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;
    
    Linear linr(3,3);
    // linr.forward(arr1);
    // linr.backward(arr1);
    std::vector<Linear> layers{linr,linr};
    NeuralNet nn(layers);
    nn.forward(arr2);

    auto paramsAndGrads = nn.params_and_grads();


    return 0;
}