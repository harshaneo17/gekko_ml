#include <iostream>
#include "tensor_load.hpp"
#include "layers.hpp"

int main(int argc, char* argv[])
{
    Tensor arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    Tensor arr2
      {{5.0, 6.0, 7.0}};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;
    
    Tensor a = {{3., 4.}, {5., 6.}};

    Tensor b = {{1., 2.}, {3., 4.}};

    Linear linr(3,3);
    linr.forward(arr1);
    linr.backward(arr1);
    
    // Tensor output = linr.forward(a);

    std::cout << a << "|" << b << std::endl;

    //std::cout << output << std::endl;

    return 0;
}