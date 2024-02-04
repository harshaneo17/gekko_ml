#include <iostream>
#include "tensor_load.hpp"
#include "lossfunctions.hpp"
#include "activations.hpp"

int main(int argc, char* argv[])
{
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xarray<double> arr2
      {5.0, 6.0, 7.0};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    MSE mse;
    Tanh testtanh;
    Sigmoid testsigmoid;
    Relu testrelu;
    Softmax testsoftmax;
    
    Tensor a = {{3., 4.}, {5., 6.}};

    Tensor b = {{1., 2.}, {3., 4.}};
    
    double loss_test = mse.loss(a, b);
    std::cout << loss_test << std::endl;

    auto grad_test = mse.grad(a,b);
    std::cout << grad_test << std::endl;

    Tensor test_tanh = testsoftmax.softmax(a);
    std::cout << test_tanh << std::endl;

    return 0;
}