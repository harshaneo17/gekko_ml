#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <lossfunctions.hpp>

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
    xt::xtensor<double, 2> a = {{3., 4.}, {5., 6.}};
    std::cout << a << std::endl;

    xt::xtensor<double, 2> b = {{1., 2.}, {3., 4.}};
    std::cout << b << std::endl;
    
    auto loss_test = mse.loss(a, b);

    std::cout << loss_test << std::endl;

    return 0;
}