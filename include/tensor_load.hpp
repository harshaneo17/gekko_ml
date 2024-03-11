#ifndef TENSOR_LOAD
#define TENSOR_LOAD

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xbuilder.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>
#include <iterator>
#include <algorithm>

typedef xt::xarray<double> Tensor;
typedef std::tuple<Tensor, Tensor> TensorTuple;



#endif