#ifndef TENSOR_LOAD
#define TENSOR_LOAD

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <cmath>
#include <xtensor/xbuilder.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <map>
#include <typeinfo>

typedef xt::xtensor<double,2> Tensor;
typedef std::tuple<Tensor, Tensor> BatchTuple;



#endif