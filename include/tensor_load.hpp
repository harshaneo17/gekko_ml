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

typedef xt::xtensor<double,2> Tensor;

template <typename... Types> //study a bit more on this
struct NamedTuple {
    std::tuple<Types...> values;
};



#endif