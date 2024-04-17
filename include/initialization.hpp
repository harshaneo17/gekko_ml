#ifndef INIT_
#define INIT_

#include "tensor_load.hpp"

class Glorot{
    public:
        Tensor initialize(double n_rows,double n_cols){
            std::tuple <double, double> size;
            double a = 2 / (n_rows + n_cols);
            Tensor scale = xt::sqrt(a);
            size = std::make_tuple(n_rows,n_cols);
            Tensor param = xt::random::randn<double>({scale,size});
            return param;
        }
};

// class He{
//     Tensor initialize(double n_rows, double n_cols){
//         std::tuple <double, double> size;
//         size = std::make_tuple(n_rows,n_cols);
//         Tensor low = xt::sqrt(6/n_rows);
//         Tensor high = xt::sqrt(6/n_cols);
//         return xt::random::rand<double>({low,high,size});
//     }
// };

// class LSUV{
//     public:
//         double input_stddev;
//         LSUV(double scale){
//             input_stddev = scale;
//         }

//         Tensor initialize(double n_rows,double n_cols){
//             std::tuple <double, double> size;
//             std::tuple <Tensor, Tensor, Tensor> SVD;
//             size = std::make_tuple(n_rows,n_cols);
//             Tensor rand_t = xt::random::rand<double>({size});
//             SVD = xt::linalg::svd(rand_t,full_matrices=false);
//             Tensor weights = std::get<0>(SVD).shape == (n_rows,n_cols) ?  std::get<0>(SVD) : std::get<2>(SVD);
//             Tensor input_values = xt::random::randn<double>({input_stddev,size});
//             Tensor output_values = input_values * weights;
//             Tensor std_dev = xt::stddev(output_values);
//             return weights / std_dev;
//         }

// };

#endif