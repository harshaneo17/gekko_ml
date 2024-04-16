#include "tensor_load.hpp"

class Glorot{
    public:
        Tensor intialize(double n_rows,double n_cols){
            std::tuple <double, double> size;
            double a = 2 / (n_rows + n_cols);
            Tensor scale = xt::sqrt(a);
            size = make_tuple(n_rows,n_cols);
            return xt::random::randn<double>({scale,size});
        }
};

