#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "tensor_load.hpp"
#include "layers.hpp"

class Conv2D : Layer {
private:
    std::unique_ptr<Initializer> initializer;
    std::unique_ptr<Optimizer> optimizer;
    double epsilon;
    int kernel_size;
    int n_kernels;

    double l1_regularization_param;
    double l1_sparse_regularization_param;
    double l1_regularization_threshold;
    double l2_regularization_param;
    double l2_sparse_regularization_param;
    double linf_param;
    int n_channels;
    int n_rows_in;
    int n_cols_in;
    int n_rows_out;
    int n_cols_out;

    Tensor dL_dw;
    Tensor weights;

    Tensor forward_in;
    Tensor forward_out;
    Tensor backward_in;
    Tensor backward_out;

public:
    NetworkLayer(
        double epsilon = 1e-2,
        std::unique_ptr<Initializer> initializer = std::make_unique<LSUV>(),
        int kernel_size = 3,
        double l1_param = 0.0,
        double l1_sparse_param = 0.0,
        double l1_threshold = 0.0,
        double l2_param = 0.0,
        double l2_sparse_param = 0.0,
        double linf_param = 0.0,
        int n_kernels = 5,
        std::unique_ptr<Optimizer> optimizer = std::make_unique<Adam>(1e-4)
    ) :
        epsilon(epsilon),
        kernel_size(2 * (kernel_size / 2) + 1),
        n_kernels(n_kernels),
        l1_regularization_param(l1_param),
        l1_sparse_regularization_param(l1_sparse_param),
        l1_regularization_threshold(l1_threshold),
        l2_regularization_param(l2_param),
        l2_sparse_regularization_param(l2_sparse_param),
        linf_param(linf_param),
        initializer(std::move(initializer)),
        optimizer(std::move(optimizer)),
        n_channels(0),
        n_rows_in(0),
        n_cols_in(0),
        n_rows_out(0),
        n_cols_out(0) {
    }
    
    
};

#endif 
