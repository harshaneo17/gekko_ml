#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor_load.hpp"
#include "layers.hpp"

class Tanh : public Layer {
public:
    Tensor forward(Tensor x) override {
        return tanh(x);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        auto y = tanh(inputs);
        return 1 - pow(y, 2);
    }

private:
    Tensor tanh(Tensor& x) {
        return xt::tanh(x);
    }
};

class Sigmoid : public Layer {
public:
    Tensor forward(Tensor x) override {
        return sigmoid(x);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        auto y = sigmoid(inputs);
        return y * (1 - y);
    }

private:
    Tensor sigmoid(Tensor& x) {
        auto denom_sigmoid = 1 + xt::exp(x);
        return 1 / denom_sigmoid;
    }
};

class Relu : public Layer {
public:
    Tensor forward(Tensor inputs) override {
        return relu(inputs);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        return xt::where(inputs > 0, 1, 0); // derivative of relu
    }

private:
    Tensor relu(Tensor& x) {
        return xt::maximum(x, 0);
    }
};

class Softmax : public Layer {
public:
    Tensor forward(Tensor inputs) override {
        return softmax(inputs);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        auto softmax_output = softmax(inputs);
        return softmax_output * (1 - softmax_output); // Approximate derivative for softmax
    }

private:
    Tensor softmax(Tensor& x) {
        auto exp_values = xt::exp(x);
        auto exp_values_sum = xt::sum(exp_values, 1);
        return exp_values / exp_values_sum;
    }
};

#endif
