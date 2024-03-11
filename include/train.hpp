#ifndef TRAIN_HPP
#define TRAIN_HPP

#include "tensor_load.hpp"
#include "neuralnetwork.hpp"
#include "optimizer.hpp"
#include "lossfunctions.hpp"
#include "data.hpp"
#include "activations.hpp"



class Train{

    public:
        void gui_train(float epoch,int num_epochs){};
        void train(NeuralNet net,Tensor inputs,Tensor targets,int num_epochs,BatchIterator batchit,MSE mse,Optimizer optimizer){};
};

#endif