#ifndef TRAIN_HPP
#define TRAIN_HPP

#include "tensor_load.hpp"
#include "neuralnetwork.hpp"
#include "optimizer.hpp"
#include "lossfunctions.hpp"

class Train{
    public:

        void train(NeuralNet& net,Tensor& inputs,Tensor& targets,int& num_epochs,DataIterator& BatchIterator(),Loss& loss,Optimizer& optimizer){
            for (epoch : num_epochs){
                double epoch_loss = 0.0;
                    for 
            }

        }
};