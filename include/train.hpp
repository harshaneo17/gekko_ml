#ifndef TRAIN_HPP
#define TRAIN_HPP

#include "tensor_load.hpp"
#include "neuralnetwork.hpp"
#include "optimizer.hpp"
#include "lossfunctions.hpp"

class Train{
    public:

        void train(NeuralNet& net,Tensor& inputs,Tensor& targets,int& num_epochs,DataIterator& BatchIterator(),Loss& loss,Optimizer& optimizer){
            for (auto& epoch : num_epochs){
                double epoch_loss = 0.0;
                for (auto& batch : ) { //figure out an alternative for iterator
                    auto predicted = net.forward(btach.inputs);
                    auto epoch_loss += loss.loss(predicted, batch.targets);
                    auto grad = loss.grad(predicted, batch.targets); 
                    net.backward(grad);
                    optimizer.step(net); 
                }
                std::cout << "epoch number " << epoch << "epoch loss is "   << epoch_loss;
            }

        }
};