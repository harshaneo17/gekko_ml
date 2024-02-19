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
        // void gui_train(){
        //     /*this function uses screen width and multiple for loops to write screen*/
        //         float progress = 0.0;
        //         while (progress < 1.0) {
        //             int barWidth = 70;
            
        //             std::cout << "[";
        //             int pos = barWidth * progress;
        //             for (int i = 0; i < barWidth; ++i) {
        //                 if (i < pos) std::cout << "=";
        //                 else if (i == pos) std::cout << ">";
        //                 else std::cout << " ";
        //             }
        //             std::cout << "] " << int(progress * 100.0) << " %\r";
        //             std::cout.flush();
            
        //             progress += 0.16; // for demonstration only
        //         }
        //         std::cout << std::endl;
        // }

        
        void train(NeuralNet& net,Tensor& inputs,Tensor& targets,int& num_epochs,BatchIterator& BatchIterator,MSE& loss,Optimizer& optimizer){
            for (size_t epoch = 0, epoch < num_epochs, epoch++){
                double epoch_loss = 0.0;
                for (auto& input : inputs ) {
                    for(auto& target : targets){
                    auto predicted = net.forward(input);
                    auto epoch_loss += loss.loss(predicted, target);
                    auto grad = loss.grad(predicted, target); 
                    net.backward(grad); 
                    optimizer.step(net);
                    }  
                }
                std::cout << "epoch number " << epoch << "epoch loss is "   << epoch_loss;
            }

        }
};

#endif