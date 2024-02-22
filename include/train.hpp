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

        
        void train(NeuralNet net,Tensor inputs,Tensor targets,int num_epochs,BatchIterator batchit,MSE loss,Optimizer optimizer){
            for (size_t epoch = 0; epoch < num_epochs; epoch++){
                double epoch_loss = 0.0;
                std::vector<Batch> batches = batchit.initialize(inputs, targets);
                    for (size_t i = 0; i < batches.size(); i++) {
                        Tensor predicted = net.forward(batches[i].inputs);
                        std::cout << "predicted" << predicted << std::endl;
                        std::cout << "targets" << batches[i].targets << std::endl;
                        epoch_loss += loss.loss(predicted, batches[i].targets);
                        Tensor grad = loss.grad(predicted, batches[i].targets);
                        net.backward(grad);
                        optimizer.step(net);
                    }
                std::cout << "Epoch: " << epoch + 1 << ", Loss: " << epoch_loss << std::endl;
                }

        }
};

#endif