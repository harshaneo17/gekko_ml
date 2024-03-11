#include "train.hpp"

void Train::gui_train(float epoch,int num_epochs){
    /*this function uses screen width and multiple for loops to write screen*/
    while (epoch < num_epochs) {
        int barWidth = 70;

        std::cout << "[";
        int pos = barWidth * epoch;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int((epoch + 0.1) * 10.0) << " %\r";
        std::cout.flush();

        epoch += 0.0001;
    }
    std::cout << std::endl;
}

void Train::train(NeuralNet net,Tensor inputs,Tensor targets,int num_epochs,BatchIterator batchit,MSE mse,Optimizer optimizer){
    std::cout << "Training Job started" << std::endl;
    for (size_t epoch = 0; epoch < num_epochs; epoch++){
        double epoch_loss = 0.0;
        std::vector<Batch> batches = batchit.initialize(inputs, targets);
            for (size_t i = 0; i < batches.size(); i++) {
                Tensor predicted = net.forward(batches[i].inputs);
                epoch_loss  += mse.loss(predicted, batches[i].targets); 
                Tensor grad = mse.grad(predicted, batches[i].targets);
                net.backward(grad,batches[i].inputs);
                optimizer.step(net);
                
            }
        gui_train(epoch,num_epochs);
        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << epoch_loss << std::endl;
        if(epoch+1 == num_epochs){
            std::cout << "Training Job complete" << std::endl;
        }
    }

}
