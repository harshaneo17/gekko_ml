#include <iostream>
#include "tensor_load.hpp"
#include "train.hpp"



int main(int argc, char* argv[])
{
    Tensor arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    Tensor arr2
      {{5.0, 6.0, 7.0}};

    Linear linr(3,3);
    Tanh tanh_obj;
    std::vector<Linear> layers{linr,linr,linr};
    NeuralNet nn(layers);
    Tensor inputs = arr1;
    Tensor targets = arr2;
    int num_epochs = 10;
    BatchIterator batch_it(1,true);
    MSE mse;
    SGD optim(0.01);

    Train train_obj;
    train_obj.train(nn,inputs,targets,num_epochs,batch_it,mse,optim);


    return 0;
}