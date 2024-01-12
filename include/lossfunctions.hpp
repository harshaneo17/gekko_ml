#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

/*A loss function measures how good predictions are
We can use this to adjust parameters of our network depending on its performance*/


#include <iostream>
#include <cmath>
#include "tensor_load.hpp"



class Lossfunctions {    
    public:
       
        virtual int loss(Tensor& predicted, Tensor& actual) {
            std::cout << "Error Not Implemented" ; 
        }

        virtual Tensor grad(Tensor& predicted, Tensor& actual) {
            std::cout << "Error Not Implemented";
        }
};

class MSE : public Lossfunctions {
    public:
        
        int loss(Tensor& predicted, Tensor& actual) override {
            return xt::sum(pow((predicted - actual), 2))[0];
            //https://stackoverflow.com/questions/58338761/how-to-convert-xtsum-expression-result-to-an-integer
        }

        Tensor grad(Tensor& predicted, Tensor& actual) override {
            return 2 * (predicted - actual);
        }

};



#endif