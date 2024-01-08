#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

/*A loss function measures how good predictions are
We can use this to adjust parameters of our network depending on its performance*/


#include <iostream>
#include <cmath>
#include "tensor_load.hpp"



class Lossfunctions {    
    public:
        Lossfunctions () : {}
        virtual double loss(xt::xtensor& predicted, xt::xtensor& actual) {
            std::cout << "Error Not Implemented" ; 
        }

        virtual xt::xtensor grad(xt::tensor& predicted, xt::xtensor& actual) {
            std::cout << "Error Not Implemented";
        }
};

class MSE : public Lossfunctions {
    public:
        Lossfunctions () : {}
        double loss(xt::xtensor& predicted, xt::xtensor& actual) const override {
            return xt::sum(pow((predicted - actual), 2)); 
        }

        xt::xtensor grad(xt::tensor& predicted, xt::xtensor& actual) const override {
            return 2 * (predicted - actual);
        }

};



#endif