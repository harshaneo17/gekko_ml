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
        virtual double loss(xt::xtensor<double>& predicted, xt::xtensor<double>& actual) {
            std::cout << "Error Not Implemented" ; 
        }

        virtual xt::xtensor<double> grad(xt::xtensor<double>& predicted, xt::xtensor<double>& actual) {
            std::cout << "Error Not Implemented";
        }
};

class MSE : public Lossfunctions {
    public:
        MSE () : {}
        double loss(xt::xtensor<double>& predicted, xt::xtensor<double>& actual) const override {
            return xt::sum(pow((predicted - actual), 2)); 
        }

        xt::xtensor<double> grad(xt::xtensor<double>& predicted, xt::xtensor<double>& actual) const override {
            return 2 * (predicted - actual);
        }

};



#endif