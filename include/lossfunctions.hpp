#ifndef LOSSFUNCTIONS_HPP
#define LOSSFUNCTIONS_HPP

/*A loss function measures how good predictions are
We can use this to adjust parameters of our network depending on its performance*/


#include <iostream>
#include <cmath>

class Lossfunctions
{   
    public:
        Lossfunctions () : {}
        float loss(int& predicted, int& actual) {
            std::cout << "Error Not Implemented" ;
            return pow((predicted - actual), 2); 
        }

        int grad(int& predicted, int& actual) {
            std::cout << "Error Not Implemented"
        }
};





#endif