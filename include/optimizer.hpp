#include "neuralnetwork.hpp"

class Optimizer {
    public:
        virtual void step(NeuralNet& net){
            std::cout << "Not implemented"
        }
};

class SGD : Optimizer {
    public:
        SGD(double lr) {
            double learning_rate;
            learning_rate = lr;
        }

        void step(NeuralNet& net) override {

        }
        
};