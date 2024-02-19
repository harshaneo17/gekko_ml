#include "tensor_load.hpp"
#include "neuralnetwork.hpp"


class Optimizer {
    public:
        virtual void step(NeuralNet& net){
            std::cout << "Not implemented";
        }
};

class SGD : Optimizer {
    public:
        double learning_rate;
        SGD(double lr):learning_rate(lr) {}

        void step(NeuralNet& net) override {
            std::vector<std::tuple<Tensor,Tensor>> step_var = net.params_and_grads();
            for(auto &tuple : step_var){
                    std::get<0>(tuple) -= learning_rate * std::get<1>(tuple);
            }
        }
        
};