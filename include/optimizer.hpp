#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor_load.hpp"
#include "neuralnetwork.hpp"


class Optimizer {
    public:
        virtual void step(NeuralNet& net){}
};

class SGD : public Optimizer {
    public:
        double learning_rate;
        SGD(double lr):learning_rate(lr) {}

        void step(NeuralNet& net) override {
            std::vector<TensorTuple> step_var = net.params_and_grads();
            for(auto &tuple : step_var){
                std::get<1>(tuple) -= learning_rate * std::get<2>(tuple);
            }
        }
        
};

class Adam : public Optimizer {
    public:
        Adam(double lr):learning_rate(lr){}

        void step(NeuralNet& net) override {
            std::vector<TensorTuple> step_var = net.params_and_grads();
            double t = 0;
            for(auto &tuple : step_var){
                t += 1;
                if (std::get<0>(tuple) == "weights"){
                    s_dw = (beta2*s_dw) + (1 - beta2)*pow(std::get<2>(tuple),std::get<2>(tuple));
                }
                else {
                    s_dw = (beta2*s_dw) + (1 - beta2)*std::get<2>(tuple);
                }
                v_dw = (beta1*v_dw) + (1 - beta1)*std::get<2>(tuple);
                
                v_dw_corr = v_dw/(1 - pow(beta1,t));
                s_dw_corr = s_dw/(1 - pow(beta2,t));
                Tensor adam_var = v_dw_corr / (xt::sqrt(s_dw_corr) + epsilon);
                std::get<1>(tuple) -= learning_rate * adam_var;
            }
        }

    private:
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 0.000000010;
        Tensor v_dw,s_dw;
        double learning_rate;
        Tensor v_dw_corr,s_dw_corr;
};

#endif