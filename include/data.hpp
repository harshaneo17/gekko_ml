#ifndef DATA_HPP
#define DATA_HPP

#include "tensor_load.hpp"
#include <stdexcept>

struct Batch {
    Tensor inputs;
    Tensor targets;
};

class DataIterator {
public:
    virtual std::vector<Batch> initialize(Tensor inputs, Tensor targets) = 0;
};

class BatchIterator : public DataIterator {

public:
    
    BatchIterator(int batch_size = 32, bool shuffle = true) : batch_size(batch_size), shuffle(shuffle) {}

    // std::vector<double> arange(double start, double stop, double step) {
    //     if (step > stop){
    //         throw std::invalid_argument( "reduce batch size: its greater than input size" );
    //     }
    //     std::vector<double> result;
    //     for (double value = start; value < stop; value += step) {
    //         result.push_back(value);
    //         value = std::min(value + step, stop);
    //     }
    //     return result;
    // }
    
    std::vector<Batch> initialize(Tensor inputs, Tensor targets) override {
        std::vector<Batch> batches;

        Tensor starts = xt::arange(0,static_cast<int>(inputs.size()),batch_size);

        if (shuffle) {
            xt::random::shuffle(starts);
        }
        
        Batch batch;
        for (auto start : starts) {
            batch.inputs = inputs;
            batch.targets = targets;
            batches.push_back(batch);
        }
        return batches;
    }

private:
    int batch_size;
    bool shuffle;
    
};

#endif