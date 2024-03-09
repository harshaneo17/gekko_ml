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