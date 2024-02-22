#ifndef DATA_HPP
#define DATA_HPP


#include "tensor_load.hpp"

// Define Iterator for Batch
struct Batch {
    Tensor inputs;
    Tensor targets;
};


std::vector<double> arange(double start, double stop, double step) {
    std::vector<double> result;
    for (double value = start; value < stop; value += step) {
        result.push_back(value);
        value = std::min(value + step, stop);
    }
    return result;
}

// Define your data iterator interface
class DataIterator {
public:
    virtual std::vector<Batch> initialize(Tensor inputs, Tensor targets) = 0;
};

// Define your batch iterator class
class BatchIterator : public DataIterator {
private:
    int batch_size;
    bool shuffle;
    
public:
    
    BatchIterator(int batch_size = 32, bool shuffle = true) : batch_size(batch_size), shuffle(shuffle) {}
    
    std::vector<Batch> initialize(Tensor inputs, Tensor targets) override {
        std::vector<Batch> batches;
        std::vector<double> starts = arange(0,inputs.size(),batch_size);
        std::cout << starts.size() << std::endl;
        // std::vector<Batch> starts(10);
        if (shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(starts.begin(), starts.end(), g);
        }
        
        Batch batch;
        for (auto start : starts) {
            batch.inputs = inputs;
            batch.targets = targets;
            batches.push_back(batch);
        }
        return batches;
    }
};

#endif