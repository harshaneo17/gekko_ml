#ifndef DATA_HPP
#define DATA_HPP


#include "tensor_load.hpp"

// Define Iterator for Batch
struct Batch {
    Tensor inputs;
    Tensor targets;
};


std::vector<Batch> arange(double start, double stop, int step = 1) {
    std::vector<Batch> result;
    // if (step == 0) {
    //     // Avoid division by zero
    //     return result;
    // }
    for (double value = start; value < stop; value += step) {
        result.push_back(value);
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
    std::vector<Batch> batches;
    BatchIterator(int batch_size = 32, bool shuffle = true) : batch_size(batch_size), shuffle(shuffle) {}
    
    std::vector<Batch> initialize(Tensor inputs, Tensor targets) override {

        std::vector<Batch> starts = arange(0,inputs.size(),batch_size);
        if (shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(starts.begin(), starts.end(), g);
        }
        
        for (int j; j < starts.size(); j++) {
            std::cout << "This is easy"<< starts[j] << std::endl;
            int end = std::min(starts[j] + batch_size, static_cast<int>(inputs.size()));
            Batch batch;
            for (int i = starts[j]; i < end; i++) {
                batch.inputs = inputs;
                batch.targets = targets;
            }
            batches.push_back(batch);
        }
        return batches;
    }
};

#endif