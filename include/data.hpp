#ifndef DATA_HPP
#define DATA_HPP

#include "tensor_load.hpp"
#include <stdexcept>

struct Batch {
    Tensor inputs;
    Tensor targets;
};


class BatchIterator {

    public:
        
        BatchIterator(int batch_size = 32, bool shuffle = true) : batch_size(batch_size), shuffle(shuffle) {}
        std::vector<Batch> initialize_batch(Tensor inputs, Tensor targets) {};
    private:
        int batch_size;
        bool shuffle;
    
};

#endif