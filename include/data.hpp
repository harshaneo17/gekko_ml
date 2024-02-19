#ifndef DATA_HPP
#define DATA_HPP


#include "tensor_load.hpp"

// Define Iterator for Batch
class BatchIterator {
public:
    BatchIterator(int batch_size = 32, bool shuffle = true) : batch_size(batch_size), shuffle(shuffle) {}

    std::vector<BatchTuple> operator()(const Tensor& inputs, const Tensor& targets) {
        std::vector<BatchTuple> batches;
        std::vector<int> starts(inputs.size() / batch_size);
        
        starts = xt::arange(0, inputs.size(), batch_size);// Fill starts with 0, 1, 2, ..., n-1

        if (shuffle) {
            xt::random::shuffle(starts);
        }

        for (int start : starts) {
            int end = start + batch_size;
            Tensor batch_inputs(inputs.begin() + start, inputs.begin() + end);
            Tensor batch_targets(targets.begin() + start, targets.begin() + end);
            batches.push_back({batch_inputs, batch_targets});
        }

        return batches;
    }
private:
    int batch_size;
    bool shuffle;
};

#endif