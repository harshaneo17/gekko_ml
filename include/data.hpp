#include "tensor_load.hpp"
#include <iostream>
#include <vector>
#include <tuple>
#include <iterator>
#include <algorithm>


// Define Batch as a NamedTuple of Tensor
using Batch = NamedTuple<Tensor, Tensor>;

// Define Iterator for Batch
class BatchIterator {
public:
    BatchIterator(int batch_size = 32, bool shuffle = true) : batch_size(batch_size), shuffle(shuffle) {}

    std::vector<Batch> operator()(const Tensor& inputs, const Tensor& targets) {
        std::vector<Batch> batches;
        std::vector<int> starts(inputs.size() / batch_size);
        
        auto starts = xt::arange(0, inputs.size(), batch_size);// Fill starts with 0, 1, 2, ..., n-1

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