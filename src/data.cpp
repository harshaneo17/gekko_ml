#include "data.hpp"

std::vector<Batch> BatchIterator::initialize_batch(Tensor inputs, Tensor targets) {
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