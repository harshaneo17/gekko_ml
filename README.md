# Gekko ML

<img src="logo_gekko.png" width="250">

A from-scratch neural network framework written in C++17 on top of [xtensor](https://xtensor.readthedocs.io/), built for **educational purposes** — to understand how forward/backward propagation, optimizers, and training loops work without a black-box autodiff engine.

This README is both a getting-started guide **and** an honest, line-by-line audit of the current codebase. The framework compiles and "runs" today, but several of its core mechanics (matrix multiplication, backprop, batching, Adam) are stubbed out incorrectly, so a network trained with it currently will **not** learn correctly. The checklist below documents every problem found, why it matters, and the concrete fix — in the order you'd need to fix them to get a working MNIST classifier, and eventually a minimal LLM.

## Contents

- [Build & Run](#build--run)
- [Project Map](#project-map)
- [Problem Checklist: Core Engine (blockers)](#problem-checklist-core-engine-blockers)
- [Problem Checklist: Layers & Activations](#problem-checklist-layers--activations)
- [Problem Checklist: Loss Functions](#problem-checklist-loss-functions)
- [Problem Checklist: Optimizers](#problem-checklist-optimizers)
- [Problem Checklist: Data Pipeline](#problem-checklist-data-pipeline)
- [Problem Checklist: Persistence & Build](#problem-checklist-persistence--build)
- [Roadmap: MNIST Classifier](#roadmap-mnist-classifier)
- [Roadmap: Minimal LLM (character/token transformer)](#roadmap-minimal-llm-charactertoken-transformer)
- [Suggested Fix Order](#suggested-fix-order)

## Build & Run

Requires [xtensor](https://xtensor.readthedocs.io/en/latest/installation.html) and `xtl`.

```bash
# Ubuntu/Debian (used by this repo's CI)
sudo apt-get install xtensor-dev

cmake -S . -B build
make -C build/
./build/main
```

If you installed xtensor/xtl somewhere non-standard, point `CMakeLists.txt`'s `find_package(... PATHS ...)` at that directory (see [#22](#problem-checklist-persistence--build)).

## Project Map

| File | Purpose |
|---|---|
| `include/tensor_load.hpp` | `Tensor` typedef (`xt::xarray<double>`) and includes |
| `include/layers.hpp` | `Layer` base class, `Linear` (dense) layer |
| `include/activations.hpp` | `Tanh`, `Sigmoid`, `Relu`, `Softmax` |
| `include/initialization.hpp` | `Glorot`, `He`, `LSUV` weight-init strategies |
| `include/lossfunctions.hpp` | `MSE`, `MAE`, `MAS` |
| `include/optimizer.hpp` | `SGD`, `Adam` |
| `include/data.hpp` | `Batch`, `BatchIterator` |
| `include/neuralnetwork.hpp` | `NeuralNet` — stacks layers, runs forward/backward |
| `include/train.hpp` | Training loop + ASCII progress bar |
| `src/main.cpp` | Example: 3 `Linear` layers + `Tanh`, trained with `Adam` |

---

## Problem Checklist: Core Engine (blockers)

These break correctness for *any* network with more than one layer, so nothing else matters until they're fixed.

- [ ] **1. `Linear::forward` re-randomizes its own weights on every call.**
  `layers.hpp:41` calls `initialize()` at the top of `forward()`. Every forward pass — including every one inside training — throws away whatever the optimizer just learned and replaces `weights`/`bias` with fresh random values. The network can never converge; it's random every step.
  **Fix:** call `initialize()` once (in the constructor, or lazily on first call only), never inside `forward()`.
  ```cpp
  Linear(double input_size, double output_size)
      : input_class_size(input_size), output_class_size(output_size) {
      initialize();     // once
  }
  Tensor forward(Tensor inputs) override {
      last_input = inputs;                       // cache for backward, see #3
      return xt::linalg::dot(inputs, params.weights) + params.bias;
  }
  ```

- [ ] **2. `Linear` uses elementwise `*` instead of matrix multiplication.**
  `outputs = inputs * params.weights` (and the equivalent lines in `backward`) is xtensor's elementwise/broadcast multiply, not `Y = XW + b`. For anything but a diagonal special case this produces the wrong shape or wrong values.
  **Fix:** use `xt::linalg::dot(a, b)` (requires `#include <xtensor-blas/xlinalg.hpp>` and linking `xtensor::optimize`/BLAS, or a hand-written matmul if you want to avoid the BLAS dependency). Update `forward` and `backward` (weight-grad, bias-grad, and the returned upstream gradient) to use `dot` throughout.

- [ ] **3. `NeuralNet` doesn't cache per-layer activations, so `backward` feeds every layer the network's original input.**
  `neuralnetwork.hpp:21-32` calls `(*iter)->backward(res, inputs)` for *every* layer using the same `inputs` tensor. Backprop's chain rule requires each layer to see the input **it individually received during forward** (i.e. the previous layer's output), not the network's raw input. With 3 stacked `Linear` layers (as in `main.cpp`), layers 2 and 3 currently receive the wrong "input" and produce wrong gradients.
  **Fix:** cache activations during `forward`, then unwind them during `backward`:
  ```cpp
  Tensor forward(Tensor& inputs) {
      cache.clear();
      cache.push_back(inputs);
      for (auto& layer : layers_class) {
          inputs = layer->forward(inputs);
          cache.push_back(inputs);
      }
      return inputs;
  }
  Tensor backward(Tensor& grad) {
      Tensor res = grad;
      for (int i = (int)layers_class.size() - 1; i >= 0; --i)
          res = layers_class[i]->backward(res, cache[i]);   // cache[i] = that layer's input
      return res;
  }
  private:
      std::vector<Tensor> cache;
  ```
  This also removes the need to pass `inputs` into `NeuralNet::backward` from `train.hpp` at all.

- [ ] **4. Bias gradient is summed over the wrong axis.**
  `layers.hpp:59`: `xt::sum(grad, 1)` sums over the feature axis. The bias gradient should sum the loss gradient **over the batch axis** (axis 0), producing one value per output feature — the same shape as `bias`.
  **Fix:** `params.grad_biases = xt::sum(grad, {0});`

---

## Problem Checklist: Layers & Activations

- [ ] **5. `Softmax::backward` admits (in its own comment) it's an approximation and is mathematically wrong.**
  `activations.hpp:66`: `softmax_output * (1 - softmax_output)` is the sigmoid derivative shape, not softmax's Jacobian. Softmax's true derivative is a full Jacobian matrix, which is expensive and rarely needed directly.
  **Fix:** don't backprop through `Softmax` on its own — pair it with cross-entropy loss (see #9) and use the well-known simplification `d(loss)/d(logits) = predicted_probs - one_hot_targets`. Compute that directly in the loss's `grad()` and skip calling `Softmax::backward` in the chain (i.e. treat "softmax + cross-entropy" as one fused output layer, which is standard practice even in real frameworks).

- [ ] **6. `Softmax::forward`'s division isn't numerically stable and may not broadcast correctly.**
  `exp_values / exp_values_sum` divides by a reduced tensor without subtracting the row-wise max first, so large logits overflow `exp()`. Also confirm `xt::sum(..., 1)` keeps a broadcastable shape (use `xt::sum(x, {1}, xt::keep_dims)` if not).
  **Fix:**
  ```cpp
  Tensor softmax(Tensor& x) {
      auto row_max = xt::amax(x, {1}, xt::keep_dims);
      auto shifted = xt::exp(x - row_max);
      auto row_sum = xt::sum(shifted, {1}, xt::keep_dims);
      return shifted / row_sum;
  }
  ```

- [ ] **7. Weight-initialization strategies exist but are never used.**
  `Glorot`, `He`, `LSUV` (`initialization.hpp`) are fully separate from `Linear::initialize()`, which just does raw `xt::random::randn`. For deeper nets (needed for MNIST accuracy, and essential for anything transformer-shaped) unscaled init causes vanishing/exploding activations.
  **Fix:** let `Linear` accept an initializer strategy and call it in `initialize()`, e.g. `Linear(in, out, std::make_unique<He>())`.

- [ ] **8. `He::initialize` is declared without `public:`.**
  `initialization.hpp:22`: `class He { Tensor initialize(...) ...` — members of a `class` default to `private`, so `He{}.initialize(...)` doesn't compile from outside. `Glorot` and `LSUV` correctly mark their methods `public:`; `He` is missing it.
  **Fix:** add `public:` before `He`'s `initialize`.

- [ ] **9. No `Conv2D`/pooling layers.** Only `Linear` (fully-connected) exists. An MLP is enough to get decent MNIST accuracy (~97%+ once the bugs above are fixed), but a CNN needs `Conv2D`, `MaxPool`/`AvgPool`, and a `Flatten` layer, none of which exist yet. Needed only if you want CNN-level accuracy or want to build image-input LLM components later.

- [ ] **10. No `LayerNorm`/`BatchNorm`.** Not needed for a small MNIST MLP, but required for any transformer/LLM work (see the LLM roadmap below) and helps deeper MLPs train stably.

---

## Problem Checklist: Loss Functions

- [ ] **11. No cross-entropy loss.** `lossfunctions.hpp` only has `MSE`, `MAE`, and `MAS`. Classification tasks (MNIST, next-token prediction for an LLM) should train against categorical cross-entropy, not MSE — MSE works but converges slower and gives worse-calibrated probabilities.
  **Fix:** add a `CrossEntropy` class whose `grad()` implements the fused softmax+cross-entropy gradient from #5:
  ```cpp
  class CrossEntropy : public Lossfunctions {
  public:
      double loss(Tensor predicted, Tensor actual) override {
          auto clipped = xt::clip(predicted, 1e-12, 1.0);
          return -xt::sum(actual * xt::log(clipped))() / predicted.shape()[0];
      }
      Tensor grad(Tensor predicted, Tensor actual) override {
          return (predicted - actual) / predicted.shape()[0];   // softmax already applied upstream
      }
  };
  ```

- [ ] **12. `Lossfunctions` base class methods fall off the end without returning a value.**
  `lossfunctions.hpp:16-22`: `loss()`/`grad()` print "Error Not Implemented" but declare a non-`void` return type with no `return` statement — undefined behavior if ever called (e.g. through a base-class pointer, or via `MAE`/`MAS`, which don't override `grad()` at all and would silently hit this UB path if `.grad()` is called on them).
  **Fix:** make the base class methods `= 0` (pure virtual) so any missing override is a compile error instead of runtime UB, and implement `grad()` for `MAE`/`MAS` (or drop `MAS` as a loss — "mean accuracy score" is a metric, not a differentiable loss, and shouldn't share the loss interface).

- [ ] **13. `MAS` ("Mean Accuracy Score") is really an evaluation metric, not a loss** — it has no gradient and can't be used to train. Fine to keep as an eval utility, just don't wire it into the `Train` loop expecting gradients; rename it (e.g. `Accuracy`) so it isn't confused with a trainable loss.

---

## Problem Checklist: Optimizers

- [ ] **14. `Adam` shares one moment-estimate `Tensor` across every parameter in the network.**
  `optimizer.hpp:55`: `v_dw`/`s_dw` are single member tensors, reused across the loop over *all* layers' weights and biases (`optimizer.hpp:34-48`). Each parameter needs its **own** running first/second moment estimate — reusing one tensor means layer 2's bias update corrupts layer 1's weight statistics (and, since weights/biases differ in shape, will produce shape mismatches or silently wrong broadcasting).
  **Fix:** keep a `std::vector<Tensor>` (or a `std::map` keyed by parameter identity) of `v`/`s` state, one entry per parameter tensor, sized/initialized to zero on first use and indexed by position in `params_and_grads()`.

- [ ] **15. The weights-vs-bias branch in `Adam::step` never triggers.**
  `optimizer.hpp:36`: checks `std::get<0>(tuple) == "weights"`, but `NeuralNet::params_and_grads()` (`neuralnetwork.hpp:36`) pushes the string `"weight"` (singular). The string never matches, so every parameter — weights included — takes the `else` branch meant for biases, and second-moment ("weights"-branch) update math is dead code.
  **Fix:** either fix the string mismatch, or better, drop the type-branching entirely — the correct Adam update is the same formula for every parameter (`s = beta2*s + (1-beta2)*grad^2`); there's no legitimate reason weights and biases need different formulas.

- [ ] **16. Second-moment update computes `grad^grad`, not `grad^2`.**
  `optimizer.hpp:37`: `pow(std::get<2>(tuple), std::get<2>(tuple))` raises the gradient to the power of itself, which is not the Adam formula and isn't even well-defined for negative gradients.
  **Fix:** `xt::square(grad)` or `xt::pow(grad, 2)`.

---

## Problem Checklist: Data Pipeline

- [ ] **17. `BatchIterator` doesn't actually slice the data — every "batch" is the full dataset.**
  `data.hpp:32-37`: for each computed `start` offset it sets `batch.inputs = inputs; batch.targets = targets;` — the whole tensor, unsliced. `batch_size` is stored but never used to index into `inputs`/`targets`. Training currently repeats a full-batch gradient step `ceil(N/batch_size)` times per epoch instead of doing minibatch SGD.
  **Fix:** slice with `xt::view`:
  ```cpp
  std::vector<Batch> initialize(Tensor inputs, Tensor targets) override {
      std::vector<Batch> batches;
      int n = static_cast<int>(inputs.shape()[0]);
      std::vector<int> idx(n);
      std::iota(idx.begin(), idx.end(), 0);
      if (shuffle) std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});
      for (int start = 0; start < n; start += batch_size) {
          int end = std::min(start + batch_size, n);
          std::vector<int> chunk(idx.begin() + start, idx.begin() + end);
          Batch b;
          b.inputs  = xt::view(inputs,  xt::keep(chunk), xt::all());
          b.targets = xt::view(targets, xt::keep(chunk), xt::all());
          batches.push_back(b);
      }
      return batches;
  }
  ```

- [ ] **18. `inputs.size()` (used for the number of batch starts) counts total elements, not rows.**
  `data.hpp:26`: `xt::arange(0, inputs.size(), batch_size)` — `size()` on a 2D tensor is `rows * cols`, not `rows`. Combined with #17 this makes batch counts meaningless. Once #17 is fixed with `inputs.shape()[0]`, this is naturally fixed too.

- [ ] **19. No real dataset loader.** `tensor_load.hpp` only defines the `Tensor` typedef — there's no IDX/CSV/PNG reader, no normalization helper, no train/validation split utility. You'll need to write (or vendor) an MNIST IDX-file parser before you can load real data (see the MNIST roadmap below).

- [ ] **20. `DataIterator::initialize` materializes *all* batches into memory up front** (`std::vector<Batch>`), rather than yielding them lazily. Fine for MNIST (60k × 784 doubles ≈ 375MB, borderline — consider `float` instead of `double` for `Tensor`), but won't scale to LLM-sized token datasets. Worth switching to a generator/iterator pattern before doing LLM-scale training.

---

## Problem Checklist: Persistence & Build

- [ ] **21. No model save/load.** The existing README already flags this: *"This model weights are not saved."* There is no serialization anywhere in the codebase. You cannot stop training and resume, or save a trained MNIST/LLM model to disk and reload it for inference.
  **Fix:** add a simple binary (or JSON) serializer for `params_and_grads()` — write each tensor's shape + raw `double` buffer, and a loader that reconstructs `Linear` layers with matching shapes. `xtensor` tensors expose `.shape()` and contiguous data via `.data()`/iterators, which is enough for a straightforward flat binary format.

- [ ] **22. `CMakeLists.txt` hardcodes one developer's local macOS paths.**
  `find_package(xtl REQUIRED PATHS /Users/harshaarya17/xtl)` and the equivalent `xtensor` line won't exist on other machines. CI works around this because `apt-get install xtensor-dev` puts headers in a standard system path that `find_package` also checks by default — but any contributor without that exact `/Users/harshaarya17/...` layout, and not doing an apt install, has to hand-edit the file.
  **Fix:** drop the hardcoded `PATHS`, or make them optional/environment-driven, e.g. `find_package(xtensor REQUIRED)` plus a documented `-Dxtensor_DIR=...` override for non-standard installs.

- [ ] **23. `Tensor = xt::xarray<double>` everywhere.** Fine for the current toy example; for MNIST-scale (60k images) or LLM-scale (embedding tables, attention matrices) data, `double` doubles your memory footprint versus `float` for no real accuracy benefit. Worth switching to `xt::xarray<float>` before scaling up.

---

## Roadmap: MNIST Classifier

Once the [Core Engine blockers](#problem-checklist-core-engine-blockers) (#1–#4) are fixed, an MLP-based MNIST classifier is very achievable with this framework. Suggested path:

- [ ] Fix #1–#4 (matmul, weight persistence across forward calls, activation caching, bias-grad axis) — nothing trains correctly without these.
- [ ] Fix #14–#16 (Adam per-parameter state) or just use `SGD` initially — it's simpler to reason about while validating the fixes above.
- [ ] Fix #17–#18 (real minibatching) — MNIST needs actual minibatch SGD to train in reasonable time/memory.
- [ ] Write an IDX file loader (#19) for the [MNIST dataset files](http://yann.lecun.com/exdb/mnist/) (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.) — read the big-endian header, load pixels into a `(60000, 784)` `Tensor`, normalize to `[0,1]`, and one-hot encode the 10 labels into `(60000, 10)`.
- [ ] Add `CrossEntropy` loss (#11) and fix `Softmax` (#5–#6) — train with softmax output + cross-entropy loss, the standard setup for classification.
- [ ] Wire up `He`/`Glorot` init (#7–#8) — a 784→128→64→10 MLP needs proper init to train reliably.
- [ ] Build the network in `main.cpp`, e.g.:
  ```cpp
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<Linear>(784, 128));
  layers.push_back(std::make_unique<Relu>());
  layers.push_back(std::make_unique<Linear>(128, 64));
  layers.push_back(std::make_unique<Relu>());
  layers.push_back(std::make_unique<Linear>(64, 10));
  layers.push_back(std::make_unique<Softmax>());
  ```
- [ ] Add save/load (#21) so a trained model can be reused for inference without retraining.
- [ ] Add an accuracy metric over a held-out test split (10k MNIST test images) to actually measure whether it's working — `MAS`/`Accuracy` (#13) applied to `argmax(predicted)` vs `argmax(actual)`.
- [ ] (Optional, higher accuracy) Add `Conv2D`/`MaxPool`/`Flatten` (#9) for a small CNN instead of an MLP.

## Roadmap: Minimal LLM (character/token transformer)

This is a much bigger lift — Gekko ML today is a plain feedforward-layer framework with hand-written per-layer backward passes (no general autodiff graph), so a lot of new machinery is needed. Treat this as a project roadmap, not a "flip a switch" checklist:

- [ ] Everything in the MNIST roadmap first — it exercises the same core engine fixes (#1–#20) an LLM depends on.
- [ ] **Tokenizer** — start with a simple character-level or byte-pair-encoding tokenizer (not in scope for xtensor; plain C++ string processing) mapping text → integer token IDs.
- [ ] **Embedding layer** — a new `Layer` type: a learnable `(vocab_size, d_model)` weight matrix, `forward` does a row-gather (`xt::view`/fancy indexing) by token ID instead of a matmul; `backward` scatter-adds gradients back into the rows that were looked up.
- [ ] **Positional encoding** — either fixed sinusoidal encoding added to embeddings, or a second learnable `(max_seq_len, d_model)` embedding table (simpler to implement first).
- [ ] **LayerNorm** (#10) — new `Layer`: normalize across the feature axis per token, with learnable scale/shift, needed before/after attention and the feed-forward block (pre-norm transformer block is the modern default and easier to get stable).
- [ ] **Multi-head self-attention** — the biggest new component: `Q/K/V` linear projections, scaled dot-product attention (`softmax(QK^T / sqrt(d_k)) V`), a **causal mask** (upper-triangular `-inf` mask so token *t* can't attend to future tokens) for autoregressive generation, multiple heads concatenated and projected back to `d_model`. This needs batched matrix multiply over a 3rd (sequence) dimension, which `xt::linalg::dot` doesn't do for batches out of the box — you'll likely need a small batched-matmul helper looping over batch/head dims.
- [ ] **Feed-forward block** — two `Linear` layers with a `Relu`/GELU in between, applied per-token (this part reuses existing `Linear`/`Relu` once #1–#4 are fixed).
- [ ] **Residual connections** — `output = sublayer(x) + x`; requires each block's `backward` to also add the identity gradient, not just the sublayer's.
- [ ] **Transformer block** — compose attention + feed-forward + LayerNorm + residuals into one reusable unit, stack N of them in `NeuralNet`.
- [ ] **Output head** — a final `Linear(d_model, vocab_size)` + softmax + cross-entropy (#11) over the next-token prediction target.
- [ ] **Sequence-batched data pipeline** — replace/extend `BatchIterator` (#17, #20) to produce `(batch, seq_len)` integer token tensors with lazy loading, since token datasets don't fit in memory the way MNIST does.
- [ ] **Autoregressive sampling loop** — a generation function that repeatedly runs `forward` on the growing sequence, applies temperature/top-k sampling to the final logits, and appends the sampled token (not part of training, but required to actually use the model afterward).
- [ ] **Save/load (#21)**, gradient clipping, and a learning-rate schedule (warmup + decay) — needed for transformer training stability at any real scale.
- [ ] **Performance**: this is a single-threaded, CPU, double-precision, header-only framework with no batched/strided matmul optimizations beyond what xtensor+BLAS gives you for free. Expect this to work for a *toy* LLM (small vocab, short context, a few layers, a tiny corpus) as a learning exercise — not a compute-competitive training stack. If you outgrow that, this codebase is a good place to *understand* the pieces before moving to a GPU-backed framework.

## Suggested Fix Order

If you're doing this incrementally, this order minimizes rework:

1. Core engine: #1 → #2 → #3 → #4
2. Optimizer: #16 → #15 → #14 (or just use `SGD` until the network is verified correct, then swap in a fixed `Adam`)
3. Data: #18 → #17 → #20
4. Loss/activation: #6 → #5 → #11 → #12 → #13
5. Init: #8 → #7
6. Build/portability: #22, #23
7. Persistence: #21
8. Then follow the [MNIST roadmap](#roadmap-mnist-classifier), and only after that's working, the [LLM roadmap](#roadmap-minimal-llm-charactertoken-transformer).

---

## Notes

- Model weights are not currently saved (see #21).
- Axis/broadcasting conventions in xtensor: https://www.sharpsightlabs.com/blog/numpy-axes-explained/#numpy-axes-quick-explanation
- The initializers in `include/initialization.hpp` are based on ideas from three research papers (Glorot & Bengio 2010; He et al. 2015; Mishkin & Matas 2016 for LSUV) — worth reading once you wire them in (#7), to understand *why* each scale formula looks the way it does.
