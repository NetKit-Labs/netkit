# TinyRT - Neural Network Inference Engine

TinyRT is a lightweight, high-performance inference engine written in modern C++ (C++17/20) designed for neural network inference on microcontroller units (MCUs) and microprocessor units (MPUs). It provides an optimized runtime for deploying trained neural network models directly on resource-constrained embedded devices.

## Features

- **Dense/MLP Layers** - Full support for fully connected neural network layers
- **2D Convolution** - Efficient 2D convolutional operations (1D convolution can be performed via 2D)
- **Activation Functions** - Comprehensive set of activation functions:
  - ReLU, LeakyReLU, ReLU6
  - Sigmoid, Tanh
  - Softmax
- **Memory-Efficient Arena Allocator** - Custom memory management for embedded environments
- **N-Dimensional Tensor Support** - Flexible tensor operations for various model architectures
- **NHWC Layout** - Optimized for common embedded tensor memory layout

## Architecture

### Core Components

- **arena.hpp/cpp** - Memory arena allocator for efficient memory management
- **tensor.hpp** - Tensor data structure and type definitions
- **tensor_factory.hpp/cpp** - Tensor creation and manipulation utilities
- **tensor_access.hpp/cpp** - Direct tensor data access and indexing functions
- **ops.hpp/cpp** - Core neural network operations (linear algebra, activations)
- **conv2d.hpp/cpp** - 2D convolution implementation
- **test.hpp/cpp** - Comprehensive test suite for validation

## Building

### Requirements
- C++17 or later compiler (clang++, g++, MSVC)
- Make (for Unix-like systems)

### Build Commands

```bash
# Build the project
make

# Clean build artifacts
make clean

# Rebuild from scratch
make rebuild

# Build and run tests
make run
```

### Manual Compilation

```bash
clang++ -std=c++17 -g -o main main.cpp test.cpp arena.cpp tensor_factory.cpp tensor_access.cpp ops.cpp conv2d.cpp
```

## Usage Example

### MLP (Dense Layer) Inference

```cpp
#include "arena.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"
#include "ops.hpp"

using namespace TensorFactory;
using namespace Ops;

// Create memory arena
unsigned char buffer[1024];
Arena arena;
arena.init(buffer, 1024);

// Create input tensor
Tensor x = Create2D(arena, 1, 2);
Fill(x, (float[]){1.0f, 2.0f});

// Create weights and bias
Tensor W = Create2D(arena, 2, 3);
Fill(W, (float[]){1, 2, 3, 4, 5, 6});

Tensor b = Create2D(arena, 1, 3);
Fill(b, (float[]){0, 0, 0});

// Forward pass
Tensor h = Create2D(arena, 1, 3);
MatMul(x, W, h);
MatAdd(h, b, h);
ReLU(h, h);

Print(h);
```

### 2D Convolution

```cpp
#include "conv2d.hpp"

// Input: 4x4 with 1 channel
float input_data[16] = {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1};
Tensor input;
input.data = input_data;
input.rank = 3;
input.shape[0] = 4; // height
input.shape[1] = 4; // width
input.shape[2] = 1; // channels

// Output: 2x2 with 1 channel
float output_data[4] = {0};
Tensor output;
output.data = output_data;
output.rank = 3;
output.shape[0] = 2;
output.shape[1] = 2;
output.shape[2] = 1;

// 3x3 kernel
float weights[9] = {1,0,1, 0,0,0, 1,0,1};
float bias[1] = {0};

Conv2D conv;
conv.kernel_size = 3;
conv.stride = 1;
conv.in_channels = 1;
conv.out_channels = 1;
conv.weights = weights;
conv.bias = bias;

conv.forward(input, output);
```

## Project Structure

```
tinyrt/
├── arena.hpp/cpp           # Memory management
├── tensor.hpp              # Tensor definitions
├── tensor_factory.hpp/cpp  # Tensor utilities
├── tensor_access.hpp/cpp   # Tensor access functions
├── ops.hpp/cpp             # Neural network operations
├── conv2d.hpp/cpp          # 2D convolution
├── test.hpp/cpp            # Test suite
├── main.cpp                # Entry point
├── Makefile                # Build configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Supported Operations

### Arithmetic
- `MatMul()` - Matrix multiplication
- `MatAdd()` - Element-wise addition
- `Mul()` - Element-wise multiplication
- `MulScalar()` - Scalar multiplication
- `MatAddND()` - N-dimensional addition
- `MulND()` - N-dimensional multiplication

### Activation Functions
- `ReLU()` - Rectified Linear Unit
- `LeakyReLU()` - Leaky ReLU with configurable alpha
- `ReLU6()` - ReLU capped at 6
- `Sigmoid()` - Sigmoid activation
- `Tanh()` - Hyperbolic tangent
- `Softmax()` - Softmax normalization

## Design Principles

- **Lightweight** - Minimal dependencies, suitable for embedded environments
- **Memory-Conscious** - Custom arena allocator for predictable memory usage
- **Single-Threaded** - No parallelization overhead (suitable for MCUs)
- **Inference-Only** - Optimized for deployment, not training
- **Type-Safe** - Modern C++ with strong typing

## Roadmap

### Near Term
- Enhanced quantization support (int8, uint8)
- Batch normalization layer
- Pooling operations (max, average)

### Future Enhancements
- On-device fine-tuning capabilities
- Transformer architecture support
- Automatic differentiation (autodiff) for training
- Optimized kernels for specific MCU/MPU architectures
- Model serialization/deserialization

## Testing

Run the comprehensive test suite:

```bash
make run
```

Tests include:
- MLP forward pass validation
- 2D convolution with various channel configurations
- Multi-channel to single-channel convolution
- Multi-channel to multi-channel convolution

## Performance Considerations

- Tensors use row-major (C-style) layout
- NHWC format for convolutional layers (optimized for MCU memory patterns)
- In-place operations where possible to reduce memory allocation
- Linear indexing for fast tensor access

## License

[Add license information here]

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All tests pass
- New features include test coverage
- Documentation is updated

## Contact & Support

[Add contact information here]
