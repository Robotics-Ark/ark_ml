# Pi0.5 Tests and Benchmarks

This directory contains comprehensive tests and benchmarks for the Pi0.5 implementation in the ArkML framework.

## Directory Structure

```
tests_and_benchmarks/
├── pi05_tests/              # Unit and component tests for Pi0.5 functionality
├── pi05_benchmarks/         # Performance benchmarks for Pi0.5 components
└── README.md               # This file
```

## Test Files

### `pi05_tests/` - Unit and Integration Tests

- **`test_pi05_components.py`** - Component-specific tests
  - Tests Pi05 configuration utilities and training stage updates
  - Tests Pi05Dataset initialization and data format
  - Tests data loading and collate functions
  - Tests statistical computation and normalization functions
  - Tests algorithm integration with mocked components

- **`test_pi05_models.py`** - Model-specific tests
  - Tests flow matching loss functions (basic and edge cases)
  - Tests ActionFlowExpert functionality (training, inference, prediction)
  - Tests Pi05Policy with mocked LeRobot integration
  - Tests device management and mode switching methods

### `pi05_benchmarks/` - Performance Benchmarks

- **`benchmark_pi05.py`** - Comprehensive performance testing
  - Benchmarks flow matching loss computation speed
  - Benchmarks ActionFlowExpert inference operations
  - Benchmarks ActionFlowExpert training operations
  - Benchmarks memory usage for different components
  - Runs performance regression tests

## Running Tests

```bash
# Run all Pi0.5 tests
python -m pytest tests_and_benchmarks/pi05_tests/ -v

# Run specific test file
python -m pytest tests_and_benchmarks/pi05_tests/test_pi05_components.py -v

# Run all benchmarks
python tests_and_benchmarks/pi05_benchmarks/benchmark_pi05.py
```

## Test Categories

- **Unit Tests**: Test individual components in isolation (tokenizers, loss functions, utilities)
- **Component Tests**: Test integration between related components (dataset, config utils, algorithms)

## Notes

- Tests that require real HuggingFace model access use mocked models to avoid network dependencies
- All tests should pass in a properly configured environment
- Benchmarks provide performance metrics for optimization and regression tracking