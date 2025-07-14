# Testing Guide for Modified LeRobot Components

This guide explains how to test the modified components in LeRobot, including the improvements made to the Atari teleoperator, Tatbot robot, and IP camera functionality.

## Overview

The following components have been modified and tested:

1. **Atari Teleoperator** (`lerobot/teleoperators/gamepad/atari_teleoperator.py`)
   - Implemented TODOs A-1, A-2, A-3
   - Unbounded queue to prevent silent drops
   - Improved normalization calculation
   - Queue full event logging

2. **Tatbot Robot** (`lerobot/robots/tatbot/tatbot.py`)
   - Implemented TODOs T-1 through T-8
   - Programmatic joint generation
   - DRY helper methods for arm operations
   - Vectorized mismatch checking
   - Optimized logging
   - Parallel camera reading
   - Simultaneous arm movement
   - Improved URDF joint mapping

3. **IP Camera Support** (`lerobot/cameras/opencv/`)
   - Enhanced OpenCV camera with IP camera support
   - RTSP stream handling
   - Authentication support

## Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── teleoperators/
│   └── test_atari_teleoperator.py    # Atari teleoperator tests
├── robots/
│   └── test_tatbot.py                # Tatbot robot tests
├── cameras/
│   └── test_ip_camera.py             # IP camera tests
└── conftest.py                       # Pytest configuration
```

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install pytest pytest-timeout pytest-cov
```

2. Ensure you're in the lerobot directory:
```bash
cd /path/to/lerobot
```

### Running All Tests

```bash
python -m pytest tests/ -v
```

### Running Specific Test Suites

```bash
# Atari teleoperator tests
python -m pytest tests/teleoperators/test_atari_teleoperator.py -v

# Tatbot robot tests
python -m pytest tests/robots/test_tatbot.py -v

# IP camera tests
python -m pytest tests/cameras/test_ip_camera.py -v
```

### Using the Test Runner Script

A convenience script is provided:

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py atari
python run_tests.py tatbot
python run_tests.py ip_camera
```

## Test Coverage

### Atari Teleoperator Tests

**Configuration Tests:**
- Default configuration values
- Custom configuration parameters
- Queue size settings (unbounded vs bounded)

**Functionality Tests:**
- Initialization and properties
- Joystick detection and connection
- Event processing and queue management
- Button press and axis movement handling
- Normalization calculation improvements
- Queue full event logging

**Integration Tests:**
- Complete connection/disconnection cycles
- Multiple event processing
- Axis threshold filtering

### Tatbot Robot Tests

**Configuration Tests:**
- TatbotConfig initialization
- Parameter validation

**Core Functionality Tests:**
- Programmatic joint generation (T-1)
- Helper methods for position getting/setting (T-2, T-3)
- Vectorized mismatch checking (T-3)
- Optimized logging with DEBUG checks (T-4)
- Motor state dictionary building (T-5)
- Parallel camera reading (T-6)
- Simultaneous arm movement (T-7)
- URDF joint mapping improvements (T-8)

**Integration Tests:**
- Complete robot operation cycles
- Parallel camera reading performance
- Error handling and recovery

### IP Camera Tests

**Configuration Tests:**
- Basic IP camera configuration
- Environment variable password support
- Custom RTSP port configuration

**Functionality Tests:**
- IP camera initialization
- Connection and disconnection
- Frame reading (synchronous and asynchronous)
- Error handling
- String representation (password hiding)

## Test Features

### Mocking Strategy

All tests use comprehensive mocking to avoid requiring actual hardware:

- **evdev**: Mocked for Atari teleoperator tests
- **trossen_arm**: Mocked for Tatbot robot tests
- **OpenCV**: Mocked for camera tests
- **Threading**: Properly handled in async tests

### Performance Testing

Some tests include performance measurements:

- Parallel camera reading timing
- Queue processing efficiency
- Memory usage validation

### Error Handling

Tests cover various error scenarios:

- Connection failures
- Device not found
- Invalid configurations
- Timeout conditions
- Exception handling

## Test Output Examples

### Successful Test Run

```
============================= test session starts ==============================
platform linux -- Python 3.11.10, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/oop/lerobot
plugins: timeout-2.4.0, cov-3.0.0
collected 25 items

tests/teleoperators/test_atari_teleoperator.py::TestAtariTeleoperatorConfig::test_default_config PASSED
tests/teleoperators/test_atari_teleoperator.py::TestAtariTeleoperatorConfig::test_custom_config PASSED
tests/teleoperators/test_atari_teleoperator.py::TestAtariTeleoperator::test_initialization PASSED
...

============================== 25 passed in 2.34s ==============================
```

### Test with Failures

```
tests/robots/test_tatbot.py::TestTatbot::test_connect_disconnect FAILED
tests/robots/test_tatbot.py::TestTatbot::test_get_observation PASSED
...

============================== 1 failed, 24 passed in 2.45s ==============================
```

## Debugging Tests

### Verbose Output

Use `-v` flag for detailed output:

```bash
python -m pytest tests/teleoperators/test_atari_teleoperator.py -v
```

### Debug Mode

Use `-s` flag to see print statements:

```bash
python -m pytest tests/teleoperators/test_atari_teleoperator.py -s
```

### Specific Test

Run a specific test method:

```bash
python -m pytest tests/teleoperators/test_atari_teleoperator.py::TestAtariTeleoperator::test_connect_success -v
```

### Coverage Report

Generate coverage report:

```bash
python -m pytest tests/ --cov=lerobot --cov-report=html
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install pytest pytest-timeout pytest-cov
    python -m pytest tests/ -v --cov=lerobot
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct directory and have installed dependencies
2. **Mock Issues**: Check that mock patches target the correct import paths
3. **Threading Issues**: Some tests involve threading; ensure proper cleanup
4. **Timeout Issues**: Increase timeout values for slower systems

### Environment Setup

```bash
# Install dependencies
pip install -e .
pip install pytest pytest-timeout pytest-cov

# Set up environment variables if needed
export CAMERA_PASSWORD="your_password"

# Run tests
python -m pytest tests/ -v
```

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use descriptive test names
3. Include both unit and integration tests
4. Mock external dependencies
5. Test error conditions
6. Document any special setup requirements

## References

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [LeRobot documentation](https://github.com/huggingface/lerobot) 