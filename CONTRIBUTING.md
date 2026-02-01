# Contributing

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/quantum-kernel-ids.git
   cd quantum-kernel-ids
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

We follow PEP 8 with the following tools:

```bash
# Format code
black src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/
```

### Testing

Run tests before submitting:

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Documentation

- Add docstrings to all public functions
- Update relevant documentation in `docs/`
- Include examples for new features

## Submitting Changes

1. Ensure all tests pass
2. Update documentation if needed
3. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description of change"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request

## Pull Request Guidelines

- Describe what the PR does and why
- Reference any related issues
- Include test results
- Keep changes focused and atomic

## Reporting Issues

When reporting issues, please include:

- Python version
- Qiskit version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Questions?

Open an issue with the "question" label for any questions about contributing.
