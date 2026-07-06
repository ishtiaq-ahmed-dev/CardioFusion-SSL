# Contributing to CardioFusion-SSL

Thanks for your interest in improving CardioFusion-SSL! Contributions are welcome via issues and pull requests.

## Reporting bugs

1. Check existing issues first — someone may already have reported it.
2. Open a new issue with:
   - A minimal reproduction (Python snippet or command)
   - Expected vs. actual behaviour
   - Your environment: OS, Python version, PyTorch version, GPU model
   - Full error traceback if applicable

## Proposing new features

Open an issue describing the feature and the use case before writing code. This avoids duplicated effort and lets us discuss scope up front.

## Pull requests

1. **Fork the repository** and create a topic branch from `main`.
2. **Follow the code style** used in the project (see `configs/cfg.py` for hyperparameter conventions and `models/*.py` for architecture patterns).
3. **Add tests** if you change model behaviour or add new scripts.
4. **Update documentation** — if you change the CLI or add a new script, update the README and any relevant sections.
5. **Keep commits focused** — one logical change per commit, with a descriptive message.
6. **Reference issues** in the PR description (e.g. "Fixes #12").

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/CardioFusion-SSL.git
cd CardioFusion-SSL
python -m venv venv
# Activate venv, then:
pip install -r requirements.txt
pip install pytest pytest-cov  # for tests

# Verify everything works
python -m scripts.smoke_test
pytest tests/
```

## Areas where help is especially welcome

- **Multi-lead ECG extension** — the current architecture uses lead I only; extending the patch-stem to accept all 12 leads
- **Multi-class disease-subtype heads** — the current binary classifier can be replaced with a multi-head design when larger paired corpora become available
- **Test-time augmentation** — currently only mean-pooling of overlapping windows; more sophisticated TTA may improve external validation results
- **Threshold calibration for extreme-imbalance deployment** — automatic prior-adjusted threshold selection at inference time
- **Additional external datasets** — new PCG or ECG public datasets to expand the cross-dataset evaluation

## Code of conduct

Be respectful and constructive in all interactions. Reviews focus on the code, not the person.

## Questions

If you have questions about the codebase or the paper methodology, open a GitHub Discussion or contact the corresponding author directly.
