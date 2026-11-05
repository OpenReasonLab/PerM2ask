# Contributing

Include the dataset version, command, model, seed and affected request IDs when reporting an experiment issue. To propose a data correction, identify the source file and logical row or document span that supports it.

Use a separate output directory for new experiments. Install the test dependencies with `python -m pip install ".[test]"` and run `python -m pytest tests` from the repository root. Keep credentials and real patient records out of issues and pull requests.
