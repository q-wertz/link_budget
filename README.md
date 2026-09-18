# Link budget calculator

This is a link budget calculator for EM waves. Additionally the DAC conversion can be investigated.

## Usage

### Installing required Python dependencies

For packaging and dependency management [Poetry](https://python-poetry.org/) is used. Refer to the [documentation](https://python-poetry.org/docs) for in-depth information.

You can simply set up a new virtual environment with the required packages (which is placed in the project subfolder `.venv`) by running:

```bash
uv venv
```

You can run the marimo notebook using
```bash
# For editing
uv run marimo edit link_budget.py

# Start in read only web view
uv run marimo run link_budget.py
```

## Additional features

If you would like to see more features [open a new issue](https://github.com/q-wertz/link_budget/issues) or a pull request if you want to contribute.


## Contributing

Please read the following guidelines for contributing code:

- [Gitlab, "Version control best practices", 2020, pdf](https://learn.gitlab.com/c/version-control-best-practice?x=-RIZtH)


### Dependencies

#### Poetry
Updating dependencies can be done using
```bash
# Update the uv.lock file
uv sync --upgrade
```
