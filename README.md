[TOC]

## Link budget calculator

This is a link budget calculator for EM waves. Additionally the DAC conversion can be investigated.

### Usage

#### Installing required Python dependencies

For packaging and dependency management [Poetry](https://python-poetry.org/) is used. Refer to the [documentation](https://python-poetry.org/docs) for in-depth information.

You can simply set up a new virtual environment (in the project subfolder `.venv`) with the required packages by running:

```bash
poetry install

# Without the development dependencies
poetry install --no-dev
```

If you want to open a shell in the virtual environment, simply run

```bash
poetry shell
```

You can list all poetry virtual environments using

```bash
poetry env list
```

You can run the marimo notebook using
```bash
# For editing
marimo edit main.py

# Start in read only web view
marimo run main.py
```

## Additional features

If you would like to see more features open an issue or a pull request if you want to contribute.


## Contributing

Please read the following guidelines for contributing code:

- [Gitlab, "Version control best practices", 2020, pdf](https://learn.gitlab.com/c/version-control-best-practice?x=-RIZtH)


### Dependencies

#### Poetry
Updating dependencies can be done using
```bash
# Update the poetry.lock file
poetry update
```
Updating also the set requirements from the `pyproject.toml` file can be done using a poetry plugin:
```bash
# Install the plugin
poetry self add poetry-plugin-up
# Update the dependency list
poetry up
# To check for outdated dependencies you can use
poetry show --outdated
```

#### Deptry
To check for obsolete, missing or transitive dependencies in `pyproject.toml` from the project root folder run:
```shell
deptry .
```
