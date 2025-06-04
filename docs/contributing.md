If you want to contribute to eyepy, this is a guide on how to do it. We are happy to accept contributions from everyone. If you have any questions, please open an issue or contact us via email.

## Project setup
This section describes the setup of the eyepy project, mainly for internal documentation purposes, but feel free to use it as a guide for your own projects.

### Dependency management and packaging
We use [Poetry](https://python-poetry.org/docs/) for dependency management and packaging. To contribute to eyepy, install Poetry as described in their [documentation](https://python-poetry.org/docs/#installation).

Install all dependencies (including development dependencies):

```bash
poetry install --with=dev
```

You will use Poetry to install the project's dependencies in a virtual environment or run the project's scripts and tests.

To run the tests in the project's virtual environment:

```bash
poetry run pytest
```

Poetry is also used to build the package and deploy it to [PyPI](https://pypi.org/project/eyepie/). This is done via [GitHub Actions] triggered by a push to the master branch.

### Code formatting and linting
Do not spend your time on code formatting. We use yapf and isort to format the code automatically. You can run them via pre-commit hooks. See the section on [pre-commit hooks](#install-and-run-the-pre-commit-hooks) for more information.

- Automatic code formatting with [pre-commit] (check-yaml, end-of-file-fixer, trailing-whitespace, isort, yapf[google], commitizen)
- Quality checks: run pre-commit hooks and pytest via [GitHub Actions] for every pull request and push to master
- Deployment for every push to master: semantic release for GitHub releases and PyPI deployment via [GitHub Actions]
- Build documentation with MkDocs and deploy it to [GitHub Pages] via [GitHub Actions]

### Semantic commit messages
- Only allow commits with [semantic commit messages] via [Commitizen]
- Automatic version number increase based on [semantic commit messages] via [Commitizen]

---

## Setting up your development environment

### Forking the repository

First, fork the repository on GitHub to your own account. Then, clone your fork to your local machine:

```bash
git clone https://github.com/[YOUR_USERNAME]/eyepy.git
cd eyepy
git checkout -b my-feature
```

### Setting up the development environment

Install dependencies (including dev dependencies):

```bash
poetry install --with=dev
```

This will create a virtual environment and install all required and development dependencies.

---

## Making and committing changes

### Run the tests

Before committing, make sure all tests pass:

```bash
poetry run pytest
```

### Install and run the pre-commit hooks

Install the pre-commit hooks (only needed once):

```bash
poetry run pre-commit install
```

Run all hooks manually before committing:

```bash
poetry run pre-commit run --all-files
```

pre-commit might change your files to match the code style. You have to add them again before committing. After installation, hooks will also run automatically on every `git commit`. Since the commit with [Commitizen] fails if the pre-commit hooks fails, run the hooks before committing.

### Commitizen for semantic commit messages

We use [Commitizen] for semantic commit messages. To create a commit message interactively:

```bash
poetry run cz commit
```

---

## Building and previewing the documentation

To build and serve the documentation locally with live reload:

```bash
poetry run mkdocs serve
```

Then open http://127.0.0.1:8000/ in your browser.

---

## Pushing and creating a pull request

### Push to your repository

After committing, push your changes to your fork:

```bash
git push origin my-feature
```

### Create a pull request

Go to the original repository on GitHub and open a pull request from your branch. Please describe your changes and reference any related issues.

---

If you have any questions, please open an issue at https://github.com/MedVisBonn/eyepy/issues. Thank you for contributing!

[pre-commit]: https://pre-commit.com
[GitHub Actions]: https://github.com/features/actions
[semantic commit messages]: https://www.conventionalcommits.org/en/v1.0.0/
[Commitizen]: https://commitizen-tools.github.io/commitizen/
