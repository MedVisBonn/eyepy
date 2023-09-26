If you want to contribute to eyepy this is a guide on how to do it. We are happy to accept contributions from everyone. If you have any questions, please open an issue or contact us via email.

## Project setup
In the following the setup of the eyepy project is described, mainly for internal documentation purposes, but feel free to use it as a guide for your own projects.

### Depdency management and packaging
We use [poetry](https://python-poetry.org/docs/) for dependency management and packaging. Hence when you want to contribute to eyepy you need to install poetry first. Do so as described in the [documentation](https://python-poetry.org/docs/#installation).

You will use Poetry to install the project's dependencies in a virtual environment or run the project's scripts and tests.

Running the tests in the projects virtual environment is as simple as running

```bash
poetry run pytest
```

Internally poetry is also used to build the package and deploy it to [PyPI](https://pypi.org/project/eyepie/). This is done via [GitHub Actions] triggered by a push to the master branch.

### Code formatting and linting
Do not spend your time on code formatting. We use yapf and isort to format the code automatically. You can run them via pre-commit hooks. See the section on [pre-commit hooks](#pre-commit-hooks) for more information.

+ Automatic code formatting with [pre-commit] check-yaml, end-of-file-fixer, trailing-whitespace, isort, yapf[google], commitizen

+ Quality checks - run pre-commit hooks and pytest via [GitHub Actions] for every pull request and push to master

Deployment for every push to master
+ Use Semantic release for Github releases and PyPI deployment via [GitHub Actions]
+ Build documentation with mkdocs and deploy it to [GitHub Pages] via [GitHub Actions]

### Semantic commit messages
+ Only allow commits with [semantic commit messages] via [commitizen]
+ Automatic increase of version number based on [semantic commit messages] via [commitizen]


## Setting up your development environment

### Forking the repository

First, you need to fork the repository. This will create a copy of the repository under your own account. You can do this by clicking on the "Fork" button in the top right corner of the repository page.
After that, you need to clone your fork to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/[YOUR_USERNAME]/eyepy.git
```

### Setting up the development environment
We use poetry for dependency management and building

### Commiting changes

#### Run the tests

#### Run the pre-commit hooks

#### Commitizen for semantic commit messages

### Push to your repository

### Create a pull request
