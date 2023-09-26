import nox  # type: ignore - nox is not a project dependency. This module is used by the global nox command, not this project


@nox.session(python=['3.8', '3.9', '3.10', '3.11'])
def tests(session):
    # Install the project dependencies
    session.install('poetry')
    session.run('poetry', 'install')

    # Run the tests
    session.run('pytest', *session.posargs)
