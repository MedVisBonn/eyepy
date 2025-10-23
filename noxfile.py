import nox  # type: ignore - nox is not a project dependency. This module is used by the global nox command, not this project


@nox.session(python=['3.9', '3.10', '3.11'])
def tests(session):
    # Install the project dependencies using uv (supports PEP 735 dependency groups)
    session.install('uv')
    session.run('uv', 'sync', '--all-groups', '--python', session.python)

    # Run the tests
    session.run('uv', 'run', 'pytest', *session.posargs)
