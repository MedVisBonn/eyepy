import nox

nox.options.default_venv_backend = 'uv'


@nox.session(python=['3.10', '3.11', '3.12', '3.13'])
@nox.parametrize('extras', [None, 'all'])
def tests(session, extras):
    """Run tests with and without optional dependencies."""
    session.install('pytest', 'pytest-cov')

    if extras == 'all':
        session.install('.[all]')
    else:
        session.install('.')

    session.run('pytest')
