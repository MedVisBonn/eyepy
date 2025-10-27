import nox  # type: ignore - nox is not a project dependency. This module is used by the global nox command, not this project


@nox.session(
    python=['3.9', '3.10', '3.11', '3.12', '3.13'],
    venv_backend='uv',
)
def tests(session: nox.Session):
    session.run_install(
        'uv',
        'sync',
        '--all-groups',
        f'--python={session.virtualenv.location}',
        env={'UV_PROJECT_ENVIRONMENT': session.virtualenv.location},
    )
    session.run('pytest', *session.posargs)
