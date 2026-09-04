# Maintenance

eyepy uses `dev` as its integration branch and `master` as its release branch.
Normal changes are merged into `dev`; a reviewed pull request from `dev` to
`master` runs the release workflow. Urgent fixes for an already published
version may target `master` directly and must then be merged back into `dev`.

## Canonical checks

Install the complete development environment from the committed lock file:

```bash
uv sync --all-groups --all-extras --locked
```

Run the same principal checks used by CI:

```bash
uv lock --check
uv run --locked pytest --cov=eyepy
uv run --locked pre-commit run --all-files
uv run --locked mkdocs build --strict
uv build
```

For a core-only installation, omit `--all-extras`. Individual optional
features can be selected with `--extra plot`, `--extra quant`, `--extra tiff`,
`--extra pandas`, `--extra fda`, or `--extra itk`.

## Dependency updates

Dependabot proposes grouped updates once per month against `dev`. Runtime major
updates and file-reader changes should be reviewed separately from routine
tooling updates. Every dependency pull request must include the regenerated
`uv.lock` and pass the complete CI suite.

Avoid upper bounds for runtime dependencies unless an incompatibility is known
and reproduced. When a lower bound is changed, run the lowest-direct-dependency
CI job to ensure that it remains a tested compatibility claim.

## Quarterly review

Once per quarter, review:

- supported Python versions and deprecation warnings;
- optional extras and their missing-dependency error messages;
- package metadata, documentation links, and wheel/sdist contents;
- representative VOL, XML, E2E, TIFF, and `.eye` round trips;
- open compatibility and release-infrastructure issues;
- the no-release and successful-release workflow paths.

Security fixes affecting the published package should not wait for the monthly
update cycle. Prepare a focused pull request to `master`, publish the patch, and
merge the result back into `dev`.
