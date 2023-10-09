# imagetree

<div align="center">

[![Build status](https://github.com/jborrow/imagetree/workflows/build/badge.svg?branch=master&event=push)](https://github.com/jborrow/imagetree/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/imagetree.svg)](https://pypi.org/project/imagetree/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/jborrow/imagetree/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/jborrow/imagetree/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/jborrow/imagetree/releases)
[![License](https://img.shields.io/github/license/jborrow/imagetree)](https://github.com/jborrow/imagetree/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

ImageTree is an image processing library that uses quadtrees to simplify working with very large files.

</div>


## Installation

```bash
pip install -U imagetree
```

or install with `Poetry`

```bash
poetry add imagetree
```

### Makefile usage

[`Makefile`](https://github.com/jborrow/imagetree/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks coulb be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade`, `isort` and `black`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `isort`, `black` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```


## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/jborrow/imagetree/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.


## 🛡 License

[![License](https://img.shields.io/github/license/jborrow/imagetree)](https://github.com/jborrow/imagetree/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/jborrow/imagetree/blob/master/LICENSE) for more details.


## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)
