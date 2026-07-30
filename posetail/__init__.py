"""posetail: a model for tracking 2d or 3d animal pose through time.

This makes `posetail` a regular package so it is discovered by
`setuptools.find_packages()` and importable from an (editable) install regardless of
the current working directory. Subpackages: `posetail.posetail`, `posetail.datasets`.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("posetail")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+unknown"
