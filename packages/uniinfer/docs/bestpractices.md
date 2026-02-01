# Python Package Best Practices

## Versioning

### Single Source of Truth

Avoid duplicating the version number in multiple places (e.g., `setup.py`, `__init__.py`, `version.py`). Instead, maintain the version in a single location—your package configuration file—and retrieve it dynamically at runtime.

#### Recommended Approach

1.  **Define Version in `setup.py` (or `pyproject.toml`)**:
    This is the definitive source for packaging tools (pip, uv, etc.).

    ```python
    # setup.py
    setup(
        name="my-package",
        version="0.1.0",
        ...
    )
    ```

2.  **Retrieve Dynamically in `__init__.py`**:
    Use `importlib.metadata` (stdlib in Python 3.8+) to fetch the installed version. This ensures `mypackage.__version__` always matches what `pip list` shows.

    ```python
    # mypackage/__init__.py
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        # Backport for older Python versions if needed
        from importlib_metadata import version, PackageNotFoundError

    try:
        __version__ = version("my-package")
    except PackageNotFoundError:
        # Package is not installed
        __version__ = "unknown"
    ```

#### Benefits

*   **Consistency**: No risk of `setup.py` saying "1.0.0" while `__init__.py` says "0.9.0".
*   **Simplicity**: You only update the version in one file when releasing.
*   **Compatibility**: Works well with modern build tools and editable installs (`pip install -e .`).

#### Note on Editable Installs

When using this approach, if you bump the version in `setup.py`, you must reinstall the package (even in editable mode) for the metadata to update:

```bash
uv pip install -e .
# or
pip install -e .
```
