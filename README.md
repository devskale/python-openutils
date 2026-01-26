# Python Open Utils

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/devskale/python-openutils)](https://github.com/devskale/python-openutils/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/devskale/python-openutils)](https://github.com/devskale/python-openutils/issues)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A collection of powerful Python utilities for secure credential management and unified LLM inference.

## Packages

### [Credgoo](packages/credgoo/)

**The Secure Credentials Manager for Everywhere.**

Securely manage your credentials in Google Sheets with encrypted Apps Script integration and access them seamlessly across all your environments.

**Features:**

- **Use Everywhere**: Unified access for local, dev, and production.
- **Secure Google Sheets Integration**: Encrypted Apps Script backend.
- **End-to-End Encryption**: Secure transmission and storage.
- **Smart Local Caching**: Performance optimized with restrictive permissions.
- **Centralized Control**: Single source of truth for all your secrets.

### [UniInfer](packages/uniinfer/)

OpenAI-compatible proxy server providing unified access to 500+ models from 20+ providers with multi-modal support.

**Features:**

- OpenAI-compatible FastAPI proxy - Drop-in replacement for OpenAI API
- 500+ models across 20+ providers (OpenAI, Anthropic, Google Gemini, Mistral, etc.)
- Multi-modal support: LLM, Image, Text-to-Speech (TTS), and Transcription
- Secure API key management with credgoo integration
- Real-time streaming support
- Embedding support for semantic search
- Automatic fallback strategies

## Quick Start

### Install Individual Packages

```bash
# Install credgoo
pip install -r https://skale.dev/credgoo

# Install uniinfer from PyPI
pip install -r https://skale.dev/uniinfer

```

### Development Setup

This repository includes a utility script `uvinit.sh` for setting up development environments for all packages.

```bash
# Initialize all packages with uv (creates venvs, installs deps)
./uvinit.sh -x

# Initialize a specific package
./uvinit.sh -x credgoo

# Upgrade all packages
./uvinit.sh -u

# Clean all venvs
./uvinit.sh -c
```

### uvinit.sh Options

```
-x            Initialize projects with uv (discover, venv, install, activate)
-v            Create venvs with uv for matched projects
-p            Install matched projects into their venvs
-u            Upgrade installs via uv pip
-c            Remove venvs for matched projects
-s            Silent mode (no prompts, concise output)
-i            Interactive mode (confirm per-project)
-D            Use discovery mode (ignore predefined package list)
```

## Documentation

- [Credgoo Documentation](packages/credgoo/README.md)
- [UniInfer Documentation](packages/uniinfer/README.md)

## License

MIT License - see individual package directories for specific license details.
