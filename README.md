# Python Open Utils

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/devskale/python-openutils)](https://github.com/devskale/python-openutils/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/devskale/python-openutils)](https://github.com/devskale/python-openutils/issues)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A collection of powerful Python utilities for secure credential management and unified LLM inference.

## Packages

### [Credgoo](packages/credgoo/)
A Python package for securely retrieving API keys from Google Sheets with local caching.

**Features:**
- Secure key retrieval from Google Sheets
- Built-in decryption with custom encryption keys
- Local caching with restrictive permissions
- Command-line interface for easy key retrieval
- Centralized credential management

### [UniInfer](packages/uniinfer/)
Unified Inference API for LLM chat completions across 15+ providers with streaming, fallback strategies, and secure API key management.

**Features:**
- Single API for 20+ LLM providers (OpenAI, Anthropic, Google Gemini, Mistral, etc.)
- Secure API key management with credgoo integration
- Real-time streaming support
- Text-to-Speech (TTS) and Speech-to-Text (STT) support
- Embedding support for semantic search
- Automatic fallback strategies
- OpenAI-compatible FastAPI proxy server

## Quick Start

### Install Individual Packages

```bash
# Install credgoo
pip install git+https://github.com/skale-dev/python-utils#subdirectory=packages/credgoo

# Install uniinfer
pip install uniinfer credgoo
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