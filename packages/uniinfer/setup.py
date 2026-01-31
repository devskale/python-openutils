import os
import subprocess
import sys
from urllib.request import urlopen

from setuptools import find_packages, setup

REMOTE_REQUIREMENTS = [
    "https://skale.dev/credgoo",
]


def get_remote_requirements() -> list[str]:
    """Fetch and parse remote requirements files."""
    requirements = []
    for url in REMOTE_REQUIREMENTS:
        try:
            with urlopen(url) as response:
                content = response.read().decode("utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if line.startswith("git+"):
                        if "subdirectory=" in line:
                            # Extract package name from subdirectory
                            # e.g. ...#subdirectory=packages/uniinfer -> uniinfer
                            subdir = line.split("subdirectory=")[1]
                            pkg_name = subdir.split("/")[-1]
                            requirements.append(f"{pkg_name} @ {line}")
                        else:
                            # If no subdirectory, use the line as is if it's a valid requirement
                            requirements.append(line)
                    else:
                        requirements.append(line)
        except Exception as e:
            print(f"Warning: Could not fetch requirements from {url}: {e}")
    return requirements


def install_remote_requirements() -> None:
    """Install external requirement bundles for installs."""
    if not {"install", "develop"} & set(sys.argv) or "bdist_wheel" in sys.argv:
        return

    for url in REMOTE_REQUIREMENTS:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", url])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install requirements from {url}: {e}")


install_remote_requirements()

# Get all requirements
all_requirements = get_remote_requirements() + [
    "requests>=2.25.0",
    "python-dotenv>=1.0.0",
    "openai",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic",
    "python-multipart",
    "importlib-metadata; python_version<'3.8'",
]

setup(
    name="uniinfer",
    version="0.4.5",
    url="https://github.com/devskale/python-openutils",
    description="Unified Inference API for LLM chat completions across 15+ providers with streaming, fallback strategies, and secure API key management",
    long_description="UniInfer provides a unified interface for LLM inference across 15+ providers including OpenAI, Anthropic, Google Gemini, Mistral, and more. Features include real-time streaming, automatic fallback strategies, secure API key management via credgoo, and OpenAI-compatible proxy server.",
    long_description_content_type="text/plain",
    author="Han Woo",
    author_email="dev@skale.dev",
    keywords="llm, ai, inference, openai, anthropic, gemini, mistral, chat, completion, streaming, api",
    packages=find_packages(where=".", include=["uniinfer*"]),
    install_requires=all_requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    extras_require={
        "gemini": ["google-genai>=1.38.0"],
        "anthropic": ["anthropic>=0.25.0"],
        "mistral": ["mistralai>=0.4.0"],
        "cohere": ["cohere>=4.0.0"],
        "huggingface": ["huggingface-hub>=0.20.0"],
        "groq": ["groq>=0.11.0"],
        "ai21": ["ai21>=2.1.0"],
        "all": [
            "google-genai>=1.38.0",
            "anthropic>=0.25.0",
            "mistralai>=0.4.0",
            "cohere>=4.0.0",
            "huggingface-hub>=0.20.0",
            "groq>=0.11.0",
            "ai21>=2.1.0",
        ],
    },
    package_data={
        "uniinfer": ["*.py", "examples/*.py", "examples/webdemo/*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "uniinfer=uniinfer.uniinfer_cli:main",
            "uniioai-proxy=uniinfer.uniioai_proxy:main",  # <-- Add this line
        ],
    },
)
