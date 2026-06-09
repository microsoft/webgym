"""AsyncWebRL package setup — thin wrapper for backward compatibility.

Dependency versions are specified in requirements.txt; install them with
whatever environment you use (pip, uv, conda, container, ...).

The browser environment (Omniboxes) is a separate service and is no longer
shipped here — see https://github.com/microsoft/webgym/tree/webgym and
https://webgym.readthedocs.io/en/latest/server/quickstart_server.html.
"""

import setuptools

setuptools.setup(
    name="asyncwebrl",
    version="1.0.0",
    author="Hao Bai",
    description=(
        "A scalable reinforcement learning framework for training "
        "web automation agents using vision-language models."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="asyncwebrl",
    license="MIT",
    packages=setuptools.find_packages(include=["webgym", "webgym.*"]),
    # No install_requires here — all deps are declared in pyproject.toml
    # to avoid version conflicts between AReaL and WebGym.
    install_requires=[],
    include_package_data=True,
    python_requires=">=3.12",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
