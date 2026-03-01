"""Setup configuration for the SAE ML pipeline package."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sae-prediction",
    version="0.1.0",
    author="michw15",
    description=(
        "Machine Learning pipeline for predicting Serious Adverse Events "
        "(SAE) in clinical trials."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michw15/sae",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "sae-pipeline=src.pipeline:main",
        ]
    },
)
