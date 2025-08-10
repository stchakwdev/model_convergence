"""Setup configuration for Universal Alignment Patterns package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="universal-alignment-patterns",
    version="0.1.0",
    author="Samuel Chakwera",
    author_email="samuel.chakwera@example.com",
    description="Discovering universal patterns in AI model alignment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/samueltchakwera/universal-alignment-patterns",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "advanced": [
            "umap-learn>=0.5.3",
            "gudhi>=3.8.0", 
            "networkx>=3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "universal-patterns=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)