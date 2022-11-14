from setuptools import setup

dependencies = [
    "torch",
    "torchvision",
    "matplotlib",
    "jupyter",
    "notebook",
    "torch-summary",
]

test_dependencies = [
    "pytest",
    "flake8",
    "black",
    "isort",
    "coverage",
    "coveralls",
    "tox",
    "torch"
]

setup(
    name="ResNetSrc",
    version="0.0.1-dev",
    packages=["src"],
    install_requires=dependencies,
    include_package_data=True,
    tests_require=test_dependencies,
    test_suite="tests",
)