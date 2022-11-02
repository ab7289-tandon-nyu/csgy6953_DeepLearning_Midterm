from setuptools import setup

dependencies = [
    "torch",
    "torchvision",
    "matplotlib",
    "jupyter",
    "notebook",
    "torch-summary",
]

setup(
    name="ResNetSrc",
    version="0.0.1-dev",
    packages=["src"],
    install_requires=dependencies,
    include_package_data=True,
)