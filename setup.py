import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fo:
    readme = fo.read()

setup(
    name="the-daily-train",
    version="0.1.0",
    description="Training models erryday",
    author="Mancer Labs",
    url="https://github.com/Algomancer/The-Daily-Train",
    install_requires=[
        "torch>=2.1.0",
        "lightning @ git+https://github.com/Lightning-AI/lightning@6cbe9ceb560d798892bdae9186291acf9bf5d2e3",
    ],
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)
