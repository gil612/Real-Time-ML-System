from setuptools import setup, find_packages

setup(
    name="news",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "loguru>=0.7.3",
        "pandas==2.2.0",
        "numpy==1.26.4",
        "pydantic-settings>=2.6.1",
        "quixstreams>=3.4.0",
        "rarfile>=4.2",
        "requests>=2.32.3",
    ],
    python_requires=">=3.10",
)
