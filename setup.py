import os
from setuptools import setup, find_packages

version = '0.0.2'

with open('README.md') as f:
    long_description = f.read()

setup(
    name='llmtrainers',
    version=version,
    description='Packages supported for Large Language Model Trainers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/liangyuwang/Simple-LLM-Trainers",
    author='Liangyu',
    packages=find_packages(),
    python_requires=">=3.11.0",
    include_package_data=True,
    extras_require={
        "test": [
            "tqdm>=4.62.3",
        ]
    },
    install_requires=[
        "transformers>=4.36.0",
        "datasets>=2.0.0",
        "torch>=2.2.0",
    ],
    test_suite="tests",
    zip_safe=False
)
