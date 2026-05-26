import re
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
    # Strip inline HTML paragraph tags from the README before publishing.
    long_description = re.sub(r"<p.*?>.*?</p>", "", long_description, flags=re.DOTALL)

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line for line in f.read().splitlines() if line and not line.startswith("#")]

setup(
    name="knowledge-storm",
    version="1.1.1",
    description="STORM: A language-model-powered knowledge curation and article generation engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
