from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

setup(
    name="arkml",
    description="Machine learning backbone for the Ark Robotics Framework, providing core models, algorithms, and tools to enable intelligent perception, decision-making, and control in robotic applications.",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT License",
    install_requires=requirements,
)
