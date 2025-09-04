import argparse
from pathlib import Path

readme_template = """# {repo_name}

{short_description}

# Install

1. Create and activate conda/virtual environment.
2. Clone repository.
  - (ssh) `git clone {ssh}`
  - (https) `git clone {https}`
3. Change directory: `cd {repo_name}`
4. Install `pip install -e .`
"""

setup_py = """from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

setup(
    name="{pkg_name}",
    description="{short_description}",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT License",
    install_requires=requirements,
)
"""


class Arguments:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Generate an Ark Python package.")
        parser.add_argument(
            "--short_description",
            type=str,
            required=True,
            help="Short description of the package.",
        )
        self._args = parser.parse_args()
        assert self.repo_name.startswith(
            "ark_"
        ), "Repository name must start with 'ark_'"

    @property
    def repo_name(self):
        return Path(".").absolute().name

    @property
    def pkg_name(self):
        ark, name = self.repo_name.split("_")
        return ark + name

    @property
    def pkg_dir(self):
        return Path(self.pkg_name)

    @property
    def github_repo(self):
        return f"https://github.com/Robotics-Ark/{self.pkg_name}"

    @property
    def ssh(self):
        return f"git@github.com:Robotics-Ark/{self.pkg_name}.git"

    @property
    def https(self):
        return f"https://github.com/Robotics-Ark/{self.pkg_name}.git"

    def get_readme(self):
        return readme_template.format(
            repo_name=self.repo_name,
            short_description=self._args.short_description,
            ssh=self.ssh,
            https=self.https,
        )

    def get_setup_py(self):
        return setup_py.format(
            pkg_name=self.pkg_name,
            short_description=self._args.short_description,
        )


def main():
    args = Arguments()

    # Create package
    args.pkg_dir.mkdir(parents=True, exist_ok=True)
    (args.pkg_dir / "__init__.py").touch()

    # Create README.md
    with open("README.md", "w") as f:
        f.write(args.get_readme())

    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("lcm\n")

    # Create setup.py
    with open("setup.py", "w") as f:
        f.write(args.get_setup_py())

    # Finish up
    print("Completed, you can now delete gen.py")


if __name__ == "__main__":
    main()
