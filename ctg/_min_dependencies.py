"""All minimum dependencies for ctg."""

import argparse

# The values are (version_spec, comma separated tags)
dependent_packages = {
    "sentence-transformers": ("3.3.1", "install"),
    "transformers": ("4.46.3", "install"),
    "torch": ("2.5.1", "install"),
    "tqdm": ("4.67.0", "install"),
    "pandas": ("2.2.3", "install"),
    "numpy": ("2.0.2", "install"),
    "matplotlib": ("3.9.2", "install"),
    "scikit-learn": ("1.5.2", "install"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests", "all"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))
    tag_to_packages["all"].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)