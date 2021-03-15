import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpucbc",
    version="0.1.0",
    author="Colm Talbot",
    author_email="colm.talbot@monash.edu",
    description="GPU enabled CBC parameter estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ColmTalbot/GPUCBC",
    packages=["gpucbc"],
    package_dir={"gpucbc": "gpucbc"},
    install_requires=["numpy>=1.16", "astropy", "bilby<1.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
