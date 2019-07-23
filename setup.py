import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpucbc",
    version="0.0.1",
    author="Colm Talbot",
    author_email="colm.talbot@monash.edu",
    description="GPU enabled CBC parameter estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ColmTalbot/GPUCBC",
    packages=['gpucbc'],
    package_dir={'gpucbc': '.'},
    install_requires=['bilby', 'numpy', 'lalsuite'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
