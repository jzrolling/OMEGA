import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OMEGA", # Replace with your own username
    version="0.1.1",
    author="jz-rolling",
    author_email="juzhu@hsph.harvard.edu",
    description="Open source, Mycobacteria-Enhanced Graphic Analysis toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)