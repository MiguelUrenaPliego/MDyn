from setuptools import setup, find_packages

setup(
    name="MDyn",
    version="0.1.0",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "matplotlib>=3.0.0",
        "scipy>=1.0.0",
        "imageio>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
