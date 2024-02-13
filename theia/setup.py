"""install theia as a library."""
from setuptools import setup, find_packages  # type: ignore

setup(
    name="theia",
    version="0.1",
    author="udeshmukh",
    author_email="d_utkarsh@yahoo.co.in",
    description="theia - scalable and extensible network training for computer vision tasks",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
