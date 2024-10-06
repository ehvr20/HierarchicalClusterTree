from setuptools import setup, find_packages

setup(
    name="HierarchicalClusterTree",  # Replace with your package name
    version="0.1.0",
    description="A Python package for handling hierarchical clustering and pruning trees",
    author="E. H. von Rein",  # Replace with your name
    url="https://github.com/ehvr20/HierarchicalClusterTree",  # Optional, replace with your GitHub repo or project URL
    packages=find_packages(where="src"),  # Assuming your code is inside a 'src' directory
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scanpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust the license if different
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",  # Specify the minimum Python version
)
