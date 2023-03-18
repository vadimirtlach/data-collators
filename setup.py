from setuptools import setup, find_packages


setup(
    name="data_collators", 
    version="1.0.0", 
    package_dir={"": "src"},
    packages=find_packages("src"),
)