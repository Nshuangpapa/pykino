from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("pykino/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(tests_require=[])

setup(
    name="pykino",
    version=__version__,
    author="Yifan Wang",
    author_email="yifanwangzippo64@outlook.com",
    url="https://github.com/Nshuangpapa/pykino",
    install_requires=requirements,
    package_data={"pykino": ["resources/*"]},
    packages=find_packages(exclude=["pykino/tests"]),
    license="MIT",
    description="pykino: use mgcv in python",
    long_description="pykino",
    keywords=["statistics", "GAM", "regression", "analysis"],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    **extra_setuptools_args
)