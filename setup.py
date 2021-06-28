from setuptools import find_packages, setup

setup(
    name="presearch",
    version="0.0.1",
    author="Minjun Jeon, Byeongcheol JO, Yechan Hong",
    author_email="jmj@pharmcadd.com",
    description="pharmcadd research about trRosetta.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Protein structure prediction",
    license="MIT",
    url="",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires =[
        "numpy==1.19.2",
        "tensorflow==2.5",
        "pandas",
        "matplotlib",
        "tqdm",
        "bs4",
        "scipy",
        "gin-config",
        "tensorflow_addons",
        "paramiko",
        "scp",
        "sklearn",
        "biopython",
        "tqdm",
    ],
)
