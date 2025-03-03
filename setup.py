from setuptools import setup, find_packages

setup(
    name="networking-evaluation",
    version="0.1.0",
    description="Comprehensive networking skills evaluation framework",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "openpyxl>=3.1.0",
        "fpdf2>=2.7.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.8",
) 