from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'MicroVault'
LONG_DESCRIPTION = 'MicroVault - RL for Navigation'

setup(
        name="microvault", 
        version=VERSION,
        author="Alangrotti",
        author_email="grottimeireles@email.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)