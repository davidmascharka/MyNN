from setuptools import setup, find_packages

import versioneer


DISTNAME = 'mynn'
DESCRIPTION = 'A pure-Python neural network library'
LICENSE = 'MIT'
AUTHOR = 'David Mascharka'
AUTHOR_EMAIL = 'davidmascharka@gmail.com'
URL = 'https://github.com/davidmascharka/MyNN'
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
]
INSTALL_REQUIRES = ['numpy >= 1.13', 'mygrad >= 1.0']
TESTS_REQUIRE = ['pytest >= 3.8', 'hypothesis >= 4.6']
LONG_DESCRIPTION = """
MyNN is a simple NumPy-centric neural network library that builds on top of MyGrad. It provides
convenient wrappers for such functionality as

- Convenient neural network layers (e.g. convolutional, dense, batch normalization, dropout)
- Weight initialization functions (e.g. Glorot, He, uniform, normal)
- Neural network activation functions (e.g. elu, glu, tanh, sigmoid)
- Common loss functions (e.g. cross-entropy, KL-divergence, Huber loss)
- Optimization algorithms (e.g. sgd, adadelta, adam, rmsprop)

MyNN comes complete with several examples to ramp you up to being a fluent user of the library.
It was written as an extension to MyGrad for rapid prototyping of neural networks with minimal dependencies,
a clean codebase with excellent documentation, and as a learning tool.
"""

if __name__ == '__main__':
    setup(
        name=DISTNAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        license=LICENSE,
        author=AUTHOR,
        classifiers=CLASSIFIERS,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_ocntent_type="text/markdown",
        url=URL,
        python_requires=">=3.6",
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,
    )
