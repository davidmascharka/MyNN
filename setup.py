from setuptools import setup, find_packages

def do_setup():
    setup(name='MyNN',
          version='0.1',
          author='David Mascharka',
          description='A pure-Python neural network library',
          license='MIT',
          platforms=['Linux', 'Unix', 'Windows', 'Mac OS-X'],
          packages=find_packages(),
          install_requires=['numpy>=1.11', 'MyGrad>=0.0'])

if __name__ == '__main__':
    do_setup()
