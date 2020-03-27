from os.path import dirname, join

from setuptools import find_packages, setup

# from pip.req import parse_requirements


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open(join(dirname(__file__), './VERSION.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()


setup(
    name='trtis',
    version='0.1.0',
    keywords='363246',
    description='a library for DS CAA Developer',
    license='MIT License',
    url='',
    author='layersim',
    author_email='',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=[
        "numpy>=1.16.0",
        "protobuf>=3.8.0",
        "onnx>=1.6.0",
        "pycuda>=2018.1.2",
        "tensorflow-gpu>=1.15.0"
    ],
    python_requires='>=3.6'
)
