from setuptools import setup, find_packages

setup(
    name='new_MFC',
    version='1.0.0',
    description='Microsimulation free-flow accelaration model MFC',
    url='',
    author='',
    license='JRC',
    package=find_packages(exclude=['new_MFC.examples']),
    install_requires=['colored']
)