from setuptools import setup, find_packages

setup(
    name='new_MFC',
    version='1.0.0',
    description='a lightweight microsimulation free-flow acceleration model '
                '(MFC) that is able to capture the vehicle acceleration '
                'dynamics accurately and consistently',
    url='',
    author='',
    license='JRC',
    packages=find_packages(exclude=['']),
    install_requires=['requests'],
    package_data={
        'new_MFC': ['package_data.data']
    },
    data_files=None,
    author_email='',
    classifier=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Manufacturing',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Natural Language :: English',
        'Environment :: Console',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
    ],
    zip_safe=True,
    python_requires='>=3.5',
    platforms=['any'],
)