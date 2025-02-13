from setuptools import setup, find_packages


setup(
    name='ala',
    version='1.1.dev0',
    license='MIT',
    description='Automated Language Affiliation',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    url='https://github.com/lexibank/ala',
    keywords='phylogeny',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    python_requires='>=3.9',
    install_requires=["lingpy", "scipy", "torch"],
    extras_require={
        'dev': ['wheel', 'twine'],
        'test': [
            'pytest>=4.3',
            'pytest-cov',
            'coverage>=4.2',
        ],
    },
)
