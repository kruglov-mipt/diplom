from setuptools import find_packages, setup


setup(
    name="pyons",
    version="0.9.0",
    author="Andrey Larionov",
    author_email="larioandr@gmail.com",
    platforms=["any"],
    license="MIT",
    url="https://github.com/larioandr/pyons",
    packages=find_packages(),
    entry_points='''
        [console_scripts]
        ons=pyons.main:cli
    ''',
)
