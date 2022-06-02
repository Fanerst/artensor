from setuptools import setup

setup(
    name='ArTensor',
    version='0.1.0',
    author='Feng Pan',
    author_email='fan_physics@126.com',
    packages=['artensor'],# , 'artensor.tests'],
    # scripts=['bin/script1','bin/script2'],
    url='https://github.com/Fanerst/artensor',
    license='LICENSE',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
    ],
)