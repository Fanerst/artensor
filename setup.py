from setuptools import Extension, setup

setup(
    name='ArTensor',
    version='0.1.0',
    author='Feng Pan',
    author_email='fan_physics@126.com',
    packages=['artensor'],# , 'artensor.tests'],
    ext_modules=[
        Extension(
            'artensor._order_core',
            sources=['artensor/_order_core.cpp'],
            language='c++',
            extra_compile_args=['-O3', '-std=c++17'],
        ),
    ],
    # scripts=['bin/script1','bin/script2'],
    url='https://github.com/Fanerst/artensor',
    license='LICENSE',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
    ],
)
