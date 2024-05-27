from setuptools import find_packages, setup


# python setup.py install
setup(name='pywinEA2',
      version='0.0.1',
      license='MIT',
      description='Package with implementations of genetic algorithms.',
      author='Fernando García Gutiérrez',
      author_email='fegarc05@ucm.es',
      packages=find_packages(),
      include_package_data=False,
      classifiers=[
          'Development Status :: 4 - Beta',
      ],
)
