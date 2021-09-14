from setuptools import setup, find_packages
import sys, os.path

# Don't import porise module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'porise'))
from version import VERSION

setup(name='porise',
      version=VERSION,
      description='Porise: A framework for developing and comparing your personalized online recommendation systems.',
      url='https://git.rakuten-it.com/users/xiaolan.a.jiang/repos/porise/browse',
      author='Xiaolan Jiang',
      author_email='xiaolan.a.jiang@rakuten.com',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('porise')],
      zip_safe=False,
      install_requires=[
          'scipy', 'numpy>=1.10.4', 'pandas', 'torch', 'scikit-learn'
      ],
    #   extras_require=extras,
    #   tests_require=['pytest', 'mock'],
      python_requires='>=3.6',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
)