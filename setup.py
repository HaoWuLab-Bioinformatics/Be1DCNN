# Author: Bing Zhou
"""
Setup script.

This is a free software under GPLv3. Therefore, you can modify, redistribute
or even mix it with other GPL-compatible codes. See the file LICENSE
included with the distribution for more details.

"""
import os, sys, bi1dcnn, glob
import setuptools

if (sys.version_info.major!=3) or (sys.version_info.minor<6):
    print('PYTHON 3.5+ IS REQUIRED. YOU ARE CURRENTLY USING PYTHON {}'.format(sys.version.split()[0]))
    sys.exit(2)

# Guarantee Unix Format
for src in glob.glob('scripts/*'):
    text = open(src, 'r').read().replace('\r\n', '\n')
    open(src, 'w').write(text)

setuptools.setup(
    name = 'bi1dcnn',
    author = bi1dcnn.__author__,
    author_email = '1102945913@qq.com',
    url = 'https://gitee.com/BingZhou617/bi1dcnn/',
    description = 'Neural network model for chromatin loop prediction based on bagging integrated learning',
    keywords = 'bagging 1dcnn',
    packages = setuptools.find_packages(),
    scripts = glob.glob('scripts/*'),
    long_description = 'test description',
    classifiers = [
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ]
    )

