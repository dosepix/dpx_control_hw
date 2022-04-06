import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='dpx_control_hw',
    version='0.1',
    description='Control software for the single-DPX readout hardware written in Python3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sebastian Schmidt',
    author_email='schm.seb@gmail.com',
    url="https://github.com/dosepix/dpx_control_hw",
    project_urls={
        "Bug Tracker": "https://github.com/dosepix/dpx_control_hw/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license='GNU GPLv3',
    packages=['dpx_control_hw'],
    entry_points={
        'console_scripts' : [
            'dpx_control_hw = dpx_control_hw.__init__:main',
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'pyserial',
        'jupyterlab',
        'jupyter-dash',
    ]
)
