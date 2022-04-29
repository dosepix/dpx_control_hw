import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='dpx_energy_calibration',
    version='0.1',
    description='Energy calibration for the single-DPX readout hardware written in Python3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sebastian Schmidt',
    author_email='schm.seb@gmail.com',
    url="https://github.com/dosepix/dpx_control_hw/dpx_energy_calibration",
    project_urls={
        "Bug Tracker": "https://github.com/dosepix/dpx_control_hw/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license='GNU GPLv3',
    packages=['dpx_energy_calibration'],
    install_requires=[
        'dpx_control_hw',
        'tensorflow',
    ]
)
