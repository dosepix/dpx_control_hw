# Dosepix Control Software for Single Slot Hardware

Module name: dpx\_control\_hw
Author: Sebastian Schmidt  
E-Mail: schm.seb@gmail.com  

## Installation

There are multiple ways to install the module. The easiest one is to use a virtual environment. More experiened users might consider to install the module directly. Please refer to the instructions below and ensure that python3 is used.

### Virtual Environment Installation

First,  a directory for the virtual environment has to be created. To provide an example, it is called `dpx_venv` in the following.  Afterwards, the environment is created via

```bash
python3 -m venv dpx_venv
```

Activate the virtual environment by executing

```bash
source dpx_virtenv/bin/activate
```

If everything worked correctly, the name of your virtual environment should appear in parentheses in front of your command prompt. Finally, proceed like described in the "Direct Installation"-section below.

### Direct Installation

`sudo` might be required to provide installation privileges. This won't be necessary when installing in an virtual environment.  

#### via pip

If no administrator access is possible, add the parameter `--user` right behind `install`.

```bash
python3 -m pip install /path/to/package
```

If you want to modify the code later on, use  

```bash
python3 -m pip install -e /path/to/package
```

instead.

##### via `setup.py`

Execute in the module's main directory:

```bash
python3 setup.py install
```

If you want to modify the code later on, use  

```bash
python3 setup.py develop
```

instead.

## Examples

### Dosepix initialization

First, import the module.

```python
import dpx_control_hw
```

The connection to the Dosepix readout hardware is established via:

```python
dpx = dpx_control.Dosepix(port_name, config_fn=None)
```

This creates an object `dpx` of class `Dosepix`.  
Important parameters are:  

| Parameter | Function |
| :-------- | :------- |
| `port_name`           | Name of the used com-port of the PC. For Linux, it usually is `/dev/ttyUSB0`. For Windows, the port name has the form of 'COMX'. |
| `config_fn`           | Configuration file containing important parameters of the used Dosepix detectors. |

A measurement is started by using the `dpx` object. For example a ToT-measurement:

```python
dpx.measure_tot()
```

See documentation for more info.
