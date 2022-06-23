## Installation
This code requires Python>=3.7 (this a requirement of pytorch video). If you are using Anaconda, you can create a clean virtual environment with the required Python version with the following command:

`conda create -n ego4d_forecasting python=3.7`

To proceed with the installation, you should then activate the virtual environment with the following command:

`conda activate ego4d_forecasting`

Run the following commands to install the requirements:

`cat requirements.txt | xargs -n 1 -L 1 pip install`

In order to make the `ego4d` module loadable, you should add the current directory to the Python path:

`export PYTHONPATH=$PWD:$PYTHONPATH`

Please note that the command above is not persistent and hence you should run it every time you open a new shell.

