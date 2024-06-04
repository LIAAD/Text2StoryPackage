Installation
=====

Installation of text2story requires some libraries that are not python ones. These libraries are important to
the visualization module. Next, we detail

Linux / Ubuntu
-------

The installation requires graphviz software, the latex suite and the software poppler to convert pdf to png.
In Linux, to install these software open a terminal and type the following commands:

.. code-block:: bash
    sudo apt-get install graphviz libgraphviz-dev texlive-latex-base  texlive-latex-extra poppler-utils


After that, create a virtual environment using venv or other tool of your preference. For instance,
using the following command in the prompt line:

.. code-block:: bash
    $ python3 -m venv venv

Then, activate the virtual enviroment in the prompt line. Like, the following command:

.. code-block:: bash
    $ source venv/bin/activate

After that, you are ready to install


Windows
-------

First, make sure you have Microsoft C++ Build Tools. Then install graphviz software by download one suitable version
in this [link](https://graphviz.org/download/#windows). Next, install the latex-suite like these
[tutorial](https://www.tug.org/texlive/windows.html#install) explains. Then, install Popple packed for windows,
which you download [here](https://github.com/oschwartz10612/poppler-windows).

Finnally, you can install text2story using pip. If it did not recognize the graphviz installation, then you can
use the following command for pip (tested in pip == 21.1.1).

.. code-block:: powershell
    pip install text2story  --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib\"


For newer version of pip (tested in pip == 23.1.2), you can type the following command:

.. code-block:: powershell
    pip install --use-pep517  --config-setting="--global-option=build_ext"  --config-setting="--global-option=-IC:\Program Files\Graphviz\include" --config-setting="--global-option=-LC:\Program Files\Graphviz\lib"
