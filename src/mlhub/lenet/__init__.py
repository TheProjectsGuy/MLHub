# LeNet-5 trained on MNIST dataset
"""
    The detailed information page is at :doc:`/models/lenet`. This is
    the API documentation and contains the following
    
    -   :py:mod:`mlhub.lenet.export` for external use
    -   :py:mod:`mlhub.lenet.models` for LeNet-5 model
    -   :py:mod:`mlhub.lenet.utils` for utility functions
    -   :py:mod:`mlhub.lenet.data` for dataset pipelines
    -   :py:mod:`mlhub.lenet.train` for training process
    -   :py:mod:`mlhub.lenet.test` for testing the trained model
    
    It closely follows section 2 of the :ref:`LeCun1998 <lecun1998gradient>`
    paper.
    
    .. contents:: Table of contents
    
    A demo for external use can be this
    
    .. literalinclude:: /../test/lenet_test.py
        :language: python
        :linenos:
        :emphasize-lines: 4-5,12-16
    
    .. automodule:: mlhub.lenet.export
        :exclude-members:
    
    .. automodule:: mlhub.lenet.models
        :exclude-members:
    
    Utilities
    ^^^^^^^^^^
    
    .. automodule:: mlhub.lenet.utils
        :members:
    
    Data
    ^^^^^
    
    .. automodule:: mlhub.lenet.data
        :exclude-members:
    
    Training
    ^^^^^^^^^
    
    .. automodule:: mlhub.lenet.train
    
    Testing
    ^^^^^^^^
    
    .. automodule:: mlhub.lenet.test
"""

# Initialize modules in the sequence
from . import utils
from . import models
from . import data
from . import train
from . import test
from . import export

# Functionality to expose (for use outside this sub-module)
from .data import MNISTDataset
from .models import LeNet5
from .export import download_trained_model
from .utils import model_output_to_labels
