__version__ = '0.1.7'

from .kit import get_trainer
from .connectors.neptune import NeptuneConnector
from .data import Database
from .utils import show, audio, signal
