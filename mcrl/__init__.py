"""MCR-L: a CPU-first, learned-language reasoning experiment."""
__version__ = "0.8.0"

from .engine import MCRLanguageEngine
from .neural_engine import NeuralMCRLanguageEngine

__all__ = ["MCRLanguageEngine", "NeuralMCRLanguageEngine", "__version__"]
