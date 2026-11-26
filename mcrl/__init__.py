"""MCR-L: a CPU-first mathematical and learned-language experiment."""
__version__ = "0.9.0"

from .engine import MCRLanguageEngine
from .neural_engine import NeuralMCRLanguageEngine
from .semantic_network import LexicalSemanticNetwork, SemanticAnalysis

__all__ = [
    "MCRLanguageEngine",
    "NeuralMCRLanguageEngine",
    "LexicalSemanticNetwork",
    "SemanticAnalysis",
    "__version__",
]
