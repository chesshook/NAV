"""MCR-L: a CPU-first mathematical language-comprehension experiment."""
__version__ = "1.0.0"

from .engine import MCRLanguageEngine
from .neural_engine import NeuralMCRLanguageEngine
from .semantic_network import LexicalSemanticNetwork, SemanticAnalysis
from .mathematical_engine import MathematicalComprehensionEngine, UnderstandingResult
from .syntax_model import MathematicalSyntaxModel, ParseResult

__all__ = [
    "MCRLanguageEngine",
    "NeuralMCRLanguageEngine",
    "LexicalSemanticNetwork",
    "SemanticAnalysis",
    "MathematicalComprehensionEngine",
    "UnderstandingResult",
    "MathematicalSyntaxModel",
    "ParseResult",
    "__version__",
]
