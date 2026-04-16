"""Classical-CV pipeline (HOG + SVM) for the granny-square scheme parser.

A legacy baseline kept alongside the YOLO OBB pipeline for comparison
and for generating labelled crops that feed the ML models.
"""

from .classifier import SymbolClassifier
from .dataset import dataset_summary, load_labeled_dataset, split_dataset
from .extract_segments import run as extract_segments
from .labeler import DEFAULT_SYMBOL_KEYS, Labeler
from .scheme_parser import ParserConfig, SchemeParser, Symbol

__all__ = [
    "SymbolClassifier",
    "SchemeParser",
    "ParserConfig",
    "Symbol",
    "Labeler",
    "DEFAULT_SYMBOL_KEYS",
    "load_labeled_dataset",
    "dataset_summary",
    "split_dataset",
    "extract_segments",
]
