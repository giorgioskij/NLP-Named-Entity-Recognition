"""
Holds globals configuration variables, such as paths
"""

from pathlib import Path
import torch

DEVICE: torch.device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

ROOT: Path = Path(__file__).parent.parent
DATA: Path = ROOT.parent / Path('data')
MODEL: Path = ROOT.parent / Path('model')

TRAIN: Path = DATA / Path('train.tsv')
DEV: Path = DATA / Path('dev.tsv')

SEED: int = 42