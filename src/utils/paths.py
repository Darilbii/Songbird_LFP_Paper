"""Path Definitions"""
#TODO: Incoroporate this level of abstraction for paths to the entire project in a useful way (This module isn't used yet)

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Reports Paths
REPORTS_DIR = PROJECT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'


BASE_PATH = Path('Songbird_LFP_Paper/reports')
