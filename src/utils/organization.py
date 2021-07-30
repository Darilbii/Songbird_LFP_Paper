""" Utility Functions for Keeping Songbird-LFP-Paper organized and portable"""
from pathlib import Path


# BASE_PATH = Path('Songbird-LFP-Paper/reports')

# Todo: Add a safe parameter to ensure that the base_path exists first

def create_folder(base_path: Path, directory: str, rtn_path=False):
    """ Recursive directory creation function. Like mkdir(), but makes all intermediate-level directories needed to
    contain the leaf directory

    Parameters
    -----------
    base_path : pathlib.PosixPath
        Global Path to be root of the created directory(s)
    directory : str
        Location in the Songbird-LFP-Paper the new directory is meant to be made
    rtn_path : bool, optional
        If True it returns a Path() object of the path to the Directory requested to be created

    Returns
    --------
    location_to_save : class, (Path() from pathlib)
        Path() object for the Directory requested to be created

    Example
    --------
    # Will typically input a path using the Global Paths from paths.py
    >>> create_folder('/data/')

    """

    location_to_save = base_path / directory

    # Recursive directory creation function
    location_to_save.mkdir(parents=True, exist_ok=True)

    if rtn_path:
            return location_to_save.resolve()


# TODO: Utility function for handling labeling of Figures to Keep organization consistent


