from pathlib import Path


def project_root() -> Path:
    """Return the root directory of the project
    """
    return Path(__file__).parent.parent.parent


def output_dir() -> Path:
    """Return the directory where output files should be saved
    """
    return project_root() / "output"


def processed_data_dir() -> Path:
    """Return the directory where processed data should be saved
    """
    return project_root() / "data" / "preprocessed"