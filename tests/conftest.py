from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
PFILE = DATA_DIR / "RIOTSHAKE_VMP142_0002.p"


@pytest.fixture
def pfile_path():
    """Path to the VMP test P-file."""
    return PFILE


@pytest.fixture
def raw_data(pfile_path):
    """Raw P-file data (counts, not converted)."""
    from pyturb._pfile import read_pfile

    return read_pfile(pfile_path)


@pytest.fixture
def phys_data(pfile_path):
    """Converted P-file data in physical units."""
    from pyturb.pfile import load_pfile_phys

    return load_pfile_phys(pfile_path)
