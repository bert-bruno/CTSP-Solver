from pathlib import PosixPath, Path
from src.structure.constants import Structure

class ProjectPathsStructure:

    class Folders:
        ROOT: PosixPath = Path(__file__).resolve().parents[2]
        DATA: PosixPath = ROOT / Structure.Folders.DATA
        DOCS: PosixPath = ROOT / Structure.Folders.DOCS
        # ---
        CTSP_INSTANCES: PosixPath = DATA / Structure.Folders.CTSP_INSTANCES