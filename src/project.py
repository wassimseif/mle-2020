# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from pathlib import Path

# Project dependencies


# Project imports
@dataclass
class Project:
    """
    This class represents our project.
    It stores useful information about the structure
    """

    base_dir: Path = Path(__file__).parents[1]
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"
    exported_csv_dir = base_dir / 'exported_csv'

    def __post_init__(self) -> None:
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.exported_csv_dir.mkdir(exist_ok=True)
        pass


if __name__ == "__main__":
    project = Project()
