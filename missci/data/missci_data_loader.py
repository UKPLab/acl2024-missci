from os.path import join
from typing import Optional, Dict, List

from missci.util.fileutil import read_jsonl

DEFAULT_DIRECTORY_PATH: str = './dataset'
DATASET_NAME_ALL_TEST: str = 'test.missci.jsonl'
DATASET_NAME_DEV: str = 'dev.missci.jsonl'


class MissciDataLoader:
    def __init__(self, dataset_directory: Optional[str] = None):
        self.dataset_directory: str = dataset_directory or DEFAULT_DIRECTORY_PATH

    def load_raw_arguments(self, split: str) -> List[Dict]:
        assert split in {'dev', 'test'}, f'Unknown split: "{split}"'

        file_name: str = DATASET_NAME_ALL_TEST if split == 'test' else DATASET_NAME_DEV
        return list(read_jsonl(join(self.dataset_directory, file_name)))
