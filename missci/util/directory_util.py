from os import makedirs
from os.path import exists
from typing import Optional


DEFAULT_PRED_DIRECTORY_GENERATE_CLASSIFY: str = 'predictions/generate-classify'
DEFAULT_PRED_DIRECTORY_GENERATE_CLASSIFY_RAW: str = 'predictions/generate-classify-raw'

DEFAULT_PRED_DIRECTORY_CONSISTENCY: str = 'predictions/consistency'
DEFAULT_PRED_DIRECTORY_CONSISTENCY_RAW: str = 'predictions/consistency-raw'

DEFAULT_PRED_DIRECTORY_CLASSIFY_GIVEN_GOLD_PREMISE: str = 'predictions/classify-given-gold-premise'
DEFAULT_PRED_DIRECTORY_CLASSIFY_GIVEN_GOLD_PREMISE_RAW: str = 'predictions/classify-given-gold-premise-raw'

DEFAULT_PRED_DIRECTORY_ONLY_CLASSIFY: str = 'predictions/only-classify'
DEFAULT_PRED_DIRECTORY_ONLY_CLASSIFY_RAW: str = 'predictions/only-classify-raw'


def get_prediction_directory(subtask: str, assigned_directory: Optional[str] = None) -> str:
    if assigned_directory is not None:
        directory: str = assigned_directory
    else:
        if subtask == 'generate-classify':
            directory: str = DEFAULT_PRED_DIRECTORY_GENERATE_CLASSIFY
        elif subtask == 'consistency':
            directory: str = DEFAULT_PRED_DIRECTORY_CONSISTENCY
        elif subtask == 'gold-premise':
            directory: str = DEFAULT_PRED_DIRECTORY_CLASSIFY_GIVEN_GOLD_PREMISE
        elif subtask == 'classify-only':
            directory: str = DEFAULT_PRED_DIRECTORY_ONLY_CLASSIFY
        else:
            raise ValueError(subtask)

    if not exists(directory):
        makedirs(directory)
    return directory


def get_raw_prompt_prediction_directory(subtask: str, assigned_directory: Optional[str] = None) -> str:
    if assigned_directory is not None:
        return assigned_directory
    else:
        if subtask == 'generate-classify':
            directory: str = DEFAULT_PRED_DIRECTORY_GENERATE_CLASSIFY_RAW
        elif subtask == 'consistency':
            directory: str = DEFAULT_PRED_DIRECTORY_CONSISTENCY_RAW
        elif subtask == 'gold-premise':
            directory: str = DEFAULT_PRED_DIRECTORY_CLASSIFY_GIVEN_GOLD_PREMISE_RAW
        elif subtask == 'classify-only':
            directory: str = DEFAULT_PRED_DIRECTORY_ONLY_CLASSIFY_RAW
        else:
            raise ValueError(subtask)

    if not exists(directory):
        makedirs(directory)
    return directory
