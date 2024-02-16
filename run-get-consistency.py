"""run-get-consistency.py

Usage:
  run-get-consistency.py llama <file> <prompt-template> <prefix> [--dev]
  run-get-consistency.py gpt4 <file> <prompt-template> <prefix> [--dev] [--overwrite]
  run-get-consistency.py consistency-parse <file> [--dev]

Arguments:
  <file>              Path to the input file within the "predictions/generate-classify" directory.
  <prompt-template>   Prompt template to be used, e.g. "cls_with_premise/classify-D.txt".
  <prefix>            Prefix to be used when storing the results.

Options:
  -h --help           Show this screen.
  --dev               Run on the development set
  --overwrite         Overwrite existing files.
"""


from os.path import join
from typing import List, Dict

from docopt import docopt

from missci.data.missci_data_loader import MissciDataLoader
from missci.eval.eval_consistency import eval_consistency
from missci.modeling.gpt4 import GPTCaller
from missci.modeling.model_llama import query_llama_for_fallacy_consistency
from missci.prompt_templates.consistency_template_filler import FallacyConsistencyTemplateFiller
from missci.util.directory_util import get_prediction_directory, get_raw_prompt_prediction_directory
from missci.util.fileutil import read_jsonl


def run_consistency_prompting_llama(
        file_name: str, instances: List[Dict], split: str, prompt_template: str, model_size: str, prefix: str
) -> None:
    """
    Prompt Llama2 to classify the applied fallacies in the generated fallacious premises. Predictions will be stored in
    the "predictions/consistency-raw" directory.

    :param file_name: name of the file including the predictions.
    :param instances: List of all instances that will be prompted.
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param prompt_template: relative path of the prompt template within the "prompt_templates" directory.
    :param model_size: Llama2 size as string ("7b", "70b", "13b")
    :param prefix: prefix included when naming the prediction file.
    :return:
    """
    predictions: List[Dict] = list(read_jsonl(join(
        get_prediction_directory('generate-classify'), file_name
    )))

    return query_llama_for_fallacy_consistency(
        split=split,
        instances=instances,
        predictions=predictions,
        output_directory=get_raw_prompt_prediction_directory('consistency'),
        template_file=prompt_template,
        llama_size=model_size,
        prediction_file_name=file_name,
        dest_name_prefix=prefix
    )


def run_consistency_prompting_gpt(
        file_name: str, instances: List[Dict], split: str, prompt_template: str, prefix: str, overwrite: bool
) -> str:
    """
    Prompt Llama2 to classify the applied fallacies in the generated fallacious premises. Predictions will be stored in
    the "predictions/consistency-raw" directory.

    :param file_name: name of the file including the predictions.
    :param instances: List of all instances that will be prompted.
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param prompt_template: relative path of the prompt template within the "prompt_templates" directory.
    :param prefix: prefix included when naming the prediction file.
    :param overwrite: If set to true, existing GPT 4 predictions will not be re-used.
    """
    predictions: List[Dict] = list(read_jsonl(join(
        get_prediction_directory('generate-classify'), file_name
    )))

    gpt4: GPTCaller = GPTCaller(
        output_directory=get_raw_prompt_prediction_directory('consistency'),
        template_filler=FallacyConsistencyTemplateFiller(
            prompt_template, predictions, file_name, prefix
        ),
        overwrite=overwrite
    )

    return gpt4.prompt(instances, split)


def parse_and_eval_consistency(file: str):
    """
    Parse the textual LLM response to extract the applied fallacy class. The parsed answers will be stored in the
    "predictions/consistency" directory.

    :param file: name of the consistency prediction file within the "predictions/consistency-raw" directory.
    """
    eval_consistency(file)


def main():
    args = docopt(__doc__)

    split: str = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MissciDataLoader().load_raw_arguments(split)

    if args['llama']:
        run_consistency_prompting_llama(
            args['<file>'], instances, split, args['<prompt-template>'], '70b', args['<prefix>']
        )
    elif args['gpt4']:
        run_consistency_prompting_gpt(
            args['<file>'], instances, split, args['<prompt-template>'], args['<prefix>'], args['--overwrite']
        )
    elif args['consistency-parse']:
        parse_and_eval_consistency(args['<file>'])
    else:
        raise NotImplementedError(args)


if __name__ == '__main__':
    main()
