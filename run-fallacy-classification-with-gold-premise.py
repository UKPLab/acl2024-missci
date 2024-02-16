"""run-fallacy-classification-with-gold-premise

Usage:
  run-fallacy-classification-with-gold-premise.py llama <prompt-template> <model-size> [<seed>] [--dev] [--8bit]
  run-fallacy-classification-with-gold-premise.py gpt4  <prompt-template> [--dev] [--overwrite]
  run-fallacy-classification-with-gold-premise.py parse-llm-output <file> [--dev]

Arguments:
  <prompt-template>   Prompt template to be used, e.g. "cls_with_premise/classify-D.txt".
  <model-size>        Model size to be used ("70b", "13b", "7b").
  <seed>              Seed value (default=1).

Options:
  -h, --help          Show this help message and exit
  --dev               Enable development mode.
  --8bit              Use 8-bit precision. Only relevant if the model is "70b". Default is 4-bit for 70b and 8-bit
                        otherwise
  --overwrite         Overwrite existing files.
"""


from os.path import join
from typing import List, Dict, Optional, Set

from docopt import docopt
from sklearn.metrics import classification_report

from missci.data.missci_data_loader import MissciDataLoader
from missci.modeling.gpt4 import GPTCaller
from missci.modeling.model_llama import query_llama_for_fallacy_classification_with_gold_premise
from missci.output_parser.llm_output_parser_fallacy import ClassifyFallacyLLMOutputParser
from missci.prompt_templates.fallacy_classify_template_filler import FallacyWiseTemplateFiller
from missci.util.directory_util import get_raw_prompt_prediction_directory, get_prediction_directory
from missci.util.fallacy_util import get_valid_fallacy_names
from missci.util.fileutil import read_jsonl, write_jsonl, write_json
from missci.util.post_processing import remove_scores


def run_llama_fallacy_classification_with_gold_premise(
        prompt_template: str, model_size: str, split: str, instances: List[Dict], seed: int, use8bit: bool
) -> str:
    """
    Prompt Llama2 to classify the applied fallacy given the claim, the accurate premise, the context and a fallacious
    premise. The LLM outputs will be stored in the  "predictions/classify-given-gold-premise-raw" directory.

    :param prompt_template: relative path of the prompt template within the "prompt_templates" directory.
    :param model_size:  Llama2 size as string ("7b", "70b", "13b")
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances that will be prompted.
    :param seed: Random seed (default=1)
    :param use8bit:
    :return: file name of the prediction file.
    """
    return query_llama_for_fallacy_classification_with_gold_premise(
        split=split,
        instances=instances,
        output_directory=get_raw_prompt_prediction_directory('gold-premise'),
        template_file=prompt_template,
        llama_size=model_size,
        seed=seed,
        use8bit=use8bit
    )


def run_gpt4_fallacy_classification(prompt_template: str, split: str, instances: List[Dict], overwrite: bool) -> str:
    """
    Prompt GPT 4 to classify the applied fallacy given the claim, the accurate premise, the context and a fallacious premise.
    The LLM outputs will be stored in the  "predictions/classify-given-gold-premise-raw" directory.

    :param prompt_template: relative path of the prompt template within the "prompt_templates" directory.
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances that will be prompted.
    :param overwrite: If set to true, existing GPT 4 predictions will not be re-used.
    :return: file name of the prediction file.
    """
    gpt4: GPTCaller = GPTCaller(
        output_directory=get_raw_prompt_prediction_directory('gold-premise'),
        template_filler=FallacyWiseTemplateFiller(prompt_template),
        overwrite=overwrite
    )

    return gpt4.prompt(instances, split)


def parse_llm_output(file_name: str) -> str:

    predictions: List[Dict] = list(read_jsonl(join(get_raw_prompt_prediction_directory('gold-premise'), file_name)))
    predictions = list(map(remove_scores, predictions))
    prompt_template_name: str = predictions[0]['params']['template']
    parser: ClassifyFallacyLLMOutputParser = ClassifyFallacyLLMOutputParser(prompt_template_name)
    predictions = list(map(parser.parse, predictions))
    dest_path: str = join(
        get_prediction_directory('gold-premise'), file_name.replace('.jsonl', '.parsed.jsonl')
    )
    write_jsonl(dest_path, predictions)
    return dest_path


def main():
    args = docopt(__doc__)

    split = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MissciDataLoader().load_raw_arguments(split)

    prediction_file: Optional[str] = None

    if args['llama']:
        seed: int = 1
        if args['<seed>']:
            seed = int(args['<seed>'])

        run_llama_fallacy_classification_with_gold_premise(
            args['<prompt-template>'], args['<model-size>'], split, instances, seed, args['--8bit']
        )
    elif args['gpt4']:
        run_gpt4_fallacy_classification(args['<prompt-template>'], split, instances, args['--overwrite'])
    elif args['parse-llm-output']:
        prediction_file = parse_llm_output(args['<file>'])
    else:
        raise NotImplementedError()

    if prediction_file is not None:
        predictions: List[Dict] = list(read_jsonl(prediction_file))
        gold: List[str] = list(map(lambda x: x['data']['gold_fallacy_class'], predictions))
        predicted: List[str] = list(map(lambda x: x['predicted_parsed']['fallacy_name'], predictions))

        classes_only_in_predicted: Set[str] = set(predicted) - set(get_valid_fallacy_names())
        if len(classes_only_in_predicted) > 0:
            raise ValueError(classes_only_in_predicted)

        predicted = list(map(lambda x: x if x in gold else 'unk', predicted))
        print(classification_report(gold, predicted, digits=3, zero_division=0))

        dest_path: str = prediction_file.replace('.jsonl', '.metrics.json')
        write_json(classification_report(gold, predicted, zero_division=0, output_dict=True), dest_path, pretty=True)


if __name__ == '__main__':
    main()
