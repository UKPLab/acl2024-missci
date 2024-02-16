"""run-fallacy-classification-without-premise

Usage:
  run-fallacy-classification-without-premise.py llama <prompt-template> <model-size> [<seed>] [--dev] [--8bit]
  run-fallacy-classification-without-premise.py gpt4  <prompt-template> [--dev] [--overwrite]
  run-fallacy-classification-without-premise.py parse-llm-output <file> [--dev]

Arguments:
  <prompt-template>   Prompt template to be used, e.g. "cls_without_premise/p4-connect-cls-D.txt".
  <model-size>        Model size to be used ("70b", "13b", "7b").
  <seed>              Seed value (default=1).

Options:
  -h, --help          Show this help message and exit
  --dev               Run on the development set
  --8bit              Use 8-bit precision. Only relevant if the model is "70b". Default is 4-bit for 70b and 8-bit
                        otherwise
  --overwrite         Overwrite existing files.
"""
from os.path import join
from typing import List, Dict

from docopt import docopt
from sklearn.metrics import classification_report

from missci.data.missci_data_loader import MissciDataLoader
from missci.modeling.gpt4 import GPTCaller
from missci.modeling.model_llama import query_llama_for_classification_with_implicit_premise
from missci.output_parser.llm_output_parser_fallacy import ClassifyFallacyLLMOutputParser
from missci.prompt_templates.classify_generate_template_filler import ClassifyGenerateTemplateFiller
from missci.util.directory_util import get_raw_prompt_prediction_directory, get_prediction_directory
from missci.util.fileutil import read_jsonl, write_jsonl


def run_llama_fallacy_classification_without_premise(
        template_file: str, llama_size: str, split: str, instances: List[Dict], seed: int
) -> str:
    """
    Prompt Llama2 to classify the applied fallacy only given the claim, the accurate premise and the context (but not
    the fallacious premise). The LLM outputs will be stored in the  "predictions/only-classify-raw" directory.

    :param template_file:  relative path of the prompt template within the "prompt_templates" directory.
    :param llama_size: Llama2 size as string ("7b", "70b", "13b")
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances that will be prompted.
    :param seed: random seed (default=1)
    :return:
    """
    output_directory: str = get_raw_prompt_prediction_directory('classify-only')

    return query_llama_for_classification_with_implicit_premise(
        split=split,
        instances=instances,
        output_directory=output_directory,
        template_file=template_file,
        llama_size=llama_size,
        seed=seed
    )


def run_gpt4_fallacy_classification(template_file: str, split: str, instances: List[Dict], overwrite: bool) -> str:
    """
    Prompt GPT 4 to classify the applied fallacy only given the claim, the accurate premise and the context (but not
    the fallacious premise). The LLM outputs will be stored in the  "predictions/only-classify-raw" directory.

    :param template_file: relative path of the prompt template within the "prompt_templates" directory.
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances that will be prompted.
    :param overwrite: If set to true, existing GPT 4 predictions will not be re-used.
    :return:
    """

    template_filler = ClassifyGenerateTemplateFiller(template_file)
    gpt4: GPTCaller = GPTCaller(
        output_directory=get_raw_prompt_prediction_directory('classify-only'),
        template_filler=template_filler,
        overwrite=overwrite
    )

    return gpt4.prompt(instances, split)


def parse_prompt_llm_output(file_name: str, gold_instances: List[Dict], formatted=False):
    gold_instance_dict = {
        fallacy['id']: [
            interchangeable_fallacy['class'] for interchangeable_fallacy in fallacy['interchangeable_fallacies']
        ]
        for instance in gold_instances
        for fallacy in instance['argument']['fallacies']
    }

    prompt_directory: str = get_raw_prompt_prediction_directory('classify-only')
    predictions = list(read_jsonl(join(prompt_directory, file_name)))
    assert len(predictions) == len(gold_instance_dict), f'MISMATCH {len(predictions)} vs {len(gold_instance_dict)}: {file_name}'

    all_gold = []
    all_pred = []
    for pred in predictions:
        if formatted:
            predicted = ClassifyFallacyLLMOutputParser('').parse(pred)['predicted_parsed']['fallacy_name']
        else:
            assert False
            predicted = get_single_fallacy_from_answer(pred['answer'])
        gold_labels = gold_instance_dict[pred['data']['fallacy_id']]

        # It is correct if it is among the interchangeable fallacies!
        pred['predicted-label'] = predicted
        pred['gold-labels'] = gold_labels

        if predicted in gold_labels:
            gold_label = predicted
        else:
            gold_label = gold_labels[0]

        all_gold.append(gold_label)
        all_pred.append(predicted)

    write_jsonl(
        join(get_prediction_directory('classify-only'), file_name.replace('.jsonl', '.parsed.jsonl')),
        predictions
     )

    report = classification_report(all_gold, all_pred, zero_division=0, output_dict=True)

    acc = round(report['accuracy'], 3)
    f1 = round(report['macro avg']['f1-score'], 3)

    print(file_name, f'\nACC: {acc} ; F1 MACRO: {f1}')
    print()


def main():
    args = docopt(__doc__)

    split = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MissciDataLoader().load_raw_arguments(split)

    if args['llama']:
        seed: int = 1
        if args['<seed>']:
            seed = int(args['<seed>'])

        run_llama_fallacy_classification_without_premise(
            args['<prompt-template>'], args['<model-size>'], split, instances, seed
        )
    elif args['gpt4']:
        run_gpt4_fallacy_classification(args['<prompt-template>'], split, instances, args['--overwrite'])
    elif args['parse-llm-output']:
        parse_prompt_llm_output(args['<file>'], instances, formatted=True)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
