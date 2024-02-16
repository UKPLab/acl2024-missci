"""run-argument-reconstruction.py

Usage:
  run-argument-reconstruction.py llama <prompt-template> <model-size> [<seed>] [--dev] [--8bit]
  run-argument-reconstruction.py gpt4 <prompt-template> [--dev] [--overwrite]
  run-argument-reconstruction.py parse-llm-output <file> <k> [--dev]
  run-argument-reconstruction.py eval-random <premise> [<seed>] [--dev]


Options:
  -h, --help                           Show this help message and exit
  --dev                                Only run on the validation split
  --8bit                               Use 8-bit quantization. Only applies when using 70b model. Otherwise, 4-bit is
                                        used.
  --overwrite                          Do not re-use existing results (if they exist)

Arguments:
  <model-size>                         The size of the model to be used. Possible values: "7b", "13b", "70b".
  <seed>                               The random seed (default=1 for Llama 2; default=[1,2,3,4,5] for random baseline)
  <prompt-template>                    Path to the prompt template (relative to the "prompt_templates" directory).
  <file>                               Name (not path) of the file containing the raw LLM outputs.
  <premise>                            For the random baseline only: Which part of the argument should be used as
                                        fallacious premise. Possible values: "claim", "p0"

"""
import json
from os.path import join
from typing import List, Dict, Optional, Union

from docopt import docopt

from missci.data.missci_data_loader import MissciDataLoader
from missci.eval.evaluator_gen_classify import GenClassifyEvaluator
from missci.modeling.gpt4 import GPTCaller
from missci.modeling.model_llama import query_llama_for_classification_generation
from missci.modeling.random_baseline import FallacyGenerationBaseline
from missci.output_parser.llm_output_parser_generate_classify import GenerateClassifyLLMOutputParser
from missci.prompt_templates.classify_generate_template_filler import ClassifyGenerateTemplateFiller
from missci.util.directory_util import get_prediction_directory, get_raw_prompt_prediction_directory
from missci.util.fileutil import write_jsonl, read_jsonl
from missci.util.post_processing import remove_scores, context_to_argument_level_predictions, \
    prepare_context_level_fallacy_generation_for_evaluation


def eval_random_baselines(
        premise_strategy: str,
        split: str,
        instances: List[Dict],
        use_seed: Optional[int] = None) -> List[str]:
    """
    Run the random baseline on the selected instances. Each time, random fallacy class will be predicted. The generated
    fallacious premise is either the claim, or the accurate premise, depending on the selected strategy.

    :param premise_strategy: Must be either "claim" or "p0" and determines which is returned as fallacious premise.
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances used for prediction.
    :param use_seed: An optional random seed (default=[1,2,3,4,5])

    :return: A list of the prediction files (relative path within "predictions/generate-classify").
    """

    prediction_files: List[str] = []
    baseline: FallacyGenerationBaseline = FallacyGenerationBaseline(premise_strategy)

    # By default, run over five seeds.
    if use_seed is None:
        seeds = range(1, 6)
    else:
        seeds = [use_seed]

    evaluator: GenClassifyEvaluator = GenClassifyEvaluator(
        do_nli_eval=True,
        do_meteor_eval=True,
        do_bert_score_eval=True,
        split=split
    )

    for seed in seeds:
        predictions: List[Dict] = baseline.run(instances, random_seed=seed)
        file_name: str = f'baseline-{premise_strategy}-s{seed}_{split}.jsonl'
        write_jsonl(join(get_prediction_directory('generate-classify'), file_name), predictions)
        prediction_files.append(file_name)
        evaluator.evaluate_file(file_name)

    return prediction_files


def run_llama_fallacy_classification(
        split: str,
        instances: List[Dict],
        seed: int,
        prompt_template: str,
        model_size: str,
        run_8bit: bool = False,

) -> str:
    """
    Prompt Llama 2 to generate fallacious premises together with the fallacy class. The LLM output will be stored in the
    "predictions/generate-classify-raw" directory.

    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances used for prediction.
    :param seed: Random seed (default=1)
    :param prompt_template: relative path of the prompt template within the "prompt_templates" directory.
    :param model_size: Llama2 size as string ("7b", "70b", "13b")
    :param run_8bit: Use 8-bit precision. Only relevant if the model is "70b". Default is 4-bit for 70b and 8-bit
                        otherwise

    :return name of the prediction file within "predictions/generate-classify-raw"
    """

    output_directory: str = get_raw_prompt_prediction_directory('generate-classify')

    return query_llama_for_classification_generation(
        split=split,
        instances=instances,
        output_directory=output_directory,
        template_file=prompt_template,
        llama_size=model_size,
        seed=seed,
        temperature=None,
        run_8bit=run_8bit
    )


def run_gpt4_fallacy_classification_generation(
        prompt_template: str, split: str, instances: List[Dict], overwrite: bool
) -> str:
    """
    Prompt GPT 2 to generate fallacious premises together with the fallacy class. The LLM output will be stored in the
    "predictions/generate-classify-raw" directory.

    :param prompt_template: relative path of the prompt template within the "prompt_templates" directory.
    :param split: Data split ("train" or "dev"). Only used for naming.
    :param instances: List of all instances used for prediction.
    :param overwrite: If set to true, existing GPT 4 predictions will not be re-used.

    :return: name of the prediction file within "predictions/generate-classify-raw"
    """
    gpt4: GPTCaller = GPTCaller(
        output_directory=get_raw_prompt_prediction_directory('generate-classify'),
        template_filler=ClassifyGenerateTemplateFiller(prompt_template),
        overwrite=overwrite
    )

    return gpt4.prompt(instances, split)


def parse_and_eval_llm_output(file_name: str, k: Union[int, str], gold_instances: List[Dict], split: str) -> str:
    """
    Parse an LLM output file and evaluate the reconstructed arguments (fallacies and generated premises). The parsed
    output and the evaluation will be stored in "predictions/generate-classify".

    :param file_name: name of the file including the predictions.
    :param k: evaluate based on the top "k" predictions. k must be either a number or "all".
    :param gold_instances: Gold instances for which to expect predictions.
    :param split: Split which must be evaluated.

    :return: filename of the evaluation.
    """

    if k != 'all':
        k = int(k)

    # Load and clean predictions
    predictions: List[Dict] = list(
        read_jsonl(join(get_raw_prompt_prediction_directory('generate-classify'), file_name))
    )
    predictions = list(map(remove_scores, predictions))

    # remember the experiment parameters
    prompt_template_name: str = predictions[0]['params']['template']
    params: Dict = predictions[0]['params']

    # parse LLM outputs
    prompt_parser: GenerateClassifyLLMOutputParser = GenerateClassifyLLMOutputParser(prompt_template_name)
    predictions = list(map(prompt_parser.parse, predictions))

    for prediction in predictions:
        prediction['params'] = params
        prediction['params']['k'] = k

    # Convert context-level prompt outputs to argument-level predictions
    argument_level_predictions: List[Dict[str, Dict]] = context_to_argument_level_predictions(predictions)
    argument_level_predictions: List[Dict[str, Dict]] = list(prepare_context_level_fallacy_generation_for_evaluation(
        argument_level_predictions, gold_instances, k
    ))

    file_name = file_name.replace('.jsonl', f'k-{k}.jsonl')
    dest_file_path: str = join(get_prediction_directory('generate-classify'), file_name)

    # Write and eval
    write_jsonl(dest_file_path, argument_level_predictions)
    evaluator: GenClassifyEvaluator = GenClassifyEvaluator(
        do_nli_eval=True,
        do_meteor_eval=True,
        do_bert_score_eval=True,
        split=split
    )

    scores: Dict = evaluator.evaluate_file(file_name)
    print(json.dumps(scores, indent=2))

    return file_name


def main():
    args = docopt(__doc__)

    split: str = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MissciDataLoader().load_raw_arguments(split)

    if args['eval-random']:
        eval_random_baselines(args['<premise>'], split, instances, args['<seed>'])
    elif args['llama']:
        seed: int = 1
        if args['<seed>']:
            seed = int(args['<seed>'])
        run_llama_fallacy_classification(
            split, instances, seed, args['<prompt-template>'], args['<model-size>'], args['--8bit']
        )
    elif args['gpt4']:
        run_gpt4_fallacy_classification_generation(args['<prompt-template>'], split, instances, args['--overwrite'])
    elif args['parse-llm-output']:
        parse_and_eval_llm_output(args['<file>'], args['<k>'], instances, split)
    else:
        raise NotImplementedError(args)


if __name__ == '__main__':
    main()
