from os.path import join
from typing import Dict, List

from sklearn.metrics import classification_report

from missci.output_parser.llm_output_parser_fallacy import ClassifyFallacyLLMOutputParser
from missci.util.directory_util import get_raw_prompt_prediction_directory, get_prediction_directory
from missci.util.fileutil import read_jsonl, write_json
from missci.util.post_processing import remove_scores


def eval_consistency(file_name: str) -> None:
    predictions: List[Dict] = list(read_jsonl(join(
        get_raw_prompt_prediction_directory('consistency'), file_name
    )))
    predictions = list(map(remove_scores, predictions))
    prompt_template_name: str = predictions[0]['params']['template']
    parser: ClassifyFallacyLLMOutputParser = ClassifyFallacyLLMOutputParser(prompt_template_name)
    predictions = list(map(parser.parse, predictions))

    # Store fallacy classifications for comparison
    originally_predicted_fallacies: List[str] = []
    re_predicted_fallacies: List[str] = []

    for prediction in predictions:
        re_predicted_cls: str = prediction['predicted_parsed']['fallacy_name']
        orig_predicted_cls: str = prediction['data']['generated']['fallacy_class']
        orig_generated_premise = prediction['data']['generated']['premise']

        originally_predicted_fallacies.append(orig_predicted_cls)
        re_predicted_fallacies.append(re_predicted_cls)

        # Make sure the fallacious premise does not directly name the fallacy!
        if re_predicted_cls.lower() in orig_generated_premise.lower():
            print(f'[{re_predicted_cls==orig_predicted_cls}] Cheating in "{orig_generated_premise}"!')
            assert False

    # Eval and store outputs
    scores = classification_report(
        originally_predicted_fallacies, re_predicted_fallacies, output_dict=True, zero_division=True
    )
    out_path: str = join(
        get_prediction_directory('consistency'), f'evaluation__{file_name}'.replace('.jsonl', '.json')
    )
    write_json(scores, out_path, pretty=True)

    print('Fallacy Matching Accuracy:', round(scores['accuracy'], 3))
