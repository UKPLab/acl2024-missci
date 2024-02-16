from collections import defaultdict
from typing import Dict, Optional, List, Iterable, Union

from missci.util.fallacy_util import get_valid_fallacy_names


def remove_scores(prediction: Dict) -> Dict:
    if 'transition_scores' in prediction:
        prediction.pop('transition_scores', None)
    if 'log_probabilities' in prediction:
        prediction.pop('log_probabilities', None)
    return prediction


def normalize_fallacious_premise(premise: Optional[str]) -> Optional[str]:
    if premise is None:
        return None
    else:
        return premise.replace('"', '').strip()


def context_to_argument_level_predictions(predictions: List[Dict]) -> List[Dict[str, Dict]]:
    argument_to_fallacy_id: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
    for prediction in predictions:
        argument_id: str = prediction['data']['argument']
        fallacy_id: str = prediction['data']['fallacy_id']
        argument_to_fallacy_id[argument_id][fallacy_id] = prediction['predicted_parsed']

    arg_level_predictions = [
        {
            'argument': argument_id,
            'predictions': argument_to_fallacy_id[argument_id],
            'params': predictions[0]['params']
        } for argument_id in argument_to_fallacy_id
    ]
    return arg_level_predictions


def prepare_context_level_fallacy_generation_for_evaluation(
        predictions: List[Dict], instances: List[Dict], k: Union[int, str]
) -> Iterable[Dict]:

    prediction_dict: Dict[str, Dict] = {
        pred['argument']: pred for pred in predictions
    }

    for gold_instance in instances:
        current_prediction: Dict = prediction_dict[gold_instance['id']]
        single_fallacy_prediction: Dict = dict()
        multi_label_predictions: Dict = dict()

        for gold_fallacy in gold_instance['argument']['fallacies']:

            # We need to be aware of all possible gold fallacies, each of them equally valid.
            gold_fallacy_dict: Dict[str, Dict] = {
                interchangeable_fallacy['class']: {
                    'id': interchangeable_fallacy['id'],
                    'premise': interchangeable_fallacy['premise']
                } for interchangeable_fallacy in gold_fallacy['interchangeable_fallacies']
            }
            predictions: List[Dict] = current_prediction['predictions'][gold_fallacy['id']]

            # If we don't find a hit, use the prediction ranked first
            hit_predicted_fallacy: Dict = predictions[0]
            hit_gold_fallacy: Optional[Dict] = None

            if k == 'all':
                use_predictions: List[Dict] = predictions[:]
            else:
                use_predictions = predictions[:k]

            # Remember multi-label classifications
            multi_label_predictions[gold_fallacy['id']] = use_predictions

            # Get the top single-label prediction within the top k results
            for fallacy_prediction in use_predictions:
                fallacy_name: str = fallacy_prediction['fallacy_name']
                if fallacy_name not in get_valid_fallacy_names():
                    raise ValueError(f'Unknown Fallacy Name: {fallacy_name}')

                # If we find a hit use this instead
                if fallacy_name in gold_fallacy_dict:
                    hit_gold_fallacy: Dict = gold_fallacy_dict[fallacy_name]
                    hit_predicted_fallacy: Dict = fallacy_prediction

            single_fallacy_prediction[gold_fallacy['id']] = {
                'predicted': hit_predicted_fallacy,
                'gold': hit_gold_fallacy
            }

        current_prediction['single_fallacy_predictions'] = single_fallacy_prediction
        current_prediction['predictions_at_k'] = multi_label_predictions
        yield current_prediction