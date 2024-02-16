from typing import Dict, List, Optional, Iterable
import random

from missci.util.fallacy_util import get_valid_fallacy_names


class FallacyGenerationBaseline:
    def __init__(self, premise_strategy: str, fallacies: Optional[List[str]] = None):
        self.premise_strategy: str = premise_strategy
        self.fallacies: List[str] = fallacies or get_valid_fallacy_names(include_other=False)

    def run(self, instances: List[Dict], random_seed: int) -> List[Dict]:
        random.seed(random_seed)
        predictions: Iterable[Dict] = map(self._arg_to_baseline_prediction, instances)
        predictions = map(
            lambda x: x | {'params': {'seed': random_seed, 'premise_strategy': self.premise_strategy}}, predictions
        )
        return list(predictions)

    def _arg_to_baseline_prediction(self, argument: Dict) -> Dict:

        prediction_dict: Dict = {
            'argument': argument['id'],
            'single_fallacy_predictions': dict(),
            'predictions_at_k': dict()
        }

        for fallacy in argument['argument']['fallacies']:
            fallacy_id: str = fallacy['id']

            if self.premise_strategy == 'claim':
                fallacious_premise: str = argument['argument']['claim']
            elif self.premise_strategy == 'p0':
                fallacious_premise: str = argument['argument']['accurate_premise_p0']['premise']
            else:
                raise ValueError(f'UNK fallacy baseline strategy: "{self.premise_strategy}"')

            prediction_dict['single_fallacy_predictions'][fallacy_id] = {
                'predicted': {
                    'fallacy_name': random.choice(self.fallacies),
                    'fallacious_premise': fallacious_premise
                }
            }
            prediction_dict['predictions_at_k'][fallacy_id] = [
                prediction_dict['single_fallacy_predictions'][fallacy_id]['predicted']
            ]
        return prediction_dict
