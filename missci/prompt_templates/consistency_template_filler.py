from typing import List, Dict, Iterable, Tuple

from missci.prompt_templates.base_template_filler import TemplateFiller


class FallacyConsistencyTemplateFiller(TemplateFiller):

    KEY_CLAIM: str = '@@claim@@'
    KEY_FALLACIOUS_PREMISE: str = '@@fallacious_premise@@'
    KEY_P0: str = '@@p0@@'
    KEY_CONTEXT: str = '@@context@@'

    def extract_id(self, prediction_data: Dict) -> Tuple:
        return (
            prediction_data['argument'],
            prediction_data['fallacy_id']
        )

    def __init__(self, prompt_template_name: str, predictions: List[Dict], prediction_file: str, dest_file_prefix: str):
        self.id_to_predicted: Dict[str, Dict[str, str]] = dict()
        self.prediction_file: str = prediction_file

        for argument_prediction in predictions:
            for arg_fallacy_id in argument_prediction['single_fallacy_predictions'].keys():
                pred: Dict = argument_prediction['single_fallacy_predictions'][arg_fallacy_id]['predicted']
                fallacy_cls: str = pred['fallacy_name']
                premise: str = pred['fallacious_premise']
                assert arg_fallacy_id not in self.id_to_predicted
                self.id_to_predicted[arg_fallacy_id] = {
                    'fallacy_class': fallacy_cls, 'premise': premise
                }

        super().__init__(prompt_template_name, dest_file_prefix=dest_file_prefix)

    def _get_items_for_prompt(self, argument: Dict) -> Iterable[Dict]:
        claim: str = argument['argument']['claim']
        arg_id: str = argument['id']
        p0: str = argument['argument']['accurate_premise_p0']['premise']

        for fallacy in argument['argument']['fallacies']:
            fallacy_context: str = fallacy['fallacy_context']
            fallacy_id: str = fallacy['id']

            generated_premise: str = self.id_to_predicted[fallacy_id]['premise']
            generated_fallacy: str = self.id_to_predicted[fallacy_id]['fallacy_class']

            yield {
                'original_prediction_file': self.prediction_file,
                'argument': arg_id,
                'fallacy_id': fallacy_id,
                'claim': claim,
                'p0': p0,
                'fallacy_context': fallacy_context,
                'generated': {
                    'fallacy_class': generated_fallacy,
                    'premise': generated_premise
                }
            }

    def _fill_template(self, item: Dict, instance: Dict) -> str:

        return self.prompt_template.replace(
            FallacyConsistencyTemplateFiller.KEY_CLAIM, item['claim']
        ).replace(
            FallacyConsistencyTemplateFiller.KEY_FALLACIOUS_PREMISE, item['generated']['premise']
        ).replace(
            FallacyConsistencyTemplateFiller.KEY_P0, item['p0']
        ).replace(
            FallacyConsistencyTemplateFiller.KEY_CONTEXT, item['fallacy_context']
        )

    def _get_item_data(self, item: Dict, argument: Dict) -> Dict:
        return item
