from typing import Dict, Iterable, Tuple

from missci.prompt_templates.base_template_filler import TemplateFiller


class FallacyWiseTemplateFiller(TemplateFiller):

    def __init__(self, prompt_template_name: str):
        super().__init__(prompt_template_name, dest_file_prefix='')

    def extract_id(self, prediction_data: Dict) -> Tuple:
        return (
            prediction_data['argument'],
            prediction_data['fallacy_id']
        )

    KEY_CLAIM: str = '@@claim@@'
    KEY_FALLACIOUS_PREMISE: str = '@@fallacious_premise@@'
    KEY_P0: str = '@@p0@@'
    KEY_CONTEXT: str = '@@context@@'

    def _get_items_for_prompt(self, argument: Dict) -> Iterable[Dict]:
        claim: str = argument['argument']['claim']
        arg_id: str = argument['id']
        p0: str = argument['argument']['accurate_premise_p0']['premise']

        for fallacy in argument['argument']['fallacies']:
            fallacy_context: str = fallacy['fallacy_context']

            for fallacious_reasoning in fallacy['interchangeable_fallacies']:
                premise: str = fallacious_reasoning['premise']
                fallacy_id: str = fallacious_reasoning['id']
                fallacy_class: str = fallacious_reasoning['class']

                yield {
                    'argument': arg_id,
                    'fallacy_id': fallacy_id,
                    'claim': claim,
                    'p0': p0,
                    'fallacy_context': fallacy_context,
                    'premise': premise,
                    'gold_fallacy_class': fallacy_class
                }

    def _fill_template(self, item: Dict, instance: Dict) -> str:

        return self.prompt_template.replace(
            FallacyWiseTemplateFiller.KEY_CLAIM, item['claim']
        ).replace(
            FallacyWiseTemplateFiller.KEY_FALLACIOUS_PREMISE, item['premise']
        ).replace(
            FallacyWiseTemplateFiller.KEY_P0, item['p0']
        ).replace(
            FallacyWiseTemplateFiller.KEY_CONTEXT, item['fallacy_context']
        )

    def _get_item_data(self, item: Dict, argument: Dict) -> Dict:
        return item
