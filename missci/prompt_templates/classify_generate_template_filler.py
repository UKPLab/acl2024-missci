from typing import Iterable, Dict, Tuple

from missci.prompt_templates.base_template_filler import TemplateFiller


def get_prefix(template: str) -> str:
    template_name = template.split('/')[-1].split('.')[0]
    return f'missci_st3-generate--{template_name}'


class ClassifyGenerateTemplateFiller(TemplateFiller):

    KEY_CLAIM: str = '@@claim@@'
    KEY_P0: str = '@@p0@@'
    KEY_PASSAGE: str = '@@context@@'

    def __init__(self, prompt_template_name: str):
        super().__init__(prompt_template_name, dest_file_prefix='')

    def extract_id(self, prediction_data: Dict) -> Tuple:
        return (
            prediction_data['argument'],
            prediction_data['fallacy_id']
        )

    def _get_items_for_prompt(self, argument: Dict) -> Iterable[Dict]:
        for fallacy in argument['argument']['fallacies']:
            yield fallacy

    def _fill_template(self, item: Dict, instance: Dict) -> str:
        claim: str = instance['argument']['claim']
        p0: str = instance['argument']['accurate_premise_p0']['premise']
        context: str = item['fallacy_context']

        return self.prompt_template.replace(
            ClassifyGenerateTemplateFiller.KEY_CLAIM, claim
        ).replace(
            ClassifyGenerateTemplateFiller.KEY_PASSAGE, context
        ).replace(
            ClassifyGenerateTemplateFiller.KEY_P0, p0
        )

    def _get_item_data(self, item: Dict, argument: Dict) -> Dict:
        return {'fallacy_id': item['id']}
