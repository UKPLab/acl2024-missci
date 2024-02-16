from os.path import join
from typing import Dict, Iterable, Tuple

from missci.modeling.prompting import filled_template_to_prompt_llama, filled_template_to_prompt_gpt
from missci.util.fileutil import read_text


class TemplateFiller:

    def __init__(self,
                 prompt_template_name: str,
                 prompt_template_dir: str = './prompt_templates',
                 dest_file_prefix: str = ''
                 ):
        self.prompt_template_name: str = prompt_template_name
        self.dest_file_prefix: str = dest_file_prefix
        self.prompt_template: str = read_text(join(prompt_template_dir, prompt_template_name))

    def extract_id(self, prediction_data: Dict) -> Tuple:
        raise NotImplementedError()

    def get_file_prefix(self) -> str:
        return self.dest_file_prefix

    def _get_items_for_prompt(self, argument: Dict) -> Iterable[Dict]:
        raise NotImplementedError()

    def _fill_template(self, item: Dict, argument: Dict) -> str:
        raise NotImplementedError()

    def _get_item_data(self, item: Dict, argument: Dict) -> Dict:
        raise NotImplementedError()

    def get_prompts_llama(self, argument: Dict) -> Iterable[Dict]:

        for item in self._get_items_for_prompt(argument):
            filled_template: str = self._fill_template(item, argument)

            yield {
                'data': self._get_base_data(argument, self._get_item_data(item, argument)),
                'prompt': filled_template_to_prompt_llama(filled_template)
            }

    def get_prompts_gpt(self, argument: Dict) -> Iterable[Dict]:
        for item in self._get_items_for_prompt(argument):
            filled_template: str = self._fill_template(item, argument)

            yield {
                'data': self._get_base_data(argument, self._get_item_data(item, argument)),
                'prompt': filled_template_to_prompt_gpt(filled_template)
            }

    def _get_base_data(self, argument: Dict, data: Dict) -> Dict:
        claim: str = argument['argument']['claim']
        arg_id: str = argument['id']
        p0: str = argument['argument']['accurate_premise_p0']['premise']

        return_data: Dict = {
            'argument': arg_id,
            'claim': claim,
            'p0': p0,
            'prompt_template': self.prompt_template
        }
        for key in data:
            return_data[key] = data[key]

        return return_data
