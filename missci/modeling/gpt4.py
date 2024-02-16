import os
from datetime import datetime
from os.path import join, exists
from typing import List, Dict, Optional, Tuple, Set, Iterable

from openai.lib.azure import AzureOpenAI
from tqdm import tqdm

from missci.prompt_templates.base_template_filler import TemplateFiller
from missci.util.fileutil import read_json, read_jsonl, append_to_write_jsonl


class GPTCaller:
    def __init__(
            self,
            output_directory: str,
            template_filler: TemplateFiller,
            gpt_version: Optional[str] = 'gpt-4',
            api_version: str = '2023-10-01-preview',
            overwrite: bool = False,
            max_new_token_len: int = 1000,
    ):

        self.output_directory: str = output_directory
        self.gpt_version: str = gpt_version
        self.api_version: str = api_version
        self.overwrite: bool = overwrite
        self.template_filler: TemplateFiller = template_filler
        self.max_new_token_len: int = max_new_token_len

        self.log_params: Dict = {
            'template': self.template_filler.prompt_template_name,
            'max_new_token_len': max_new_token_len,
            'model': gpt_version,
            'api_version': api_version
        }

        credentials: Dict = read_json('llm-config.json')[self.gpt_version]
        self.client: AzureOpenAI = AzureOpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            api_version=self.api_version,
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"]
        )

    def prompt(self, instances: List[Dict], split: str):

        template_name_file: str = self.template_filler.prompt_template_name
        template_name_file: str = template_name_file.replace('\\', '--').replace('/', '--').replace('.txt', '')
        dest_file_name: str = f'missci{self.template_filler.get_file_prefix()}_{template_name_file}_gpt4_{split}.jsonl'

        # template_key: str = self.template_filler.get_file_prefix()
        # dest_file_name: str = f'{template_key}_gpt4_{split}.jsonl'
        dest_path: str = join(self.output_directory, dest_file_name)

        # Use existing predictions instead of re-querying online GPT
        predicted_ids: Set[Tuple] = set()

        if exists(dest_path):
            if self.overwrite:
                print('ATTENTION: Old predictions were removed')
                os.remove(dest_path)
            else:
                for prediction in read_jsonl(dest_path):
                    predicted_ids.add(self.template_filler.extract_id(prediction['data']))
                print(f'Reusing {len(predicted_ids)} predictions')

        for argument in tqdm(instances):
            prompt_tasks: Iterable[Dict] = self.template_filler.get_prompts_gpt(argument)

            for prompt_task in prompt_tasks:
                current_id: Tuple = self.template_filler.extract_id(prompt_task['data'])
                if current_id not in predicted_ids:
                    prompt: str = prompt_task['prompt']
                    output, usage = self._get_output(prompt)

                    result: Dict = {
                        'answer': output,
                        'transition_scores': [],
                        'log_probabilities': [],
                        'params': self.log_params,
                        'data': prompt_task['data'] | {
                            "prompt": prompt,
                            "time": str(datetime.now()),
                            'usage': usage
                        }
                    }
                    append_to_write_jsonl(dest_path, result)
        print('Predictions are in', dest_file_name)
        return dest_file_name

    def _get_output(self, prompt: str) -> Tuple[str, int]:
        messages = [
            {
                "role": "user",
                'content': prompt
            }]
        completion = self.client.chat.completions.create(
            model=self.gpt_version, messages=messages, max_tokens=self.max_new_token_len
        )
        output = completion.choices[0].message.content
        usage = completion.usage.total_tokens
        return output, usage
