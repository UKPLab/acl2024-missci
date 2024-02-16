from os.path import join
from typing import List, Dict, Iterable, Optional

from tqdm import tqdm
from transformers import set_seed

from missci.modeling.llama import LlamaCaller, get_llama_caller
from missci.prompt_templates.base_template_filler import TemplateFiller
from missci.prompt_templates.classify_generate_template_filler import ClassifyGenerateTemplateFiller
from missci.prompt_templates.consistency_template_filler import FallacyConsistencyTemplateFiller
from missci.prompt_templates.fallacy_classify_template_filler import FallacyWiseTemplateFiller
from missci.util.fileutil import write_jsonl


def query_llama_for_classification_generation(
        split: str, instances: List[Dict], output_directory: str, template_file: str, llama_size: str,
        seed: int, temperature: Optional[float], run_8bit: bool = False
):
    template_filler: TemplateFiller = ClassifyGenerateTemplateFiller(template_file)

    return query_llama(
        template_file=template_file,
        llama_size=llama_size,
        split=split,
        instances=instances,
        template_filler=template_filler,
        output_directory=output_directory,
        seed=seed,
        temperature=temperature,
        run_8bit=run_8bit
    )


def query_llama_for_fallacy_consistency(
        split: str, instances: List[Dict], predictions: List[Dict], output_directory: str, template_file: str,
        llama_size: str, prediction_file_name: str, dest_name_prefix: str
):
    return query_llama(
        template_file=template_file,
        llama_size=llama_size,
        split=split,
        instances=instances,
        template_filler=FallacyConsistencyTemplateFiller(
            template_file, predictions, prediction_file_name, dest_name_prefix
        ),
        output_directory=output_directory
    )


def query_llama_for_fallacy_classification_with_gold_premise(
        split: str, instances: List[Dict], output_directory: str, template_file: str, llama_size: str,
        seed: int, use8bit: bool = False
) -> str:
    return query_llama(
        template_file=template_file,
        llama_size=llama_size,
        split=split,
        instances=instances,
        template_filler=FallacyWiseTemplateFiller(template_file),
        output_directory=output_directory,
        seed=seed,
        run_8bit=use8bit
    )


def query_llama_for_classification_with_implicit_premise(
        split: str, instances: List[Dict], output_directory: str, template_file: str, llama_size: str,
        seed: int, use8bit: bool = False
) -> str:

    template_filler = ClassifyGenerateTemplateFiller(template_file)

    return query_llama(
        template_file=template_file,
        llama_size=llama_size,
        split=split,
        instances=instances,
        template_filler=template_filler,
        output_directory=output_directory,
        seed=seed,
        run_8bit=use8bit
    )


def query_llama(
        template_file: str, llama_size: str, split: str,
        instances: List[Dict], template_filler: TemplateFiller, output_directory: str,
        seed: int = 1,
        max_prompt_len: int = 5000, max_new_token_len: int = 1000, temperature: Optional[float] = None,
        run_8bit: bool = False,
        dest_name: Optional[str] = None
) -> str:
    key2llama = {
        '70b': '70B-Chat', '13b': '13B-Chat', '7b': '7B-Chat'
    }
    if llama_size not in set(key2llama.keys()):
        raise ValueError(llama_size)

    set_seed(seed)

    run_8bit = run_8bit or llama_size != '70b'

    llama2: LlamaCaller = get_llama_caller(
        model_variant=key2llama[llama_size],
        max_prompt_input_size=max_prompt_len,
        max_new_tokens=max_new_token_len,
        temperature=temperature,
        load_in_4bit=not run_8bit,
        load_in_8bit=run_8bit
    )

    template_name_file: str = template_file.replace('\\', '--').replace('/', '--').replace('.txt', '')
    temperature_key: str = ''
    if temperature is not None:
        temperature_key = f'_T{str(temperature).replace(".", "-")}'

    if dest_name is None:
        dest_name: str = f'missci{template_filler.get_file_prefix()}_{template_name_file}_{llama_size}_{temperature_key}_{split}.jsonl'
    if run_8bit:
        dest_name = dest_name.replace('.jsonl', '.8bit.jsonl')

    log_params: Dict = {
        'template': template_file,
        'temperature': temperature,
        'seed': seed,
        'max_prompt_len': max_prompt_len,
        'max_new_token_len': max_new_token_len,
        'llama_variant': key2llama[llama_size]
    }

    predictions: List[Dict] = []
    for argument in tqdm(instances):
        prompt_tasks: Iterable[Dict] = template_filler.get_prompts_llama(argument)

        for prompt_task in prompt_tasks:
            prompt: str = prompt_task['prompt']
            data: Dict = prompt_task['data']
            result = llama2.get_output(prompt)
            result['params'] = log_params
            result['data'] = data
            predictions.append(result)

    if seed != 1:
        dest_name = dest_name.replace('.jsonl', f'_S{seed}.jsonl')

    write_jsonl(join(output_directory, dest_name), predictions)
    print('Done.')
    return dest_name
