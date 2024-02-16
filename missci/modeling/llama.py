from os.path import join
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

from missci.util.fileutil import read_json


class LlamaCaller:
    def __init__(self, model, tokenizer, generation_config,
                 max_prompt_input_size: int,
                 max_new_tokens: int,
                 temperature: Optional[float],
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_prompt_input_size: int = max_prompt_input_size
        self.max_new_tokens: int = max_new_tokens
        self.temperature: Optional[float] = temperature

    def get_prompt_len(self, prompt: str) -> int:
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_prompt_input_size + 1,
            truncation=True,
            padding=False,
            add_special_tokens=False
        )
        return inputs['input_ids'].size()[1]

    def get_output(self, prompt: str) -> Dict:
        """
        Single batch prompting
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_prompt_input_size + 1,
            truncation=True,
            padding=False,
            add_special_tokens=False
        ).to(self.model.device)
        if inputs['input_ids'].size()[1] > self.max_prompt_input_size:
            # Rather do it this way than truncating
            raise ValueError(
                f"Too long input: {inputs['input_ids'].size()} (expected max. {self.max_prompt_input_size})"
            )

        if self.temperature is not None:
            with torch.inference_mode():
                output_dict = self.model.generate(
                    **inputs,
                    # input_ids=inputs["input_ids"].to("cuda"),
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1,
                    num_return_sequences=1,
                    temperature=self.temperature,
                    do_sample=True,
                    generation_config=self.generation_config
                )
        else:
            with torch.inference_mode():
                output_dict = self.model.generate(
                    **inputs,
                    # input_ids=inputs["input_ids"].to("cuda"),
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1,
                    num_return_sequences=1,
                    generation_config=self.generation_config
                )

        scores = output_dict['scores']
        sequences = output_dict['sequences']

        transition_scores = self.model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )

        prompt_len: int = inputs["input_ids"].size()[1] + 1  # because of " "
        assert inputs["input_ids"].size()[0] == 1, 'code below is only for single batch'

        transition_scores = transition_scores[:, :prompt_len][0].cpu().numpy()
        sequences = sequences[:, prompt_len:]
        output_text: str = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        log_probabilities = np.exp(transition_scores)

        return {
            'answer': output_text,
            'transition_scores': list(map(float, transition_scores)),
            'log_probabilities': list(map(float, log_probabilities))
        }


def get_llama_caller(
        model_variant: str,
        max_prompt_input_size: int,
        max_new_tokens: int,
        temperature: Optional[float],
        llm_config_path: str = 'llm-config.json',
        load_in_8bit: bool = True,
        load_in_4bit: bool = False
) -> LlamaCaller:

    if load_in_8bit:
        assert not load_in_4bit

    cache_dir: str = read_json(llm_config_path)['llama2']['directory']
    stored_model_path: str = join(cache_dir, model_variant)
    generation_config = GenerationConfig.from_pretrained(stored_model_path)

    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            stored_model_path,
            device_map="auto",
            load_in_8bit=True
        )
    elif load_in_4bit:
        conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            stored_model_path,
            device_map="auto",
            load_in_4bit=True,
            quantization_config=conf
        )
    else:
        raise NotImplementedError()
    model = model.eval()

    is_cuda = next(model.parameters()).is_cuda
    print('IS CUDA:', is_cuda)
    assert is_cuda

    tokenizer = AutoTokenizer.from_pretrained(stored_model_path)

    return LlamaCaller(
        model, tokenizer, generation_config,
        max_prompt_input_size=max_prompt_input_size, max_new_tokens=max_new_tokens, temperature=temperature
    )
