from typing import Dict, Tuple, List

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5EntailmentEvaluator:

    MODEL_NAME_T5_TRUE: str = 'google/t5_xxl_true_nli_mixture'

    def __init__(self):
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
            T5EntailmentEvaluator.MODEL_NAME_T5_TRUE
        ).to('cuda')
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            T5EntailmentEvaluator.MODEL_NAME_T5_TRUE
        )

        assert next(self.model.parameters()).is_cuda , 'not running on CUDA!'

        # This is used in the model (from different possible token_ids for "1")
        self.token_index_entailment = 209  # self.tokenizer.convert_tokens_to_ids("1")
        assert self.tokenizer.decode(self.token_index_entailment) == '1'
        self.token_index_no_entailment = self.tokenizer.convert_tokens_to_ids("0")

    def get_entailment_scores(self, generated_text: str, reference_text: str) -> Dict[str, float]:

        # standard NLI score, i.e. the entailment probability with the reference text a premise and generated
        # text as hypothesis
        entailment_probability_asym: float = self._compute_probability(p=reference_text, h=generated_text)

        # Compute ethe symmetric probability which allow for the reversed entailment
        entailment_probability_reversed: float = self._compute_probability(p=generated_text, h=reference_text)

        return {
            'nli_score': float(entailment_probability_asym),
            'symmetric_nli_score': float(max([entailment_probability_asym, entailment_probability_reversed]))
        }

    def _compute_probability(self, p: str, h: str) -> float:
        # T5 takes input in this form and predicts 1 or 0 for entailment or no entailment
        # The probability of "1" is used as the metric
        input_text: str = f'premise: {p} hypothesis: {h}'

        inputs = self.tokenizer([input_text], return_tensors="pt").to('cuda')
        output = self.model.generate(**inputs, max_length=5, output_scores=True, return_dict_in_generate=True)

        # Compute the probabilities for the desired output token ("0" or "1")
        generated_output_seq: torch.LongTensor = output.sequences
        input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]

        orig_generated_tokens = [self.tokenizer.decode(tok) for tok in generated_output_seq[0, input_length:]]
        if '1' not in orig_generated_tokens:

            # Find position of no entailment
            answer_idx_no_entailment = (generated_output_seq[0, :] == self.token_index_no_entailment).nonzero(
                as_tuple=True
            )[0]
            if len(answer_idx_no_entailment) == 0:
                raise ValueError(f'No entailment label found in {generated_output_seq[0, :]} ({orig_generated_tokens})')

            # change it with Entailment to compute the (low) entailment probability
            generated_output_seq[0, answer_idx_no_entailment[0].item()] = self.token_index_entailment

        transition_scores_e = self.model.compute_transition_scores(
            generated_output_seq, output.scores, normalize_logits=True
        )

        generated_tokens = generated_output_seq[:, input_length:]
        return_val = -1
        for tok, score in zip(generated_tokens[0], transition_scores_e[0]):
            if self.tokenizer.decode(tok).strip() == '1':
                return_val = np.exp(score.to('cpu').numpy())

        assert return_val >= 0, return_val
        return return_val

    def _get_transition_probabilities(self, scores: Tuple, token_idx: int):
        device = scores[0].device
        transition_scores = self.model.compute_transition_scores(
            torch.LongTensor([[token_idx] * len(scores)]).to(device), scores, normalize_logits=True
        )

        probabilities = np.array([np.exp(s.numpy()) for s in transition_scores[0]])
        print(f'Probabilities for {token_idx}:', [round(v, 4) for v in probabilities])
        return probabilities
