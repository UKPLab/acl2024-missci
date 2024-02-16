from typing import Dict, List

import numpy as np
from evaluate import load


class PremiseEvaluator:
    def __init__(self,
                 use_bert_score: bool = True,
                 use_bleu_score: bool = True,
                 use_meteor: bool = True
                 ):
        self.use_bert_score: bool = use_bert_score
        self.use_bleu_score: bool = use_bleu_score
        self.use_meteor: bool = use_meteor

        self.metric_bert_score = None
        self.metric_bleu_score = None
        self.metric_meteor_score = None

        # Load metrics
        self._init_metrics()

    def _init_metrics(self):
        if self.use_bert_score:
            self.metric_bert_score = load("bertscore")
        if self.use_bleu_score:
            self.metric_bleu_score = load("bleu")
        if self.use_meteor:
            self.metric_meteor_score = load('meteor')

    def evaluate(self, generated: List[str], references: List[str]) -> Dict:
        metrics: Dict = dict()
        assert len(generated) == len(references)

        if self.use_bert_score:
            metrics['bert_score'] = self._compute_bert_score(generated, references) if len(generated) > 0 else None

        if self.use_bleu_score:
            metrics['bleu'] = self._compute_bleu_score(generated, references) if len(generated) > 0 else None

        if self.use_meteor:
            metrics['meteor'] = self._compute_meteor_score(generated, references) if len(generated) > 0 else None

        return metrics

    def _compute_bert_score(self, generated: List[str], references: List[str]) -> Dict:
        metrics: Dict = self.metric_bert_score.compute(
            predictions=generated, references=references, model_type="microsoft/deberta-xlarge-mnli", lang="en"
        )
        for key in ['precision', 'recall', 'f1']:
            metrics[key] = float(np.mean(metrics[key]))
        return metrics

    def _compute_bleu_score(self, generated: List[str], references: List[str],) -> float:
        metrics: Dict = self.metric_bleu_score.compute(predictions=generated, references=references)
        return metrics['bleu']

    def _compute_meteor_score(self, generated: List[str], references: List[str]) -> Dict[str, float]:
        metrics: Dict = self.metric_meteor_score.compute(predictions=generated, references=references)
        return metrics['meteor']
