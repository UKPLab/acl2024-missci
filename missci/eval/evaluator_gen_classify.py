from os.path import join
from typing import Optional, Dict, List, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from missci.data.missci_data_loader import MissciDataLoader
from missci.eval.evaluator_entailment_t5 import T5EntailmentEvaluator
from missci.eval.evaluator_premise import PremiseEvaluator
from missci.util.fileutil import read_jsonl, write_json
from missci.util.directory_util import get_prediction_directory


class GenClassifyEvaluator:
    def __init__(
            self,
            prediction_directory: Optional[str] = None,
            data_directory: Optional[str] = None,
            only_evaluate_predicted: bool = False,
            split: str = 'test',
            do_nli_eval: bool = True,
            do_meteor_eval: bool = True,
            do_bert_score_eval: bool = True,

    ):
        self.prediction_directory: str = prediction_directory or get_prediction_directory('generate-classify')
        self.data_loader: MissciDataLoader = MissciDataLoader(data_directory)
        self.gold_instances: List[Dict] = self.data_loader.load_raw_arguments(split)
        self.only_evaluate_predicted: bool = only_evaluate_predicted
        self.sbert: SentenceTransformer = SentenceTransformer("all-mpnet-base-v2")

        if do_nli_eval:
            self.nli_evaluator: Optional[T5EntailmentEvaluator] = T5EntailmentEvaluator()
        else:
            self.nli_evaluator = None

        self.premise_evaluator: PremiseEvaluator = PremiseEvaluator(
            use_bert_score=do_bert_score_eval, use_meteor=do_meteor_eval, use_bleu_score=False
        )

    def evaluate_file(self, file_name: str) -> Dict:
        predictions: List[Dict] = list(read_jsonl(join(self.prediction_directory, file_name)))
        scores: Dict = self.evaluate_instances(predictions)
        score_file_name = file_name.replace('.jsonl', '.json')
        score_file_name = f'evaluation__{score_file_name}'
        write_json(scores, join(self.prediction_directory, score_file_name), pretty=True)
        return scores

    def evaluate_instances(self, predictions: List[Dict]) -> Dict:

        prediction_dict: Dict[str, Dict[str, List[Dict]]] = {
            pred['argument']: pred for pred in predictions
        }

        if self.only_evaluate_predicted:
            predicted_ids: Set[str] = set(map(lambda x: x['id'], predictions))
            gold_instances: List[Dict] = list(filter(lambda x: x['id'] in predicted_ids, self.gold_instances))
        else:
            gold_instances: List[Dict] = self.gold_instances

        # Multi-label classification metrics
        return self._get_metrics(prediction_dict, gold_instances)

    def _get_metrics(self, prediction_dict: Dict[str, Dict[str, List[Dict]]], gold_instances: List[Dict]) -> Dict:

        multi_label_gold: List[List[str]] = []
        multi_label_pred: List[List[str]] = []

        selected_premises: List[Dict] = []

        # Compute argument-level metrics
        argument_level_num_fallacies_gold: List[int] = []
        # We consider correct if we found at least one fallacy class that matches any if the interchangeable fallacy
        # classes
        argument_level_num_fallacies_correct: List[int] = []

        for gold_instance in gold_instances:
            argument_id: str = gold_instance['id']
            pred: Dict = prediction_dict[argument_id]

            current_num_gold_fallacies: int = 0
            current_num_correct_fallacies: int = 0

            # Each argument can have multiple fallacies
            for gold_fallacy in gold_instance['argument']['fallacies']:

                # Add gold fallacies
                current_interchangeable_gold_fallacies: List[str] = list(
                    map(lambda x: x['class'], gold_fallacy['interchangeable_fallacies'])
                )

                # Add predicted fallacies
                current_predicted_fallacies: List[str] = list(
                    map(lambda x: x['fallacy_name'], pred['predictions_at_k'][gold_fallacy['id']])
                )

                current_num_gold_fallacies += 1
                if len(set(current_interchangeable_gold_fallacies) & set(current_predicted_fallacies)) > 0:
                    current_num_correct_fallacies += 1

                multi_label_gold.append(current_interchangeable_gold_fallacies)
                multi_label_pred.append(current_predicted_fallacies)

                # Get selected premises
                selected_premises.append({
                    'id': gold_fallacy['id'],
                    'gold_fallacies': gold_fallacy['interchangeable_fallacies'],
                    'predicted_fallacies': pred['predictions_at_k'][gold_fallacy['id']]
                })

            # Update for each argument:
            argument_level_num_fallacies_gold.append(current_num_gold_fallacies)
            argument_level_num_fallacies_correct.append(current_num_correct_fallacies)

        # Compute P@k
        predictions_with_positive_at_k: int = 0
        assert len(multi_label_gold) == len(multi_label_pred)

        for instance_gold, instance_pred in zip(multi_label_gold, multi_label_pred):
            if len(set(instance_gold) & set(instance_pred)) > 0:
                predictions_with_positive_at_k += 1

        # Compute argument-level metrics
        assert len(argument_level_num_fallacies_gold) == len(argument_level_num_fallacies_correct)
        num_args: int = len(argument_level_num_fallacies_gold)
        arg_1: float = len(list(filter(lambda x: x > 0, argument_level_num_fallacies_correct))) / num_args

        arg_at_t: Dict = dict()
        for t in [0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            count_args_correct_at_t: int = 0
            for num_gold_fallacies, num_correctly_predicted in zip(
                    argument_level_num_fallacies_gold, argument_level_num_fallacies_correct
            ):
                ratio_correct: float = num_correctly_predicted / num_gold_fallacies
                if ratio_correct >= t:
                    count_args_correct_at_t += 1
            arg_at_t[t * 100] = count_args_correct_at_t / num_args

        # Compute micro/macro F1
        classes = sorted(list(set(cls for gold in multi_label_gold for cls in gold)))
        mlb = MultiLabelBinarizer()
        mlb.fit([classes])

        encoded_gold = mlb.transform(multi_label_gold)
        encoded_pred = mlb.transform(multi_label_pred)

        fallacy_wise_metrics: Dict = dict()
        fallacy_level_f1: np.ndarray = f1_score(encoded_gold, encoded_pred, zero_division=0, average=None)
        fallacy_level_precision: np.ndarray = precision_score(encoded_gold, encoded_pred, zero_division=0, average=None)
        fallacy_level_recall: np.ndarray = recall_score(encoded_gold, encoded_pred, zero_division=0, average=None)

        # Compute metrics per fallacy class
        for i, cls in enumerate(mlb.classes_):
            fallacy_wise_metrics[cls] = {
                'precision': float(fallacy_level_precision[i]),
                'recall': float(fallacy_level_recall[i]),
                'f1': float(fallacy_level_f1[i])
            }

        metrics: Dict = {
            'arg-1': arg_1,
            'arg_at_t': arg_at_t,
            'has_positive_at_k': predictions_with_positive_at_k / len(multi_label_gold),
            'f1_micro': float(f1_score(encoded_gold, encoded_pred, average="micro", zero_division=0)),
            'f1_macro': float(f1_score(encoded_gold, encoded_pred, average="macro", zero_division=0)),
            'f1_sample_avg': float(f1_score(encoded_gold, encoded_pred, average="samples", zero_division=0)),
            'fallacy_wise': fallacy_wise_metrics,
            'fallacious_premises': self.evaluate_fallacious_premises(selected_premises, classes)
        }
        return metrics

    def evaluate_fallacious_premises(self, selected_premises: List[Dict], gold_labels: List[str]):
        matched_instances: List[Dict] = []
        for instance in tqdm(selected_premises):
            matched_instance: Dict = self._select_premise_pairs_for_eval(instance)
            if self.nli_evaluator is not None:
                # Remember them because more expensive
                matched_instance['nli_scores'] = self.nli_evaluator.get_entailment_scores(
                    generated_text=matched_instance['predicted_premise'],
                    reference_text=matched_instance['gold_premise']
                )
            matched_instances.append(matched_instance)

        # Evaluate over  correct / incorrect / all
        correct_predictions: List[Dict] = []
        incorrect_predictions: List[Dict] = []
        for pred in matched_instances:
            if pred['predicted_class'] == pred['gold_class']:
                correct_predictions.append(pred)
            else:
                incorrect_predictions.append(pred)

        premise_scores: Dict = dict()
        for name, predictions in [
            ('correct', correct_predictions), ('incorrect', incorrect_predictions), ('all', matched_instances)
        ]:
            if len(predictions) > 0:
                generated: List[str] = list(map(lambda x: x['predicted_premise'], predictions))
                references: List[str] = list(map(lambda x: x['gold_premise'], predictions))
                current_metrics: Dict = self.premise_evaluator.evaluate(generated, references)
                current_metrics['count'] = len(generated)

                if self.nli_evaluator is not None:
                    nli_a = []
                    nli_s = []
                    for pred in predictions:
                        nli_a.append(pred['nli_scores']['nli_score'])
                        nli_s.append(pred['nli_scores']['symmetric_nli_score'])

                    assert len(nli_a) > 0
                    current_metrics['nli'] = {
                        'nli-a': float(np.mean(nli_a)) if len(nli_a) > 0 else None,
                        'nli-s': float(np.mean(nli_s)) if len(nli_a) > 0 else None
                    }
                premise_scores[name] = current_metrics

        # Fallacy wise
        fallacy_wise_metrics: Dict = dict()
        for fallacy_cls in gold_labels:
            fallacy_correct_predictions: List[Dict] = list(
                filter(lambda x: x['predicted_class'] == fallacy_cls, correct_predictions)
            )
            num_correct: int = len(fallacy_correct_predictions)
            generated: List[str] = list(map(lambda x: x['predicted_premise'], fallacy_correct_predictions))
            references: List[str] = list(map(lambda x: x['gold_premise'], fallacy_correct_predictions))
            fallacy_wise_metrics[fallacy_cls] = self.premise_evaluator.evaluate(
                generated=generated, references=references
            )
            if self.nli_evaluator is not None:
                nli_a = []
                nli_s = []
                for pred in fallacy_correct_predictions:
                    nli_a.append(pred['nli_scores']['nli_score'])
                    nli_s.append(pred['nli_scores']['symmetric_nli_score'])

                fallacy_wise_metrics[fallacy_cls]['nli'] = {
                    'nli-a': float(np.mean(nli_a)) if len(nli_a) > 0 else None,
                    'nli-s': float(np.mean(nli_s)) if len(nli_a) > 0 else None
                }
            fallacy_wise_metrics[fallacy_cls]['count_correct'] = num_correct

        return {
            'mappings': matched_instances,
            'per_fallacy': fallacy_wise_metrics,
            'per_correct': premise_scores
        }

    def _select_premise_pairs_for_eval(self, instance: Dict):
        _id: List[Dict] = instance['id']
        predicted_fallacies: List[Dict] = instance['predicted_fallacies']
        interchangeable_gold_fallacies: List[Dict] = instance['gold_fallacies']

        # select single mapping
        selected_gold_premise: Optional[str] = None
        selected_gold_id: Optional[str] = None
        selected_predicted_premise: Optional[str] = None
        selected_gold_class: Optional[str] = None
        selected_predicted_class: Optional[str] = None

        gold_fallacy_dict: Dict[str, Tuple[str, str]] = {
            f['class']: (f['id'], f['premise']) for f in interchangeable_gold_fallacies
        }

        # Select the first correct fallacy (akin to HasPositive@k) that matches
        for prediction in predicted_fallacies:
            if prediction['fallacy_name'] in gold_fallacy_dict:
                selected_gold_id, selected_gold_premise = gold_fallacy_dict[prediction['fallacy_name']]
                selected_predicted_premise = prediction['fallacious_premise']
                selected_gold_class = prediction['fallacy_name']
                selected_predicted_class = prediction['fallacy_name']
                break

        # If no match was found we select the best based on cosine similarity with SBERT
        if selected_gold_premise is None:
            # Find the best match based on the highest cosine similarity
            gold_premise_sentences: List[str] = list(map(lambda x: x['premise'], interchangeable_gold_fallacies))
            pred_premise_sentences: List[str] = list(map(lambda x: x['fallacious_premise'], predicted_fallacies))

            embeddings_gold = self.sbert.encode(gold_premise_sentences, convert_to_tensor=True)
            embeddings_pred = self.sbert.encode(pred_premise_sentences, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings_gold, embeddings_pred).cpu()
            gold_idx, pred_idx = np.where(cosine_scores == cosine_scores.max())
            assert len(gold_idx) > 0
            gold_idx = gold_idx[0]
            pred_idx = pred_idx[0]

            selected_gold_premise: Optional[str] = interchangeable_gold_fallacies[gold_idx]['premise']
            selected_gold_id: Optional[str] = interchangeable_gold_fallacies[gold_idx]['id']
            selected_predicted_premise: Optional[str] = predicted_fallacies[pred_idx]['fallacious_premise']
            selected_gold_class: Optional[str] = interchangeable_gold_fallacies[gold_idx]['class']
            selected_predicted_class: Optional[str] = predicted_fallacies[pred_idx]['fallacy_name']

        return {
            'gold_premise': selected_gold_premise,
            'gold_id': selected_gold_id,
            'gold_class': selected_gold_class,
            'predicted_premise': selected_predicted_premise,
            'predicted_class': selected_predicted_class,
        }
