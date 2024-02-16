import re
from typing import Dict, Optional, Iterable, List

from missci.util.fallacy_util import get_all_fallacy_names_from_prediction, get_transform_other_dict, \
    normalize_fallacy_name


def get_first_fallacy_name_from_full_text(answer):

    fallacies = get_all_fallacy_names_from_prediction()
    fallacies = {f: f for f in fallacies}
    fallacies['False Causality'] = 'Causal Oversimplification'
    fallacies['Causal Simplification'] = 'Causal Oversimplification'
    fallacies['False Cause'] = 'Causal Oversimplification'
    fallacies['Fallacy of Equivocation'] = 'Ambiguity'
    fallacies['Fallacy of Equivalence'] = 'False Equivalence'
    fallacies['Other'] = 'Other'

    matches = map(lambda x: (answer.find(x), x), fallacies.keys())
    matches = filter(lambda x: x[0] >= 0, matches)
    matches = sorted(list(matches), key=lambda x: x[0])
    matches = map(lambda x: x[1], matches)
    matches = list(matches)

    if len(matches) == 0:
        for k in get_transform_other_dict().keys():
            if k.lower() in answer.lower():
                matches.append(get_transform_other_dict()[k])

    if len(matches) == 0:
        print('Nothing Found in:')
        print(answer)
        print('--\n')

    fallacy_class: str = fallacies[matches[0]]
    return fallacy_class


class ClassifyFallacyLLMOutputParser:

    def __init__(self, template_name: str):
        self.template_name: str = template_name
        self.pattern: re.Pattern = re.compile(r'^fallacy: (.+)$', re.IGNORECASE)

    def parse(self, prediction: Dict) -> Dict:
        answer: str = prediction['answer']

        fallacy_name: Optional[str] = self._get_fallacy(answer)
        normalized_fallacy_name: Optional[str] = normalize_fallacy_name(fallacy_name, fail_if_unk_fallacy=True)
        if normalized_fallacy_name is None:
            print(answer)
            assert False
        prediction['predicted_parsed'] = {
            'original_answer': fallacy_name,
            'fallacy_name': normalized_fallacy_name
        }
        return prediction

    def _get_fallacy(self, answer: str) -> Optional[str]:
        lines: Iterable[str] = answer.split('\n')
        lines = list(map(lambda x: x.strip(), lines))
        lines = list(filter(lambda x: len(x) > 0, lines))
        lines: List[str] = list(filter(lambda x: re.match(self.pattern, x), lines))
        if len(lines) == 0:
            return get_first_fallacy_name_from_full_text(answer)
        else:
            match = re.match(self.pattern, lines[0])
            fallacy: str = match.group(1).strip()
            return fallacy
