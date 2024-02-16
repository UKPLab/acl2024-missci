import re
from typing import Dict, List, Optional, Iterable

from missci.util.fallacy_util import normalize_fallacy_name
from missci.util.post_processing import normalize_fallacious_premise


class GenerateClassifyLLMOutputParser:

    def __init__(self, template_name: str):
        self.template_name: str = template_name
        self.pattern_line1: re.Pattern = re.compile(
            r'^.*fallacious premise( 3\.?\d?)?: (.+)[;\."] applied fallacy class: (.+)$', re.IGNORECASE
        )
        self.pattern_line2: re.Pattern = re.compile(
            r'^.*fallacious premise( 3\.?\d?)?: (.+)[;\."] applied fallacy: (.+)$', re.IGNORECASE
        )
        self.pattern_full_text: re.Pattern = re.compile(
            r'^fallacious premise( 3\.?\d?)?: (.+)\n+applied fallacy class: (.+)$', re.IGNORECASE | re.MULTILINE
        )

    def parse(self, prediction: Dict) -> Dict:
        answer: str = prediction['answer']

        parsed_results: List[Dict] = self._get_answer(answer)

        if len(parsed_results) == 0:
            parsed_results = [{
                'fallacy': 'Other',
                'fallacious_premise': '',
                'answer_line': answer
            }]

        prediction['predicted_parsed'] = []
        for parsed_result in parsed_results:
            fallacy_name: Optional[str] = parsed_result['fallacy']
            fallacious_premise: Optional[str] = parsed_result['fallacious_premise']

            if fallacy_name is None:
                print('Answer was:', parsed_result['answer_line'])
                raise ValueError('no fallacy found!')

            normalized_premise: str = normalize_fallacious_premise(fallacious_premise)
            if normalized_premise is None:
                print(parsed_result['answer_line'])
                raise ValueError('no fallacious premise!')

            prediction['predicted_parsed'].append({
                'original_answer': parsed_result['answer_line'],
                'fallacy_name': normalize_fallacy_name(fallacy_name),
                'fallacious_premise': normalized_premise
            })

        return prediction

    def _parse_single_line(self, line: str) -> Dict:
        match = re.match(self.pattern_line1, line)
        if not match:
            match = re.match(self.pattern_line2, line)
        fallacy: str = match.group(3).strip()
        fallacious_premise: str = match.group(2).strip()
        return {
            'answer_line': line,
            'fallacy': fallacy,
            'fallacious_premise': fallacious_premise
        }

    def _get_answer(self, answer: str) -> List[Dict]:
        lines: Iterable[str] = answer.split('\n')
        lines = list(map(lambda x: x.strip(), lines))
        lines = list(filter(lambda x: len(x) > 0, lines))
        lines: List[str] = list(filter(
            lambda x: re.match(self.pattern_line1, x) or re.match(self.pattern_line2, x), lines
        ))
        if len(lines) == 0:
            return list(self._get_answer_from_full_text(answer))
        else:
            return list(map(self._parse_single_line, lines))

    def _get_answer_from_full_text(self, answer: str) -> Iterable[Dict]:
        for match in re.finditer(self.pattern_full_text, answer):
            yield {
                'answer_line': answer,
                'fallacy': match.group(3),
                'fallacious_premise': match.group(2)
            }
