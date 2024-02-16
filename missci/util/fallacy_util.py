from typing import List, Dict, Optional, Iterable, Tuple


def get_valid_fallacy_names(include_other: bool = True) -> List[str]:
    """
    Only these are applicable for ealuation.
    """
    fallacy_names: List[str] = [
        'Ambiguity',
        'Biased Sample Fallacy',
        'Causal Oversimplification',
        'Fallacy of Division/Composition',
        'Fallacy of Exclusion',
        'False Dilemma / Affirming the Disjunct',
        'False Equivalence',
        'Hasty Generalization',
        'Impossible Expectations'
    ]

    if include_other:
        fallacy_names.append('Other')

    return fallacy_names


def get_all_fallacy_names_from_prediction() -> List[str]:
    return [
        'Ambiguity', 'Impossible Expectations', 'False Equivalence', 'False Dilemma', 'Biased Sample Fallacy',
        'Hasty Generalization', 'Causal Oversimplification', 'Fallacy of Composition', 'Fallacy of Exclusion'
    ]


def get_advanced_fallacy_mapping_dict() -> Dict[str, str]:
    return {
        'False Cause': 'Causal Oversimplification',
        'False Causality': 'Causal Oversimplification',
        'Fallacy of Oversimplification': 'Causal Oversimplification',
        'Correlation does not imply causation': 'Causal Oversimplification',
        'Causal oversimplification': 'Causal Oversimplification',
        'Correlation-Causality Fallacy': 'Causal Oversimplification',

        'False Exclusion': 'Fallacy of Exclusion',
        'Exclusion': 'Fallacy of Exclusion',
        'Argument from Ignorance': 'Fallacy of Exclusion',

        'Fallacy of Equivocation': 'Ambiguity',
        'ambiguous': 'Ambiguity',
        'false-equivocation': 'Ambiguity',

        'False Expectations': 'Impossible Expectations',
        'Imaginary Expectations': 'Impossible Expectations',

        'Fallacy of Biased Sample': 'Biased Sample Fallacy',

        'Fallacy of Equivalence': 'False Equivalence',
        'False Analogy': 'False Equivalence',

        'Bias Sample Fallacy': 'Biased Sample Fallacy'
    }


def get_transform_other_dict() -> Dict:
    return {
        fallacy: 'Other' for fallacy in [
            'Non Sequitur', 'False Conclusion', 'Coercive Definition', 'False Negative',
            'Appeal to Authority', 'Fallacy of Authority', 'False Generalization', 'False Authority',
            'Confirmation Bias', 'None', 'False Assumption', 'Other', 'Fallacy of Repetition', 'Begging the Question',
            'Fallacy of Tradition', 'Fallacy of Misleading Statistics',
            'impossible to correctly identify the applied fallacy class.',
            "it's impossible to accurately determine what fallacy might be in play",
            "ot enough information given in this example to determine an applied fallacy",
            "cannot determine the fallacy applied"
        ]
    }


def normalize_fallacy_name(
        extracted_name: Optional[str], fail_if_unk_fallacy: bool = True, transform_other: bool = True
) -> Optional[str]:
    if extracted_name is None:
        return extracted_name
    else:

        # Find the first occurring fallacy based on the actual fallacies!
        fallacies: Dict[str, str] = {
            f: f for f in get_all_fallacy_names_from_prediction()
        } | get_advanced_fallacy_mapping_dict()

        # Find exact match: locate the position of each known fallacy in the text 8in case multiple exist)
        positions: Iterable[Tuple[str, int]] = list(map(lambda x: (x, extracted_name.find(x)), list(fallacies.keys())))
        # get rid of all fallacies that do not exist in the text

        # as tie-breaker consider the first mentioned fallacy
        positions = filter(lambda x: x[1] >= 0, positions)
        positions: List[Tuple[str, int]] = sorted(list(positions), key=lambda x: x[0])

        other_fallacy_dict: Dict = get_transform_other_dict()
        if len(positions) == 0:
            # Check if we hit fallacies out of our labels that cannot be mapped
            positions: Iterable[Tuple[str, int]] = list(
                map(lambda x: (x, extracted_name.find(x)), list(other_fallacy_dict.keys()))
            )
            positions = filter(lambda x: x[1] >= 0, positions)
            positions: List[Tuple[str, int]] = sorted(list(positions), key=lambda x: x[0])

        if len(positions) > 0:
            if positions[0][0] in fallacies:
                fallacy_name: str = fallacies[positions[0][0]]
            else:
                fallacy_name: str = other_fallacy_dict[positions[0][0]]
        elif fail_if_unk_fallacy:

            raise NotImplementedError(f'unknown fallacy: {extracted_name}')
        else:
            return None

        # Convert to fallacy names as stated in the labels of the dataset
        conversion_dict: Dict[str, str] = {
            'Fallacy of Composition': 'Fallacy of Division/Composition',
            'False Dilemma': 'False Dilemma / Affirming the Disjunct'
        }

        if fallacy_name in conversion_dict:
            fallacy_name = conversion_dict[fallacy_name]

        assert extracted_name is not None

        return fallacy_name
