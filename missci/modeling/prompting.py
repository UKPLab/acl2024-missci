def filled_template_to_prompt_llama(filled_template: str) -> str:
    b_inst: str = " [INST] "
    e_inst: str = " [/INST]"
    s_prompt: str = "<s>"

    if '@@system_prompt@@' in filled_template:
        filled_template = filled_template.replace('@@system_prompt@@', '')

    filled_template = filled_template.strip()

    # Make sure it is completely filled
    if '@@' in filled_template:
        raise ValueError(f'The template still contains unfilled fields: {filled_template}!')

    return s_prompt + b_inst + filled_template + e_inst


def filled_template_to_prompt_gpt(filled_template: str) -> str:

    if '@@system_prompt@@' in filled_template:
        filled_template = filled_template.replace('@@system_prompt@@', '')

    filled_template = filled_template.strip()

    # Make sure it is completely filled
    if '@@' in filled_template:
        raise ValueError(f'The template still contains unfilled fields: {filled_template}!')

    return filled_template
