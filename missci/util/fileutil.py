import codecs
import json
import os.path
from os.path import join
from typing import Dict, List, Union, Iterable
from datetime import datetime


def get_current_day_string() -> str:
    return datetime.now().strftime('%Y-%m-%d')


def read_json(src: str):
    with codecs.open(src, encoding='utf-8') as f_in:
        return json.load(f_in)


def read_json_from_dir(directory: str, file: str) -> Union[Dict, List[Dict]]:
    return read_json(join(directory, file))


def read_jsonl(src: str) -> Iterable[Dict]:
    with codecs.open(src, encoding='utf-8') as f_in:
        for line in f_in.readlines():
            yield json.loads(line)


def read_jsonl_from_dir(directory: str, file: str):
    """
    Read a jsonl file.
    """
    return read_jsonl(join(directory, file))


def write_text(dest: str, text: str):
    with codecs.open(dest, 'w', encoding='utf-8') as f_out:
        f_out.write(text)


def write_pdf(dest: str, content):
    with codecs.open(dest, 'wb') as f_out:
        f_out.write(content)


def write_jsonl(dest_path: str, data: List[Dict], pretty=False) -> None:
    with codecs.open(dest_path, 'w', encoding='utf-8') as f_out:
        for sample in data:
            if pretty:
                f_out.write(json.dumps(sample, indent=2) + '\n')
            else:
                f_out.write(json.dumps(sample) + '\n')


def append_to_write_jsonl(dest_path: str, sample: Dict) -> None:
    with codecs.open(dest_path, 'a', encoding='utf-8') as f_out:
        f_out.write(json.dumps(sample) + '\n')


def write_jsonl_to_dir(directory: str, file: str, data: List[Dict]):
    """
    Write a jsonl file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    write_jsonl(join(directory, file), data)


def write_json(data: Dict, dest: str, pretty: bool = False) -> None:
    with codecs.open(dest, 'w', encoding='utf-8') as f_out:
        if pretty:
            f_out.write(json.dumps(data, indent=4))
        else:
            json.dump(data, f_out)


def write_json_to_dir(directory: str, file: str, data: Dict):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with codecs.open(join(directory, file), 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(data, indent=4))


def read_text(src: str) -> str:
    with codecs.open(src, encoding='utf-8') as f_in:
        doc = f_in.read()
    return doc.strip()

