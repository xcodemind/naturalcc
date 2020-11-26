import re
import ujson


def space_tokenizer(line):
    """json string => space tokenizer => list"""
    line = ujson.loads(line)
    tokens = re.sub('r\s+', ' ', line).strip()
    return tokens.split()


def list_tokenizer(line):
    """json string => list"""
    tokens = ujson.loads(line)
    return tokens
