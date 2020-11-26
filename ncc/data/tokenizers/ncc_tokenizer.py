from typing import List


class NCCTokenizer(object):
    """
    A tokenizer for sub/bep tokenization.
    """

    def __init__(self, *args, **kwargs):
        pass

    def token(self, x: str) -> List[str]:
        pass

    def decode(self, x: List[str]) -> str:
        return ' '.join(x)
