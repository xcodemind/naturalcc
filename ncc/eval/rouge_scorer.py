# -*- coding: utf-8 -*-

class RougeScorer(object):
    """
    compute rouge score of string
    >>> rouge_scorer = RougeScorer()
    >>> rouge_scorer.add_string(ref='The dog bit the man.', hyp='The dog bit the man.')
    >>> score = rouge_scorer.score()
    {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0}, 'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0}, 'rouge-l': {'f': 0.999999995, 'p': 1.0, 'r': 1.0}}
    """

    def __init__(self):
        from rouge import Rouge
        self.rouge = Rouge()
        self.reset()

    def reset(self):
        self.refs = []
        self.hyps = []

    def add_string(self, ref, hyp):
        self.refs.append(ref)
        self.hyps.append(hyp)

    def add_strings(self, refs, hyps):
        self.refs.extend(refs)
        self.hyps.extend(hyps)

    def score(self, avg=True):
        return self.rouge.get_scores(hyps=self.hyps, refs=self.refs, avg=avg)
