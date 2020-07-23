# -*- coding: utf-8 -*-

import torch
import numpy as np


def NDCG(relevance: torch.Tensor):
    """

    Args:
        relevance: (cosine) relevance matrix
        relevance(i, j) = cosine_similarity(repr_i, repr_j)

    Returns:

    """
    for query_idx, query_rel in enumerate(relevance):
        current_rank = 1
        query_dcg = 0

        sorted_query_rel, indices = torch.sort(query_rel, descending=True)




def ndcg(predictions: Dict[str, List[str]], relevance_scores: Dict[str, Dict[str, float]],
         ignore_rank_of_non_annotated_urls: bool = True) -> float:
    num_results = 0
    ndcg_sum = 0

    for query, query_relevance_annotations in relevance_scores.items():
        current_rank = 1
        query_dcg = 0
        for url in predictions[query]:
            if url in query_relevance_annotations:
                query_dcg += (2 ** query_relevance_annotations[url] - 1) / np.log2(current_rank + 1)
                current_rank += 1
            elif not ignore_rank_of_non_annotated_urls:
                current_rank += 1

        query_idcg = 0
        for i, ideal_relevance in enumerate(sorted(query_relevance_annotations.values(), reverse=True), start=1):
            query_idcg += (2 ** ideal_relevance - 1) / np.log2(i + 1)
        if query_idcg == 0:
            # We have no positive annotations for the given query, so we should probably not penalize anyone about this.
            continue
        num_results += 1
        ndcg_sum += query_dcg / query_idcg
    return ndcg_sum / num_results


if __name__ == '__main__':
    ...
