# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# import os
# import importlib
# from ncc import registry
# from ncc.data.tokenizer.pretrained_tokenizer import PreTrainedTokenizer # , LegacyFairseqCriterion
#
# print('PreTrainedTokenizer: ', PreTrainedTokenizer)
#
# build_tokenizer, register_tokenizer, TOKENIZER_REGISTRY = registry.setup_registry(
#     'tokenizer',
#     base_class=PreTrainedTokenizer,
#     default='roberta',
# )
#
#
# print('build_tokenizer: ', build_tokenizer)
# print('TOKENIZER_REGISTRY: ', TOKENIZER_REGISTRY)
#
# # automatically import any Python files in the criterions/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith('.py') and not file.startswith('_'):
#         module = file[:file.find('.py')]
#         importlib.import_module('ncc.data.tokenizer.' + module)
