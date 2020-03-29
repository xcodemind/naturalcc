# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from ncc import registry
from ncc.criterions.fairseq_criterion import FairseqCriterion, LegacyFairseqCriterion


build_criterion, register_criterion, CRITERION_REGISTRY = registry.setup_registry(
    'criterion',
    base_class=FairseqCriterion,
    default='cross_entropy',
)
print('build_criterion: ', build_criterion)
print('criterions.py')
print('CRITERION_REGISTRY: ', CRITERION_REGISTRY)

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('ncc.criterions.' + module)
