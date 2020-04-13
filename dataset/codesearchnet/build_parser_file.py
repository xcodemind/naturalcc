# -*- coding: utf-8 -*-
from tree_sitter import Language

Language.build_library(
    '/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/codesearchnet/ruby.so',  # your language parser file

    # Include one or more languages
    [
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-c-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-cpp-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter /tree-sitter-c-sharp-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-go-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-python-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-java-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-javascript-master',
        '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-ruby-master',
        # '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/tree-sitter/tree-sitter-php-master',
    ]
)
print('exit...')
