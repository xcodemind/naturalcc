# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

from tree_sitter import Language

Language.build_library(
    '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/dataset/parser_zips/languages.so',  # your language parser file

    # Include one or more languages
    [
        # '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-c-master',
        # '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-cpp-master',
        # '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-c-sharp-master',
        '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-go-master',
        '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-python-master',
        '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-java-master',
        '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-javascript-master',
        '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-ruby-master',
        '/data/wanyao/yang/ghproj_d/GitHub/tree_sitter/tree-sitter-php-master',
    ]
)
