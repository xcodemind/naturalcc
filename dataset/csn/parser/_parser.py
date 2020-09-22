# -*- coding: utf-8 -*-

from typing import *

import sys

import re
import os
import ujson
import itertools
from copy import deepcopy
from tree_sitter import Language, Parser

try:
    from dataset.csn.utils import constants
    from dataset.csn.utils import util
    from dataset.csn.utils import util_ast
except ImportError:
    from ..utils import constants
    from ..utils import util
    from ..utils import util_ast

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(constants.RECURSION_DEPTH)  # recursion depth


class CodeParser(object):
    '''parse code data into ast'''
    __slots__ = ('parser', 'to_lower', 'LANGUAGE', 'operators',)

    def __init__(self, SO_FILE: str, LANGUAGE: str, to_lower: bool = False, operators_file: str = None):
        self.parser = Parser()
        if LANGUAGE == 'csharp':
            LANGUAGE = 'c_sharp'
        self.parser.set_language(Language(SO_FILE, LANGUAGE))
        self.LANGUAGE = LANGUAGE
        self.to_lower = to_lower

        if operators_file is None:
            operators_file = os.path.join(os.path.dirname(__file__), 'operators.json')
        with open(operators_file, 'r') as reader:
            self.operators = ujson.load(reader)

    def parse_docstring(self, docstring: str) -> List[str]:
        '''parse comment from docstring'''
        docstring = re.sub(r'\{\@\S+', '', docstring)
        docstring = re.sub(r'{.+}', '', docstring)
        docstring = ''.join([char for char in docstring if char not in constants.MEANINGLESS_TOKENS])
        docstring = [util.split_identifier(token, str_flag=False) for token in docstring.split(' ')]
        docstring = list(itertools.chain(*docstring))
        docstring = util.stress_tokens(docstring)
        if self.to_lower:
            docstring = util.lower(docstring)
        return docstring

    def parse_docstring_tokens(self, docstring_tokens: List) -> List[str]:
        # parse comment from docstring_tokens
        docstring_tokens = [''.join([char for char in token if char not in constants.MEANINGLESS_TOKENS]) \
                            for token in docstring_tokens]
        docstring_tokens = itertools.chain(
            *[util.split_identifier(token, str_flag=False) for token in docstring_tokens]
        )
        docstring_tokens = util.stress_tokens(docstring_tokens)
        if self.to_lower:
            docstring_tokens = util.lower(docstring_tokens)
        return docstring_tokens

    def parse_comment(self, docstring: str, docstring_tokens: List[str], ) -> Optional[List[str]]:
        '''
        our refined comment parse function. if you prefer original comment, use docstring_tokens instead
        '''
        if (docstring_tokens[-1] in constants.COMMENT_END_TOKENS) or \
            (len(docstring_tokens) > constants.MAX_COMMENT_TOKEN_LIST_LEN):
            # if docstring_tokens is too long or docstring_tokens is wrong parsed
            ''' exceptions in CodeSearchNet, eg.
            docstring: 'Set {@link ServletRegistrationBean}s that the filter will be registered against.
                        @param servletRegistrationBeans the Servlet registration beans'
            docstring_tokens:  <class 'list'>: ['Set', '{'] ['.']
            '''

            # skip this code snippet, if there are non-ascii tokens
            if not util.is_ascii(docstring):
                return None
            docstring = re.sub(
                '(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', ' ',
                docstring)  # delete url
            # remove additional and useless tails info
            docstring = str.split(docstring, '\n\n')[0].replace('\n', ' ')
            docstring = re.split(r'[\.|:]', docstring)
            docstring = docstring[0] + '.'  # add . at the end of sentence
            comment_tokens = self.parse_docstring(docstring)
        else:
            # skip this code snippet, if there are non-ascii tokens
            for comment_token in docstring_tokens:
                if not util.is_ascii(comment_token):
                    return None
            comment_tokens = self.parse_docstring_tokens(docstring_tokens)

        ########################################################################################
        # add . at the end of sentence
        if comment_tokens[-1] == ':':
            comment_tokens[-1] = '.'
        else:
            comment_tokens.append('.')

        ########################################################################################
        comment_tokens = ' '.join(comment_tokens)
        comment_tokens = re.sub(r'[\-|\*|\=|\~]{2,}', ' ', comment_tokens)  # remove ----+/****+/====+,
        comment_tokens = re.sub(r'[!]{2,}', '!', comment_tokens)  # change !!!! -> !
        comment_tokens = re.sub(r'[`]{2,}', ' ` ', comment_tokens)  # change ```, -> ` ,

        ########################################################################################
        # for rouge
        # remove <**> </**>
        comment_tokens = re.sub(r'<.*?>', '', comment_tokens)
        # remove =>
        comment_tokens = comment_tokens.replace('= >', ' ').replace('=>', ' ')
        # remove < >
        comment_tokens = comment_tokens.replace('<', ' ').replace('>', ' ')

        ########################################################################################
        comment_tokens = re.sub(r'\s+', ' ', comment_tokens)
        comment_tokens = comment_tokens.split(' ')

        new_comment_tokens = []
        for token in comment_tokens:
            token = token.strip()
            if len(token) > 0:
                # rule 1: +XX+ -> XX
                if token[0] == '+' or token[-1] == '+':
                    new_comment_tokens.append(token[1:-1].strip())
                else:
                    new_comment_tokens.append(token.strip())
        comment_tokens = new_comment_tokens

        if 3 < len(comment_tokens) <= 60:
            return comment_tokens
        else:
            return None

    def subcode(self, start: Tuple, end: Tuple, code_lines: List) -> str:
        '''
        extract substring from code lines
        :param start: start point
        :param end: end point
        :param code_lines: codes.split('\n')
        :return: substring of code
        '''
        if start[0] == end[0]:
            return code_lines[start[0]][start[1]:end[1]]
        elif start[0] < end[0]:
            sub_code_lines = [code_lines[start[0]][start[1]:]]
            for line_num in range(start[0] + 1, end[0]):
                sub_code_lines.append(code_lines[line_num])
            sub_code_lines.append(code_lines[end[0]][:end[1]])
            return b'\n'.join(sub_code_lines)
        else:
            raise NotImplemented

    def define_node_type(self, token: str) -> str:
        '''
        in tree_sitter library, operator and keyword nodes are no pre-define node type, like:
        [type: 'def'/'&&', value: 'def'/'&&']
        :param token: node value
        :return: if token is operator, its type will be EN name
                  if token is keyword, its type will be {}Kw
        '''
        is_keyword = True
        for chr in token:
            if str.isalpha(chr):
                continue
            else:
                is_keyword = False
                break
        if is_keyword:
            return '{}_keyword'.format(str.lower(token))
        else:
            if self.operators and (token in self.operators):
                return self.operators[token]
            else:
                return token

    def build_tree(self, root, code_lines: List[str]) -> Dict:
        '''
        build ast with tree_sitter, operator and keyword has no pre-defined type
        :param root: ast tree root node
        :param code_lines: [...], ...
        :return:
            [
                {'type': "node_type", 'children': "node_ids(List)", ‘children’: “node_ids(List)” }, # non-leaf node
                {'type': "node_type", 'children': "node_ids(List)", ‘value’: “node_value(str)” }, # leaf node
                ...
            ]
        '''
        ast_tree = {}

        def dfs(cur_node, parent_node_idx):
            if len(cur_node.children) == 0:
                # current node has no child node, it's leaf node, build a leaf node
                new_node_idx = len(ast_tree)
                if cur_node.is_named:
                    # leaf node's value is None. we have to extract its value from source code
                    value = self.subcode(cur_node.start_point, cur_node.end_point, code_lines).decode()
                    if not value:  # value='', delete current node
                        return
                    ast_tree[new_node_idx] = {
                        'type': cur_node.type, 'parent': parent_node_idx,
                        'value': value,
                    }
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
                else:
                    # leaf node is operator or keyword
                    ast_tree[new_node_idx] = {
                        'type': self.define_node_type(cur_node.type),
                        'parent': parent_node_idx,
                        'value': cur_node.type,
                    }
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
            else:
                # current node has many child nodes
                cur_node_idx = len(ast_tree)
                ast_tree[cur_node_idx] = {'type': cur_node.type, 'parent': parent_node_idx, 'children': []}
                # update parent node's children
                if parent_node_idx is not None:
                    ast_tree[parent_node_idx]['children'].append(cur_node_idx)
                # update current node's children
                for child_node in cur_node.children:
                    dfs(child_node, parent_node_idx=cur_node_idx)

        dfs(root, parent_node_idx=None)
        return ast_tree

    def parse_raw_ast(self, code: str, ) -> Optional[Dict]:
        # must add this head for php code
        if self.LANGUAGE == 'php':
            code = '<?php ' + code

        ast_tree = self.parser.parse(code.encode())

        code_lines = [line.encode() for line in code.split('\n')]

        # 1) build ast tree in Dict type
        try:
            code_tree = self.build_tree(ast_tree.root_node, code_lines)
            # if str.lower(code_tree[0]['type']) == 'error':
            #     raise RuntimeError
            if self.LANGUAGE == 'php':
                """
                first 3 nodes would be follow:
                0: {'type': 'program', 'parent': None, 'children': [1, 2, 6]}
                1: {'type': 'php_tag', 'parent': 0, 'value': '<?php'}
                2: {'type': 'ERROR', 'parent': 0, 'children': [3, 5]}
                solution: remove 2nd, connect 3rd to 1st, rename 3rd node's type to ‘local_variable_declaration’
                """
                php_tag_node = code_tree.pop(1)
                del code_tree[0]['children'][code_tree[0]['children'].index(1)]
                code_tree[2]['type'] = 'local_variable_declaration'
                # node index: from 2-..., should move forward and update children info
                code_tree[0]['children'] = [index - 1 for index in code_tree[0]['children']]
                for idx in sorted(code_tree.keys())[1:]:
                    new_idx = idx - 1
                    new_node = code_tree.pop(idx)
                    if new_node['parent'] > 1:
                        new_node['parent'] = new_node['parent'] - 1
                    if 'children' in new_node:
                        new_node['children'] = [index - 1 for index in new_node['children'] if index > 0]
                    code_tree[new_idx] = new_node

            assert len(code_tree) > 1, AssertionError('AST parsed error.')
            # check whether an ast contains nodes with null children
            for node in code_tree.values():
                if 'children' in node:
                    assert len(node['children']) > 0, AssertionError('AST has a node without child and value')
                if 'value' in node:
                    assert len(node['value']) > 0, AssertionError('AST has a node without child and value')
            return code_tree
        except RecursionError as err:
            # RecursionError: maximum recursion depth exceeded while getting the str of an object
            print(err)
            # raw_ast is too large, skip this ast
            return None
        except AssertionError as err:
            print(err)
            return None

    def parse_method(self, func_name: str, ) -> Optional[List[str]]:
        # our defined method parse function
        method = ''
        for char in func_name:
            if str.isalpha(char) or str.isdigit(char):
                method += char
            else:
                method += ' '
        method = self.parse_docstring(method)
        method = [token.strip() for token in method if len(token.strip()) > 1]
        if len(method) > 0:
            return method
        else:
            return [constants.NO_METHOD]

    def parse_code_tokens(self, code_tokens: List[str], ) -> Union[None, List[str]]:
        '''
        our refined code tokens parse function. if you prefer original code tokens, use code_tokens instead
        '''
        # skip this code snippet, if there are non-ascii tokens or code token is too long
        # filter comment in code_tokens, eg. //***\n /* */\n
        code_tokens = [
            token for token in code_tokens
            if not (str.startswith(token, '//') or str.startswith(token, '#') or \
                    (str.startswith(token, '/*') and str.endswith(token, '*/')))
        ]

        for idx, token in enumerate(code_tokens):
            code_tokens[idx] = token.strip()
            if not util.is_ascii(code_tokens[idx]) or len(code_tokens[idx]) > constants.MAX_CODE_TOKEN_LEN:
                return None

        code_tokens = util.filter_tokens(code_tokens)
        if self.to_lower:
            code_tokens = util.lower(code_tokens)

        if len(code_tokens) > 3:
            return code_tokens
        else:
            return None


if __name__ == '__main__':
    # unittest
    import gzip
    from dataset.utils.ast import tranv_trans

    lang = 'ruby'
    so_file = '/home/yang/.ncc/code_search_net/libs/{}.so'.format(lang)
    parser = CodeParser(so_file, lang, to_lower=False, operators_file='operators.json')

    while True:
        # code = "# addition operator\ndef add(a, b):\n\treturn a+b".strip()
        # code = "def add(a, b):\n\treturn a+b".strip()
        code = "def merge(tab)\n      raise ArgumentError, 'Only tables could be merged' unless tab.is_a?(self.class)\n      raise ArgumentError, 'Primary keys not the same' unless tab.primary_key == primary_key\n\n      # Empty content\n      content = []\n      row_keys = (@row_keys.keys | tab.row_keys).map.with_index { |row, i| [row, i] }.to_h\n      col_keys = (@col_keys.keys | tab.col_keys).map.with_index { |col, i| [col, i] }.to_h\n      row_keys.size.times { content << Array.new(col_keys.size, '') }\n\n      # rubocop:disable Lint\/Eval, Style\/SpaceInsideStringInterpolation\n\n      # Fill content with self\n      eval <<-END_OF_DOC\n        @row_keys.each do |row, old_ri|\n          new_ri = row_keys[row]\n          #{\n            str = []\n            @col_keys.map do |col, old_ci|\n              new_ci = col_keys[col]\n              str << \"content[new_ri][#{new_ci}] = @content[old_ri][#{old_ci}]\"\n            end\n            str.join(\"\\n\" + ' ' * 8)\n          }\n        end\n      END_OF_DOC\n\n      # Fill content with tab\n      @content_merged_with = tab.instance_variable_get(:@content)\n      eval <<-END_OF_DOC\n        tab.row_keys.each_with_index do |row, old_ri|\n          new_ri = row_keys[row]\n          #{\n            str = []\n            tab.col_keys.each_with_index do |col, old_ci|\n              new_ci = col_keys[col]\n              str << \"content[new_ri][#{new_ci}] = @content_merged_with[old_ri][#{old_ci}]\"\n            end\n            str.join(\"\\n\" + ' ' * 8)\n          }\n        end\n      END_OF_DOC\n\n      # rubocop:enable Lint\/Eval, Style\/SpaceInsideStringInterpolation, Lint\/UselessAssignment\n\n      # Create a new table\n      self.class.build(\n        primary_key: primary_key,\n        row_keys: row_keys,\n        col_keys: col_keys,\n        content: content,\n        comments: comments + tab.comments\n      )\n    end"
        raw_ast = parser.parse_raw_ast(code)

        # raw_ast = '{"0":{"type":"program","parent":null,"children":[1]},"1":{"type":"method","parent":0,"children":[2,3,4,8,20,26,28,36]},"2":{"type":"def_keyword","parent":1,"value":"def"},"3":{"type":"identifier","parent":1,"value":"set"},"4":{"type":"method_parameters","parent":1,"children":[5,6,7]},"5":{"type":"LeftParenOp","parent":4,"value":"("},"6":{"type":"identifier","parent":4,"value":"set_attributes"},"7":{"type":"LeftParenOp","parent":4,"value":")"},"8":{"type":"assignment","parent":1,"children":[9,10,11]},"9":{"type":"identifier","parent":8,"value":"old_attributes"},"10":{"type":"AsgnOp","parent":8,"value":"="},"11":{"type":"method_call","parent":8,"children":[12,13]},"12":{"type":"identifier","parent":11,"value":"compute_attributes"},"13":{"type":"argument_list","parent":11,"children":[14,15,19]},"14":{"type":"LeftParenOp","parent":13,"value":"("},"15":{"type":"call","parent":13,"children":[16,17,18]},"16":{"type":"identifier","parent":15,"value":"set_attributes"},"17":{"type":"DotOp","parent":15,"value":"."},"18":{"type":"identifier","parent":15,"value":"keys"},"19":{"type":"LeftParenOp","parent":13,"value":")"},"20":{"type":"method_call","parent":1,"children":[21,22]},"21":{"type":"identifier","parent":20,"value":"assign_attributes"},"22":{"type":"argument_list","parent":20,"children":[23,24,25]},"23":{"type":"LeftParenOp","parent":22,"value":"("},"24":{"type":"identifier","parent":22,"value":"set_attributes"},"25":{"type":"LeftParenOp","parent":22,"value":")"},"26":{"type":"yield","parent":1,"children":[27]},"27":{"type":"yield_keyword","parent":26,"value":"yield"},"28":{"type":"ensure","parent":1,"children":[29,30]},"29":{"type":"ensure_keyword","parent":28,"value":"ensure"},"30":{"type":"method_call","parent":28,"children":[31,32]},"31":{"type":"identifier","parent":30,"value":"assign_attributes"},"32":{"type":"argument_list","parent":30,"children":[33,34,35]},"33":{"type":"LeftParenOp","parent":32,"value":"("},"34":{"type":"identifier","parent":32,"value":"old_attributes"},"35":{"type":"LeftParenOp","parent":32,"value":")"},"36":{"type":"end_keyword","parent":1,"value":"end"}}'
        # raw_ast = ujson.loads(raw_ast)
        ast = util_ast.convert(raw_ast)

        max_len = 10
        masks = tranv_trans.get_rel_masks(raw_ast, max_len)
        masks = tranv_trans.separate_rel_mask(masks, max_len)
        sep_asts = util_ast.separate_ast(raw_ast, max_len)

        for sep_ast, ext in sep_asts:
            if len(sep_ast) > 1:
                [util_ast.dfs_traversal(sep_ast), ext]

    # raw_file = '/home/yang/.ncc/CodeSearchNet/raw_unzip/{}/{}_test_0.jsonl.gz'.format(lang, lang)
    # with gzip.GzipFile(raw_file, 'r') as reader:
    #     for line in reader:
    #         if line:
    #             line = ujson.loads(line)
    #             parser.parse_raw_ast(line['code'])
