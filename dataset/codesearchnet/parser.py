# -*- coding: utf-8 -*-

from typing import *

import sys

import re
import itertools
from copy import deepcopy
from tree_sitter import Language, Parser

from dataset.codesearchnet.utils import constants
from dataset.codesearchnet.utils import util
from dataset.codesearchnet.utils import util_ast

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(constants.RECURSION_DEPTH)  # recursion depth


class CodeParser(object):
    '''parse code data into ast'''
    __slots__ = ('parser', 'to_lower', 'LANGUAGE',)

    def __init__(self, SO_FILE: str, LANGUAGE: str, to_lower: bool = True, ):
        self.parser = Parser()
        self.parser.set_language(Language(SO_FILE, LANGUAGE))
        self.LANGUAGE = LANGUAGE
        self.to_lower = to_lower

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
            return '\n'.join(sub_code_lines)
        else:
            raise NotImplemented

    def define_node_type(self, token: str) -> str:
        '''
        in tree_sitter library, operator and keyword nodes are no pre-define node type, like:
        [type: 'def'/'&&', value: 'def'/'&&']
        :param token: node value
        :return: if token is operator, its type will be {}_operator
                  if token is keyword, its type will be {}_keyword
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
            return '{}_operator'.format(token)

    def build_tree(self, root, code_lines: List[str]) -> Dict:
        '''
        build ast with tree_sitter, operator and keyword has no pre-defined type
        :param root: ast tree root node
        :param code_lines: [...], ...
        :return: {1*NODEFI1: {'node': 'XX', 'parent': 'None', 'children': [XX, ...]}}
        '''
        ast_tree = {}

        def dfs(cur_node, parent_node_ind):
            if len(cur_node.children) == 0:
                # current node has no child node, it's leaf node, build a leaf node
                if cur_node.is_named:
                    # leaf node's value is None. we have to extract its value from source code
                    token_name = self.subcode(cur_node.start_point, cur_node.end_point, code_lines)
                    new_node = {
                        'node': cur_node.type,
                        'parent': parent_node_ind,
                        'children': [token_name.strip()],
                    }
                    new_node_ind = constants.NODE_FIX + str(len(ast_tree) + 1)
                    ast_tree[new_node_ind] = new_node
                    ast_tree[parent_node_ind]['children'].append(new_node_ind)
                else:
                    # leaf node is operator or keyword
                    new_node = {
                        'node': self.define_node_type(cur_node.type),
                        'parent': parent_node_ind,
                        'children': [cur_node.type],
                    }
                    new_node_ind = constants.NODE_FIX + str(len(ast_tree) + 1)
                    ast_tree[new_node_ind] = new_node
                    ast_tree[parent_node_ind]['children'].append(new_node_ind)
            else:
                # current node has many child nodes
                node_ind = constants.NODE_FIX + str(len(ast_tree) + 1)  # root node index
                node = {
                    'node': cur_node.type,
                    'parent': parent_node_ind,
                    'children': [],
                }
                ast_tree[node_ind] = node
                # update parent node's child nodes
                if parent_node_ind is None:
                    pass
                else:
                    ast_tree[parent_node_ind]['children'].append(node_ind)

                for child_node in cur_node.children:
                    dfs(child_node, parent_node_ind=node_ind)

        dfs(root, parent_node_ind=None)
        return ast_tree

    def delete_comment_node(self, ast_tree: Dict) -> Dict:
        '''delete comment node and its children'''

        def delete_cur_node(node_ind, cur_node):
            # update its parent's children
            parent_ind = cur_node['parent']
            parent_node = ast_tree[parent_ind]
            del_ind = parent_node['children'].index(node_ind)
            del parent_node['children'][del_ind]
            # delete node
            ast_tree.pop(node_ind)
            return parent_ind, parent_node

        def dfs(node_ind):
            cur_node = ast_tree[node_ind]
            child_node_indices = util.get_tree_children_func(cur_node)

            if cur_node['node'] == 'comment':
                node_ind, cur_node = delete_cur_node(node_ind, cur_node)

                while len(cur_node['children']) == 0:
                    node_ind, cur_node = delete_cur_node(node_ind, cur_node)

            if len(child_node_indices) == 0:
                return

            for child_name in child_node_indices:
                dfs(child_name)

        dfs(constants.ROOT_NODE_NAME)
        return ast_tree

    def parse_raw_ast(self, code: str, ) -> Optional[Dict]:
        # must add this head for php code
        if self.LANGUAGE == 'php':
            code = '<?php ' + code

        ast_tree = self.parser.parse(
            bytes(code.replace('\t', '    ').replace('\n', ' ').strip(), "utf8")
        )

        code_lines = code.split('\n')  # raw code
        # 1) build ast tree in Dict type
        try:
            code_tree = self.build_tree(ast_tree.root_node, code_lines)
        except RecursionError as err:
            # RecursionError: maximum recursion depth exceeded while getting the str of an object
            print(err)
            # raw_ast is too large, skip this ast
            return None

        # 2) delete comment node
        code_tree = self.delete_comment_node(code_tree)
        # 3) pop head node which has only 1 child
        # because in such way, head node might be Program/Function/Error and its child is the code's AST
        for ind in range(1, 1 + len(code_tree)):
            cur_node_id = constants.NODE_FIX + str(ind)
            if (code_tree[cur_node_id]['parent'] is None) and \
                    len(code_tree[cur_node_id]['children']) == 1 and \
                    code_tree[constants.NODE_FIX + str(ind)]['children'][0].startswith(constants.NODE_FIX):
                child_node = code_tree[cur_node_id]['children'][0]
                code_tree[child_node]['parent'] = None
                code_tree.pop(cur_node_id)
            else:
                break
        if len(code_tree) == 0:
            return None
        # 4) reset tree indices
        raw_ast = util_ast.reset_indices(code_tree)  # reset node indices
        return raw_ast

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

        for ind, token in enumerate(code_tokens):
            code_tokens[ind] = token.strip()
            if not util.is_ascii(code_tokens[ind]) or len(code_tokens[ind]) > constants.MAX_CODE_TOKEN_LEN:
                return None

        code_tokens = util.filter_tokens(code_tokens)
        if self.to_lower:
            code_tokens = util.lower(code_tokens)

        if len(code_tokens) > 3:
            return code_tokens
        else:
            return None

    def parse(self, code_snippet: Dict, ) -> Optional[Dict]:
        # code snippets with seq/tree modalities
        new_code_snippet = dict(tok=None, raw_ast=None, comment=None, func_name=None, )
        new_code_snippet['tok'] = self.parse_code_tokens(code_snippet['code_snippet'])
        new_code_snippet['raw_ast'] = self.parse_raw_ast(code_snippet['code'])
        new_code_snippet['comment'] = self.parse_comment(code_snippet['docstring'], code_snippet['docstring_tokens'])
        new_code_snippet['func_name'] = self.parse_method(code_snippet['func_name'])
        return new_code_snippet
