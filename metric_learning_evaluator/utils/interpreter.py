""" SQI - Simple Query Interpreter
  The interpreter is designed to processing metric learning query commands

  Each `symbol` inside comands will be translated to list in the interpreter.

  The major reference of these source codes originate from the following: https://ruslanspivak.com/lsbasi-part1/
  @kv
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
# legacy
INTEGER, PLUS, MINUS, MUL, DIV, = (
    'INTEGER', 'PLUS', 'MINUS', 'MUL', 'DIV',
)
# Keywords used in evaluator
AND = 'AND'
NOT = 'NOT'
OR = 'OR'
XOR = 'XOR'
SYMBOL = 'SYMBOL'
LPAREN = '('
RPAREN = ')'
EOF = 'EOF'


class CommandExecutor(object):
    def __init__(self, dataframe):
        """Dataframe query command executor
          Args:
            dataframe: Pandas DataFrame
        """

        self.dataframe = dataframe

    def execute(self, command):
        lexer = Lexer(command)
        parser = Parser(lexer)
        interpreter = Interpreter(parser, self.dataframe)
        return interpreter.interpret()


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(SYMBOL, 'SU.seen') -> convert to []
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.word_buffer = ''

    def error(self):
        raise Exception('Invalid character')

    def forward(self):
        """Advance the `pos` pointer to the next symbol.
        """
        # clear buffer
        self.word_buffer = ''
        while self.current_char is not None and \
            (self.current_char.isalpha()
                or self.current_char == '.'
                or self.current_char == '_'):
            self.pos += 1
            self.word_buffer += self.current_char
            if self.pos >= len(self.text):
                self.current_char = None  # Indicates end of input
            else:
                # next char
                self.current_char = self.text[self.pos]

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                self.forward()
                return Token(SYMBOL, self.word_buffer)

            if self.current_char == '+':
                self.advance()
                return Token(OR, '+')

            if self.current_char == '^':
                self.advance()
                return Token(XOR, '^')

            if self.current_char == '&':
                self.advance()
                return Token(AND, '&')

            if self.current_char == '~':
                self.advance()
                return Token(NOT, '~')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Symbol(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


# Interpreter
class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        """factor : (PLUS | MINUS) factor | INTEGER | LPAREN expr RPAREN"""
        token = self.current_token
        if token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        # NOTE: where we can modified
        elif token.type == SYMBOL:
            self.eat(SYMBOL)
            return Symbol(token)
        elif token.type == AND:
            self.eat(AND)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == OR:
            self.eat(OR)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == XOR:
            self.eat(XOR)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == NOT:
            self.eat(NOT)
            node = UnaryOp(token, self.factor())
            return node

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (AND, XOR, OR):
            token = self.current_token
            if token.type == AND:
                self.eat(AND)
            elif token.type == XOR:
                self.eat(XOR)
            elif token.type == OR:
                self.eat(OR)

            node = BinOp(left=node, op=token, right=self.factor())
        return node

    def expr(self):
        """
        expr   : term ((PLUS | MINUS) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : (PLUS | MINUS) factor | INTEGER | LPAREN expr RPAREN
        """
        node = self.term()

        while self.current_token.type in (NOT):
            token = self.current_token
            if token.type == NOT:
                self.eat(NOT)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        node = self.expr()
        if self.current_token.type != EOF:
            self.error()
        return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node, cond=None):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        if cond is not None:
            return visitor(node, cond)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


# Executor
class Interpreter(NodeVisitor):
    def __init__(self, parser, dataframe):
        """
          Args:
            parser
            dataframe
        """
        self.parser = parser
        self.dataframe = dataframe

    def visit_BinOp(self, node):
        """Binary operation
            - PLUS (+)
            - AND (&)
        """
        if node.op.type == OR:
            # TODO: list operation
            left_arr = self.visit(node.left)
            right_arr = self.visit(node.right)
            return left_arr + list(set(right_arr) - set(left_arr))
        elif node.op.type == AND:
            # TODO:
            left_arr = self.visit(node.left)
            right_arr = self.visit(node.right)
            return list(set(left_arr) & set(right_arr))
        elif node.op.type == XOR:
            # TODO:
            left_arr = self.visit(node.left)
            right_arr = self.visit(node.right)
            return list(set(left_arr) ^ set(right_arr))

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == NOT:
            return self.visit(node.expr, 'not')

    def visit_Symbol(self, node, query_mode='direct'):
        """Visit `Symbol` Token and query list from given df
            node.value is `supercategory_name.菓子`
            we should query df @here (?)
            query_mode:
                - direct: query with given key, value
                - not: query all other values except the given one
                - all
        """
        _return_column = 'instance_id'

        def _query_key_value_command(key, val, return_column='instance_id'):
            """
            Args:
                cmd: string, format = item_key.item_value
            NOTE:
              if val is nan would be dropped
            """
            return self.dataframe.query('{}==\'{}\''.format(key, val))[return_column].tolist()
        operand_symbol = node.value
        ret_list = []
        if query_mode == 'direct':
            key, val = operand_symbol.split('.')
            ret_list.extend(_query_key_value_command(key, val, _return_column))
        if query_mode == 'not':
            key, val = operand_symbol.split('.')
            others = [other for other in self.dataframe[key].unique()
                      if other != val]
            for other_val in others:
                ret_list.extend(_query_key_value_command(key, other_val, _return_column))
        if query_mode == 'all':
            ret_list.extend(
                self.dataframe[_return_column].tolist())
        return ret_list

    def interpret(self):
        tree = self.parser.parse()
        if tree is None:
            return []
        return self.visit(tree)
