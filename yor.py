#!/usr/bin/env python3

import os
import sys 
import subprocess
from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum, IntEnum, auto

YOR_PROGRAM_NAME= "yor"
YOR_VERSION = "Yor - 0.0.1"
YOR_EXTENSION = ".yor" 
YOR_DEBUG = True
YOR_MEM_CAPACITY = 640_000
YOR_HOST_PLATFORM = "void" # "void" | "linux" | "win32"
YOR_SILENT = False
YOR_INCLUDE_DIRS: List[str] = []

if sys.platform == "linux" or sys.platform == "linux2":
    YOR_HOST_PLATFORM = "linux"
elif sys.platform == "win32":
    YOR_HOST_PLATFORM = "win32"
else:
    YOR_HOST_PLATFORM = "void"

class Intrinsic(Enum):
    DUP  = auto()
    SWAP = auto()
    DROP = auto()
    OVER = auto()
    ROT  = auto()
    ADD  = auto()
    SUB  = auto()
    DIVMOD = auto()
    EQ   = auto()
    NE   = auto()
    GT   = auto()
    GE   = auto()
    LT   = auto()
    LE   = auto()
    BSL  = auto()
    BSR  = auto()
    MEM = auto()
    LOAD8 = auto()
    STORE8 = auto()
    LOAD64 = auto()
    STORE64 = auto()
    LINUX_SYSCALL0 = auto()
    LINUX_SYSCALL1 = auto()
    LINUX_SYSCALL2 = auto()
    LINUX_SYSCALL3 = auto()
    LINUX_SYSCALL4 = auto()
    LINUX_SYSCALL5 = auto()
    LINUX_SYSCALL6 = auto()

class Opcode(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    IF = auto()
    ELSE = auto()
    END = auto()
    WHILE = auto()
    DO = auto()
    INTRINSIC = auto()
    MEMORY_DEF = auto()
    MEMORY_REFER = auto()

Loc = Tuple[str, int, int] # file_path, row, col

class TokenKind(Enum):
    SYMBOL = auto()
    INT = auto()
    STRING = auto()

    # This should be actually an expression rather than a token
    MEMORY_DEF = auto()
    MEMORY_REFER = auto()

@dataclass
class Token:
    loc: Loc
    value: str|Tuple[str, int]
    kind: TokenKind

@dataclass
class Operation:
    kind: Opcode
    token: Token
    operand: int|str|Intrinsic|Tuple[str, int]

class DataType(IntEnum):
    INT = auto()
    PTR = auto()
    BOOL = auto()
    LABEL = auto()

def fatal(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)
    sys.exit(1)

def compilation_trap(loc: Loc, *args: str):
    print("Compilation error at line %d pos %d in `%s`:" % (loc[1], loc[2], loc[0]), file=sys.stderr)
    fatal(*["    -> %s" %arg for arg in args])

def lex_advance(source: str, i: int, row: int, col: int) -> Tuple[int, int, int]:
    if source[i] == "\n":
        row += 1
        col = -1
    i += 1
    col += 1
    return i, row, col

def translate_unescaped_str(s: str) -> str:
    s_len = len(s)
    result = ""
    i = 0

    while i < s_len:
        if s[i] == '\\':
            if i + 1 < s_len:
                if s[i + 1] == 'n':
                    result += '\n'
                    i += 2
                elif s[i + 1] == 't':
                    result += '\t'
                    i += 2
                elif s[i + 1] == '0':
                    result += "\x00"
                    i += 2
                elif s[i + 1] == '\\':
                    result += '\\'
                    i += 2
                else:
                    result += s[i]  # Treat the backslash as a literal character
                    i += 1
            else:
                # If the string ends with a backslash, treat it as a literal backslash
                result += '\\'
                i += 1
        else:
            result += s[i]
            i += 1

    return result

def lex_source(source_path: str, source: str) -> List[Token]:
    i = 0
    row = 0
    col =  0
    result: List[Token] = []
    source_len = len(source)

    while i < source_len:
        while i < source_len and source[i].isspace():
            i, row, col = lex_advance(source, i, row, col)
        if i >= source_len:
            break

        if source[i].isdigit():
            start = i
            while i < source_len and source[i].isdigit():
                i, row, col = lex_advance(source, i, row, col)
            result.append(Token((source_path, row + 1, col + 1), source[start:i], TokenKind.INT))
            if i < source_len and not source[i].isspace():
                compilation_trap((source_path, row + 1, col + 1), "Expecting whitespace after a number")
        elif source[i] == '/' and i + 1 < source_len and source[i + 1] == '/':
            while i < source_len and source[i] != '\n':
                i, row, col = lex_advance(source, i, row, col)
            i, row, col = lex_advance(source, i, row, col)
        elif source[i] == "\"":
            i, row, col = lex_advance(source, i, row, col)
            start = i
            while i < source_len and source[i] != "\"":
                i, row, col = lex_advance(source, i, row, col)
            result.append(Token((source_path, row + 1, col + 1), translate_unescaped_str(source[start:i]), TokenKind.STRING))
            i, row, col = lex_advance(source, i, row, col)
            if i < source_len and not source[i].isspace():
                compilation_trap((source_path, row + 1, col + 1), "Expecting whitespace after a string")
        else:
            start = i
            while i < source_len and not (source[i].isspace()):
                i, row, col = lex_advance(source, i, row, col)
            symbol = source[start:i]
            result.append(Token((source_path, row + 1, col + 1), symbol, TokenKind.SYMBOL))

    return result

MAP_OF_INTRINSIC_SYMBOLS_AND_INSTRINSICS = {
    "+": Intrinsic.ADD,
    "-": Intrinsic.SUB,
    "divmod": Intrinsic.DIVMOD,
    "=": Intrinsic.EQ,
    "!=": Intrinsic.NE,
    "<": Intrinsic.LT,
    "<=": Intrinsic.LE,
    ">": Intrinsic.GT,
    ">=": Intrinsic.GE,

    ">>": Intrinsic.BSR,
    "<<": Intrinsic.BSL,

    "mem": Intrinsic.MEM,
    "!8": Intrinsic.STORE8,
    "?8": Intrinsic.LOAD8,
    "!64": Intrinsic.STORE64,
    "?64": Intrinsic.LOAD64,

    "dup": Intrinsic.DUP,
    "drop": Intrinsic.DROP,
    "over": Intrinsic.OVER,
    "swap": Intrinsic.SWAP,
    "rot": Intrinsic.ROT,

    "linux-syscall-0": Intrinsic.LINUX_SYSCALL1,
    "linux-syscall-1": Intrinsic.LINUX_SYSCALL1,
    "linux-syscall-2": Intrinsic.LINUX_SYSCALL2,
    "linux-syscall-3": Intrinsic.LINUX_SYSCALL3,
    "linux-syscall-4": Intrinsic.LINUX_SYSCALL4,
    "linux-syscall-5": Intrinsic.LINUX_SYSCALL5,
    "linux-syscall-6": Intrinsic.LINUX_SYSCALL6,
}

MAP_OF_KEYWORD_SYMBOLS_AND_OPKINDS = {
    "if": Opcode.IF,
    "else": Opcode.ELSE,
    "while": Opcode.WHILE,
    "do": Opcode.DO,
    "end": Opcode.END,
}

LIST_OF_KEYWORDS = [ "if", "else", "while", "do", "end", "memory", "def" ]

LIST_OF_LINUX_ONLY_SYMBOLS = [ 
      "linux-syscall-0",
      "linux-syscall-1",
      "linux-syscall-2",
      "linux-syscall-3",
      "linux-syscall-4",
      "linux-syscall-5",
      "linux-syscall-6",
]

def compile_token_to_op(token: Token) -> Operation:
    assert len(Opcode) == 10, "There's unhandled ops in `translate_token()`"
    assert len(Intrinsic) == 28, "There's unhandled intrinsic in `translate_token()`"
    if token.kind == TokenKind.SYMBOL:
        assert isinstance(token.value, str), "Invalid type of token.value in compile_token_to_op()"
        if token.value in MAP_OF_INTRINSIC_SYMBOLS_AND_INSTRINSICS.keys():
            if YOR_HOST_PLATFORM != "linux" and token.value in LIST_OF_LINUX_ONLY_SYMBOLS:
                compilation_trap(token.loc, "Syntax `%s` is not supported on `%s` platform" % (token.value, YOR_HOST_PLATFORM))
            return Operation(Opcode.INTRINSIC, token, MAP_OF_INTRINSIC_SYMBOLS_AND_INSTRINSICS[token.value])
        elif token.value in MAP_OF_KEYWORD_SYMBOLS_AND_OPKINDS.keys():
            return Operation(MAP_OF_KEYWORD_SYMBOLS_AND_OPKINDS[token.value], token, -1)
        else:
            compilation_trap(token.loc, "Invalid syntax `%s`" % token.value)
    if token.kind == TokenKind.INT:
        assert isinstance(token.value, str), "Invalid type of token.value in compile_token_to_op()"
        return Operation(Opcode.PUSH_INT, token, int(token.value))
    elif token.kind == TokenKind.STRING:
        assert isinstance(token.value, str), "Invalid type of token.value in compile_token_to_op()"
        return Operation(Opcode.PUSH_STR, token, token.value)
    elif token.kind == TokenKind.MEMORY_DEF:
        assert isinstance(token.value, tuple), "Invalid type of token.value in compile_token_to_op()"
        if token.value[0] in LIST_OF_KEYWORDS or token.value[0] in MAP_OF_INTRINSIC_SYMBOLS_AND_INSTRINSICS:
            compilation_trap(token.loc, "Could not redefine this `%s` symbol as a memory label" % token.value[0])
        if not (token.value[0].isalnum() or '_' in token.value[0]):
            compilation_trap(token.loc, "Illegal name for memory definition only supporting alphabet, numeric and '_' characters")
        return Operation(Opcode.MEMORY_DEF, token, token.value)
    elif token.kind == TokenKind.MEMORY_REFER:
        assert isinstance(token.value, str), "Invalid type of token.value in compile_token_to_op()"
        return Operation(Opcode.MEMORY_REFER, token, token.value)
    else:
        compilation_trap(token.loc, "Unreachable token `%s` with type `%s` in translate_token()" % (token.value, token.kind))
        return Operation(Opcode.PUSH_INT, token, -1)

def data_type_as_str(t: DataType) -> str:
    if t == DataType.INT:
        return "int"
    elif t == DataType.PTR:
        return "ptr"
    elif t == DataType.BOOL:
        return "bool"
    else:
        assert False, "Invalid type"

def expect_data_type_stack_size(op: Operation, expected_data_type_stack_size: int, found_data_type_stack_size: int):
    if expected_data_type_stack_size > found_data_type_stack_size:
        compilation_trap(op.token.loc, "For `%s` operation expecting %d elements on the stack but found %d elements" % (
            op.token.value,
            expected_data_type_stack_size,
            found_data_type_stack_size,))

def invalid_type_trap(op: Operation, expecting: List[DataType], found: List[DataType]):
    expecting_str = " ".join([ "<%s>" % data_type_as_str(dt) for dt in expecting])
    found_str = " ".join([ "<%s>" % data_type_as_str(dt) for dt in found])
    compilation_trap(op.token.loc, "Expecting `%s %s` but found `%s %s`" % (
            expecting_str, op.token.value, found_str, op.token.value,
        ))

def type_check_program(program: List[Operation]):
    stack: List[Tuple[DataType, Loc]] = []

    for ip in range(len(program)):
        op = program[ip]
        assert len(Opcode) == 10, "There's unhandled ops in `translate_token()`"
        assert len(Intrinsic) == 28, "There's unhandled intrinsic in `translate_token()`"

        if op.kind == Opcode.PUSH_INT:
            stack.append((DataType.INT, op.token.loc))
        elif op.kind == Opcode.PUSH_STR:
            stack.append((DataType.INT, op.token.loc))
            stack.append((DataType.PTR, op.token.loc))
        elif op.kind == Opcode.MEMORY_REFER:
            stack.append((DataType.PTR, op.token.loc))
        elif op.kind == Opcode.MEMORY_DEF:
            pass
        elif op.kind == Opcode.IF:
            pass
        elif op.kind == Opcode.ELSE:
            pass
        elif op.kind == Opcode.END:
            pass
        elif op.kind == Opcode.WHILE:
            pass
        elif op.kind == Opcode.DO:
            expect_data_type_stack_size(op, 1, len(stack))
            a_type, a_loc = stack.pop()
            if a_type != DataType.BOOL:
                compilation_trap(a_loc, "Expecting argument of `do` operation to be `bool` but found %s" % a_type)

        elif op.kind == Opcode.INTRINSIC:
            if op.operand == Intrinsic.DUP:
                expect_data_type_stack_size(op, 1, len(stack))
                a_type, _ = stack.pop()
                stack.append((a_type, op.token.loc))
                stack.append((a_type, op.token.loc))
            elif op.operand == Intrinsic.SWAP:
                expect_data_type_stack_size(op, 2, len(stack))
                a_type, _ = stack.pop()
                b_type, _ = stack.pop()
                stack.append((a_type, op.token.loc))
                stack.append((b_type, op.token.loc))
            elif op.operand == Intrinsic.DROP:
                expect_data_type_stack_size(op, 1, len(stack))
                stack.pop()
            elif op.operand == Intrinsic.OVER:
                expect_data_type_stack_size(op, 2, len(stack))
                tmp_type, _ = stack[-2]
                stack.append((tmp_type, op.token.loc))
            elif op.operand == Intrinsic.ROT:
                compilation_trap(op.token.loc, "Not implemented")
            elif op.operand == Intrinsic.ADD:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token.loc))
                elif a_type == DataType.PTR and b_type == DataType.INT:
                    stack.append((DataType.PTR, op.token.loc))
                elif a_type == DataType.INT and b_type == DataType.PTR:
                    stack.append((DataType.PTR, op.token.loc))
                else:
                    compilation_trap(op.token.loc, "Expecting `+` operation arguments to be either `int` `int` or `int` `ptr`",
                                     "but found `%s` `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.SUB:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token.loc))
                elif a_type == DataType.PTR and b_type == DataType.INT:
                    stack.append((DataType.PTR, op.token.loc))
                elif a_type == DataType.INT and b_type == DataType.PTR:
                    stack.append((DataType.PTR, op.token.loc))
                else:
                    compilation_trap(op.token.loc, "Expecting `-` operation arguments to be either `int` `int` or `int` `ptr`",
                                     "but found `%s` `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.DIVMOD:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if a_type == DataType.INT and b_type == DataType.INT:
                    stack.append((DataType.INT, op.token.loc))
                    stack.append((DataType.INT, op.token.loc))
                else:
                    invalid_type_trap(op, [DataType.INT, DataType.INT], [a_type, b_type])
            elif op.operand == Intrinsic.EQ: 
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, _) = stack.pop()

                if (a_type == DataType.INT and b_type == DataType.INT) or (a_type == DataType.PTR and b_type == DataType.PTR):
                    stack.append((DataType.BOOL, op.token.loc))
                else:
                    compilation_trap(a_loc, "Expecting first and second argument of `=` operation to be the same type either `int` or `ptr`",
                                     "but found `%s`, `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.NE:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if (a_type == DataType.INT and b_type == DataType.INT) or (a_type == DataType.PTR and b_type == DataType.PTR):
                    stack.append((DataType.BOOL, op.token.loc))
                else:
                    compilation_trap(a_loc, "Expecting first and second argument of `!=` operation to be the same type either `int` or `ptr`",
                                     "but found `%s`, `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.GT:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if (a_type == DataType.INT and b_type == DataType.INT) or (a_type == DataType.PTR and b_type == DataType.PTR):
                    stack.append((DataType.BOOL, op.token.loc))
                else:
                    compilation_trap(a_loc, "Expecting first and second argument of `>` operation to be the same type either `int` or `ptr`",
                                     "but found `%s`, `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.GE:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if (a_type == DataType.INT and b_type == DataType.INT) or (a_type == DataType.PTR and b_type == DataType.PTR):
                    stack.append((DataType.BOOL, op.token.loc))
                else:
                    compilation_trap(a_loc, "Expecting first and second argument of `>=` operation to be the same type either `int` or `ptr`",
                                     "but found `%s`, `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.LT:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if (a_type == DataType.INT and b_type == DataType.INT) or (a_type == DataType.PTR and b_type == DataType.PTR):
                    stack.append((DataType.BOOL, op.token.loc))
                else:
                    compilation_trap(a_loc, "Expecting first and second argument of `<` operation to be the same type either `int` or `ptr`",
                                     "but found `%s`, `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.LE:
                expect_data_type_stack_size(op, 2, len(stack))
                (a_type, a_loc) = stack.pop()
                (b_type, b_loc) = stack.pop()
                if (a_type == DataType.INT and b_type == DataType.INT) or (a_type == DataType.PTR and b_type == DataType.PTR):
                    stack.append((DataType.BOOL, op.token.loc))
                else:
                    compilation_trap(a_loc, "Expecting first and second argument of `<=` operation to be the same type either `int` or `ptr`",
                                     "but found `%s`, `%s`" % (a_type, b_type))
            elif op.operand == Intrinsic.BSL:
                compilation_trap(op.token.loc, "Not implemented")
            elif op.operand == Intrinsic.BSR:
                compilation_trap(op.token.loc, "Not implemented")

            elif op.operand == Intrinsic.MEM:
                stack.append((DataType.PTR, op.token.loc))
            elif op.operand == Intrinsic.LOAD8:
                expect_data_type_stack_size(op, 1, len(stack))
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    invalid_type_trap(op, [DataType.PTR], [a_type])
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.STORE8:
                expect_data_type_stack_size(op, 2, len(stack))
                a_type, a_loc = stack.pop()
                b_type, _ = stack.pop()
                if a_type == DataType.PTR and b_type == DataType.INT:
                    invalid_type_trap(op, [DataType.PTR, DataType.INT], [a_type, b_type])
            elif op.operand == Intrinsic.LOAD64:
                expect_data_type_stack_size(op, 1, len(stack))
                a_type, a_loc = stack.pop()
                if a_type != DataType.PTR:
                    invalid_type_trap(op, [DataType.PTR], [a_type])
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.STORE64:
                expect_data_type_stack_size(op, 2, len(stack))
                a_type, a_loc = stack.pop()
                b_type, _ = stack.pop()
                if a_type == DataType.PTR and b_type == DataType.INT:
                    invalid_type_trap(op, [DataType.PTR, DataType.INT], [a_type, b_type])

            elif op.operand == Intrinsic.LINUX_SYSCALL0:
                expect_data_type_stack_size(op, 1, len(stack))
                stack.pop()
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.LINUX_SYSCALL1:
                expect_data_type_stack_size(op, 2, len(stack))
                stack.pop()
                stack.pop()
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.LINUX_SYSCALL2:
                expect_data_type_stack_size(op, 3, len(stack))
                for _ in range(3):
                    stack.pop()
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.LINUX_SYSCALL3:
                expect_data_type_stack_size(op, 4, len(stack))
                for _ in range(4):
                    stack.pop()
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.LINUX_SYSCALL4:
                expect_data_type_stack_size(op, 5, len(stack))
                for _ in range(5):
                    stack.pop()
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.LINUX_SYSCALL5:
                expect_data_type_stack_size(op, 6, len(stack))
                for _ in range(6):
                    stack.pop()
                stack.append((DataType.INT, op.token.loc))
            elif op.operand == Intrinsic.LINUX_SYSCALL6:
                expect_data_type_stack_size(op, 7, len(stack))
                for _ in range(7):
                    stack.pop()
                stack.append((DataType.INT, op.token.loc))
            else:
                compilation_trap(op.token.loc, "Invalid intrinsic operation")
        else:
            compilation_trap(op.token.loc, "Invalid operation")

    if len(stack) > 0:
        compilation_trap(stack.pop()[1], "There's %d unhandled data in the stack" % (len(stack) + 1))

def compile_tokens_to_program(tokens: List[Token]) -> List[Operation]:
    addresses = []
    program = [compile_token_to_op(token) for token in tokens]
    for ip, op in enumerate(program):
        assert len(Opcode) == 10, "There's unhandled ops in `compile_tokens_to_program()`"
        if op.kind == Opcode.IF:
            addresses.append(ip)
        elif op.kind == Opcode.WHILE:
            addresses.append(ip)
        elif op.kind == Opcode.ELSE:
            ifdo_ip = addresses.pop()
            if program[ifdo_ip].kind != Opcode.DO:
                compilation_trap(op.token.loc, "`else` should only be used in `if <cond> do` blocks")
            program[ifdo_ip].operand = ip
            addresses.append(ip)
        elif op.kind == Opcode.DO:
            block_ip = addresses.pop()
            program[ip].operand = block_ip
            addresses.append(ip)
        elif op.kind == Opcode.END:
            block_ip = addresses.pop()
            if program[block_ip].kind == Opcode.ELSE:
                program[block_ip].operand = ip
            elif program[block_ip].kind == Opcode.DO:
                if program[block_ip].operand < 0:
                    compilation_trap(op.token.loc, "Invalid usage of `do` for while loop")
                refered_op = program[program[block_ip].operand]
                if refered_op.kind == Opcode.WHILE:
                    program[ip].operand = program[block_ip].operand
                    program[block_ip].operand = ip
                elif refered_op.kind == Opcode.IF:
                    program[block_ip].operand = ip
                else:
                    compilation_trap(op.token.loc, "Invalid usage of `end` for do block %s" % program[block_ip].kind)
            else:
                compilation_trap(op.token.loc, "`end` should only be used to close `if`, `do`, or `else` blocks")
    return program

def preprocess_tokens(tokens: List[Token]) -> Tuple[List[Token], Dict[str, List[Token]]]:
    macros: Dict[str, List[Token]] = {}
    tokens_without_macro_definition: List[Token] = []
    results: List[Token] = []
    memories: List[str] = []

    i = 0
    blocks = []
    tokens_amount = len(tokens)
    while i < tokens_amount:
        if tokens[i].kind == TokenKind.SYMBOL and tokens[i].value == "def":
            macro_loc = tokens[i].loc
            macro_name = ""
            i += 1

            if i < tokens_amount:
                token_i_value = tokens[i].value
                assert isinstance(token_i_value, str), "Invalid token value for macro"
                macro_name = token_i_value
            else:
                compilation_trap(macro_loc, "Invalid macro definition that immediately find end of source")

            if macro_name in MAP_OF_INTRINSIC_SYMBOLS_AND_INSTRINSICS.keys() or macro_name in LIST_OF_KEYWORDS:
                compilation_trap(macro_loc, "Redefinition of builtin keyword `%s` as a preprocessing symbols" % macro_name)
            if macro_name in macros.keys():
                compilation_trap(macro_loc, "Redefinition of existing macro `%s`" % macro_name)
            i += 1
            
            blocks.append(macro_name)
            macro_tokens = []
            while i < tokens_amount and len(blocks) > 0:
                if tokens[i].kind == TokenKind.SYMBOL:
                    if tokens[i].value == "def":
                        compilation_trap(tokens[i].loc, "Illegal behaviour in macro `%s` which is defining macro inside a macro" % macro_name)
                    elif tokens[i].value == macro_name:
                        compilation_trap(tokens[i].loc, "Illegal behaviour in macro `%s` which is referencing itself (recursive)" % macro_name)
                    elif tokens[i].value == "end":
                        block = blocks.pop()
                        if block == "while":
                            macro_tokens.append(tokens[i])
                    elif tokens[i].value == "if" or tokens[i].value == "while":
                        blocks.append("while")
                        macro_tokens.append(tokens[i])
                    elif tokens[i].value in memories:
                        assert isinstance(tokens[i].value, str), "Invalid token value for macro"
                        macro_tokens.append(Token(kind=TokenKind.MEMORY_REFER, loc=tokens[i].loc, value=tokens[i].value))
                    elif tokens[i].value in macros.keys():
                        token_i_value = tokens[i].value
                        assert isinstance(token_i_value, str), "Invalid token value for macro"
                        macro_tokens.extend(macros[token_i_value])
                    else:
                        macro_tokens.append(tokens[i])
                else:
                    macro_tokens.append(tokens[i])
                i += 1

            if len(blocks) > 0:
                compilation_trap(macro_loc, "Macro `%s` should be closed with `end` keyword" % macro_name)
            macros[macro_name] = macro_tokens
        elif tokens[i].kind == TokenKind.SYMBOL and tokens[i].value == "include":
            include_loc = tokens[i].loc
            i += 1
            if i >= tokens_amount:
                compilation_trap(include_loc, "Expecting file path after `include` keywords")
            included_path_token = tokens[i]
            i += 1
            if included_path_token.kind != TokenKind.STRING:
                compilation_trap(included_path_token.loc, "Expecting a string after `include` keyword found `%s`" % included_path_token.value)
            absolute_include_path = included_path_token.value
            assert isinstance(absolute_include_path, str), "Invalid token value for macro"
            if not os.path.exists(absolute_include_path):
                for include_dir in YOR_INCLUDE_DIRS:
                    check_path = os.path.join(include_dir, absolute_include_path)
                    if os.path.exists(check_path):
                        absolute_include_path = check_path
                        break
            loaded_tokens, loaded_macros = preprocess_tokens(lex_file(absolute_include_path))
            tokens.extend(loaded_tokens)
            macros.update(loaded_macros)
            tokens_amount = len(tokens)
        elif tokens[i].kind == TokenKind.SYMBOL and tokens[i].value == "memory":
            if i + 3 >= len(tokens):
                compilation_trap(tokens[i].loc, "Expecting name, size and the `end` keyword after the `memory` keyword")
            if tokens[i + 1].kind != TokenKind.SYMBOL:
                compilation_trap(tokens[i + 1].loc, "Expecting token symbol as the label for `memory` found %s" % tokens[i + 1].kind)
            name = tokens[i + 1].value
            assert isinstance(name, str), "Please check preprocess_tokens(). This error should not be happened"
            
            # TODO: Check if it's a symbol token and it's in the macro listings
            size = 0
            val = tokens[i + 2].value
            assert isinstance(val, str), "Please check preprocess_tokens(). This error should not be happened"
            if tokens[i + 2].kind == TokenKind.INT:
                size = int(val)
            elif tokens[i + 2].kind == TokenKind.SYMBOL:
                if val not in macros.keys():
                    compilation_trap(tokens[i + 2].loc, "This `%s` macro is not defined" % val)
                if len(macros[val]) != 1 and macros[val][0].kind == TokenKind.INT:
                    compilation_trap(tokens[i + 2].loc, "Expecting macro `%s` is a integer token" % val)
                val = macros[val][0].value
                assert isinstance(val, str), "Please check preprocess_tokens(). This error should not be happened"
                size = int(val)
            else:
                compilation_trap(tokens[i + 2].loc, "Expecting token `int` or `symbol` as the size for `memory` found %s" % tokens[i + 2].kind)

            if not (tokens[i + 3].kind == TokenKind.SYMBOL and tokens[i + 3].value == "end"):
                compilation_trap(tokens[i + 3].loc, "Expecting `end` keyword to close `memory` declaration found %s" % tokens[i + 3].kind)

            memories.append(name)
            memory_token = Token(kind=TokenKind.MEMORY_DEF, loc=tokens[i].loc, value=(name, size))
            tokens_without_macro_definition.append(memory_token)
            i += 4
        else:
            tokens_without_macro_definition.append(tokens[i])
            i += 1

    del tokens
    for token in tokens_without_macro_definition:
        if token.kind == TokenKind.SYMBOL and token.value in macros.keys():
            assert isinstance(token.value, str), "Please check preprocess_tokens(). THis error should not be happened"
            results.extend(macros[token.value])
        elif token.kind == TokenKind.SYMBOL and token.value in memories:
            assert isinstance(token.value, str), "Please check preprocess_tokens(). THis error should not be happened"
            results.append(Token(kind=TokenKind.MEMORY_REFER, loc=token.loc, value=token.value))
        else:
            results.append(token)

    return results, macros

def lex_file(file_path: str) -> List[Token]:
    if not os.path.exists(file_path):
        fatal("Could not open file `%s`" % file_path)
    with open(file_path, "r") as file:
        return lex_source(file_path, file.read())

def compile_source(file_path: str) -> List[Operation]:
    preprocessed_tokens, _ = preprocess_tokens(lex_file(file_path))
    return compile_tokens_to_program(preprocessed_tokens)

def generate_fasm_linux_x86_64(output_path: str, program: List[Operation]):
    strs: List[str] = []
    memories: Dict[str, int] = {}
    with open(output_path, "w") as out:
        out.write("format ELF64 executable\n")
        out.write("segment readable executable\n")
        out.write("entry _start\n")
        out.write("_start:\n")
        for ip, op in enumerate(program):
            assert len(Opcode) == 10, "There's unhandled ops in `generate_fasm_linux_x86_64()`"
            assert len(Intrinsic) == 28, "There's unhandled intrinsic in `generate_fasm_linux_x86_64()`"
            if op.kind == Opcode.PUSH_INT:
                assert isinstance(op.operand, int), "Invalid operand for PUSH_INT operation. There's something wrong at source parsing"
                out.write("    ;; --- push int %d --- \n" % int(op.operand))
                out.write("    push %d\n" % int(op.operand))
            elif op.kind == Opcode.PUSH_STR:
                out.write("    ;; --- push str --- \n")
                assert isinstance(op.operand, str), "Invalid operand for PUSH_STR operation. There's something wrong at source parsing"
                value = op.operand
                out.write("    push %d\n" % len(value))
                out.write("    push str_%d\n" % len(strs))
                strs.append(value)
            elif op.kind == Opcode.MEMORY_REFER:
                out.write("    ;; --- memory refer --- \n")
                assert isinstance(op.operand, str), "Invalid operand for MEMORY_REFER operation. There's something wrong at source parsing"
                if op.operand not in memories.keys():
                    compilation_trap(op.token.loc, "Memory with name %s is not exists" % op.operand)
                out.write("    push memory_%s\n" % op.operand)
            elif op.kind == Opcode.MEMORY_DEF:
                out.write("    ;; --- memory def --- \n")
                assert isinstance(op.operand, tuple), "Invalid operand for MEMORY operation. There's something wrong at source parsing"
                name, size = op.operand
                if name in memories.keys():
                    compilation_trap(op.token.loc, "Memory with name %s is already defined" % name)
                memories[name] = size
            elif op.kind == Opcode.IF:
                out.write("    ;; --- if --- \n")
            elif op.kind == Opcode.ELSE:
                out.write("    ;; --- else --- \n")
                assert isinstance(op.operand, int), "Invalid operand for ELSE operation. There's something wrong at source parsing"
                if op.operand < 0:
                    compilation_trap(op.token.loc, 
                        "`else` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the compile_token_to_op() function" 
                            if YOR_DEBUG else "")
                out.write("    jmp addr_%d\n" % op.operand)
                out.write("addr_%d:\n" % ip)
            elif op.kind == Opcode.WHILE:
                out.write("    ;; --- while --- \n")
                out.write("addr_%d:\n" % ip)
            elif op.kind == Opcode.DO:
                out.write("    ;; --- do --- \n")
                out.write("    pop rax\n")
                out.write("    cmp rax, 1\n")
                assert isinstance(op.operand, int), "Invalid operand for DO operation. There's something wrong at source parsing"
                if op.operand < 0:
                    compilation_trap(op.token.loc, 
                        "`do` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the compile_token_to_op() function" 
                            if YOR_DEBUG else "")
                out.write("    jne addr_%d\n" % op.operand)
            elif op.kind == Opcode.END:
                out.write("    ;; --- end --- \n")
                assert isinstance(op.operand, int), "Invalid operand for END operation. There's something wrong at source parsing"
                if op.operand >= 0:
                    out.write("    jmp addr_%d\n" % op.operand)
                out.write("addr_%d:\n" % ip)

            elif op.kind == Opcode.INTRINSIC:
                if op.operand == Intrinsic.DUP:
                    out.write("    ;; --- dup --- \n")
                    out.write("    pop rax\n")
                    out.write("    push rax\n")
                    out.write("    push rax\n")
                elif op.operand == Intrinsic.OVER:
                    out.write("    ;; --- over --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    push rax\n")
                    out.write("    push rbx\n")
                    out.write("    push rax\n")
                elif op.operand == Intrinsic.DROP:
                    out.write("    ;; --- drop --- \n")
                    out.write("    pop rax\n")
                elif op.operand == Intrinsic.SWAP:
                    out.write("    ;; --- swap --- \n")
                    out.write("    pop rax\n")
                    out.write("    pop rbx\n")
                    out.write("    push rax\n")
                    out.write("    push rbx\n")
                elif op.operand == Intrinsic.ROT:
                    out.write("    ;; --- rotate --- \n")
                    out.write("    pop rcx\n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    push rax\n")
                    out.write("    push rcx\n")
                    out.write("    push rbx\n")
                elif op.operand == Intrinsic.ADD:
                    out.write("    ;; --- add --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    add rax, rbx\n")
                    out.write("    push rax\n")
                elif op.operand == Intrinsic.SUB:
                    out.write("    ;; --- sub --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    sub rax, rbx\n")
                    out.write("    push rax\n")
                elif op.operand == Intrinsic.DIVMOD:
                    out.write("    ;; --- divmod --- \n")
                    out.write("    xor rdx, rdx\n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    div rbx\n")
                    out.write("    push rax\n")
                    out.write("    push rdx\n")
                elif op.operand == Intrinsic.EQ:
                    out.write("    ;; --- eq --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    cmp rax, rbx\n")
                    out.write("    mov rcx, 0\n")
                    out.write("    mov rdx, 1\n")
                    out.write("    cmove rcx, rdx\n")
                    out.write("    push rcx\n")
                elif op.operand == Intrinsic.NE:
                    out.write("    ;; --- ne --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    cmp rax, rbx\n")
                    out.write("    mov rcx, 0\n")
                    out.write("    mov rdx, 1\n")
                    out.write("    cmovne rcx, rdx\n")
                    out.write("    push rcx\n")
                elif op.operand == Intrinsic.GT:
                    out.write("    ;; --- gt --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    cmp rax, rbx\n")
                    out.write("    mov rcx, 0\n")
                    out.write("    mov rdx, 1\n")
                    out.write("    cmovg rcx, rdx\n")
                    out.write("    push rcx\n")
                elif op.operand == Intrinsic.GE:
                    out.write("    ;; --- ge --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    cmp rax, rbx\n")
                    out.write("    mov rcx, 0\n")
                    out.write("    mov rdx, 1\n")
                    out.write("    cmovge rcx, rdx\n")
                    out.write("    push rcx\n")
                elif op.operand == Intrinsic.LT:
                    out.write("    ;; --- lt --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    cmp rax, rbx\n")
                    out.write("    mov rcx, 0\n")
                    out.write("    mov rdx, 1\n")
                    out.write("    cmovl rcx, rdx\n")
                    out.write("    push rcx\n")
                elif op.operand == Intrinsic.LE:
                    out.write("    ;; --- lt --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    cmp rax, rbx\n")
                    out.write("    mov rcx, 0\n")
                    out.write("    mov rdx, 1\n")
                    out.write("    cmovle rcx, rdx\n")
                    out.write("    push rcx\n")
                elif op.operand == Intrinsic.BSL:
                    out.write("    ;; --- bsl --- \n")
                    out.write("    pop rcx\n")
                    out.write("    pop rbx\n")
                    out.write("    shl rbx, cl\n")
                    out.write("    push rbx\n")
                elif op.operand == Intrinsic.BSR:
                    out.write("    ;; --- bsr --- \n")
                    out.write("    pop rcx\n")
                    out.write("    pop rbx\n")
                    out.write("    shr rbx, cl\n")
                    out.write("    push rbx\n")
                elif op.operand == Intrinsic.MEM:
                    out.write("    ;; --- mem --- \n")
                    out.write("    push mem\n")
                elif op.operand == Intrinsic.STORE8:
                    out.write("    ;; --- store8 --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    mov [rax], bl\n")
                elif op.operand == Intrinsic.LOAD8:
                    out.write("    ;; --- load8 --- \n")
                    out.write("    pop rax\n")
                    out.write("    xor rbx, rbx\n")
                    out.write("    mov bl, BYTE [rax]\n")
                    out.write("    push rbx\n")
                elif op.operand == Intrinsic.STORE64:
                    out.write("    ;; --- store64 --- \n")
                    out.write("    pop rbx\n")
                    out.write("    pop rax\n")
                    out.write("    mov [rax], rbx\n")
                elif op.operand == Intrinsic.LOAD64:
                    out.write("    ;; --- load64 --- \n")
                    out.write("    pop rax\n")
                    out.write("    xor rbx, rbx\n")
                    out.write("    mov rbx, QWORD [rax]\n")
                    out.write("    push rbx\n")
                else:
                    if YOR_HOST_PLATFORM == "linux":
                        if op.operand == Intrinsic.LINUX_SYSCALL0:
                            out.write("    ;; --- linux syscall 0 --- \n")
                            out.write("    pop rax\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        elif op.operand == Intrinsic.LINUX_SYSCALL1:
                            out.write("    ;; --- linux syscall 1 --- \n")
                            out.write("    pop rax\n")
                            out.write("    pop rdi\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        elif op.operand == Intrinsic.LINUX_SYSCALL2:
                            out.write("    ;; --- linux syscall 2 --- \n")
                            out.write("    pop rax\n")
                            out.write("    pop rdi\n")
                            out.write("    pop rsi\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        elif op.operand == Intrinsic.LINUX_SYSCALL3:
                            out.write("    ;; --- linux syscall 3 --- \n")
                            out.write("    pop rax\n")
                            out.write("    pop rdi\n")
                            out.write("    pop rsi\n")
                            out.write("    pop rdx\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        elif op.operand == Intrinsic.LINUX_SYSCALL4:
                            out.write("    ;; --- linux syscall 4 --- \n")
                            out.write("    pop rax\n")
                            out.write("    pop rdi\n")
                            out.write("    pop rsi\n")
                            out.write("    pop rdx\n")
                            out.write("    pop r10\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        elif op.operand == Intrinsic.LINUX_SYSCALL5:
                            out.write("    ;; --- linux syscall 5 --- \n")
                            out.write("    pop rax\n")
                            out.write("    pop rdi\n")
                            out.write("    pop rsi\n")
                            out.write("    pop rdx\n")
                            out.write("    pop r10\n")
                            out.write("    pop r8\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        elif op.operand == Intrinsic.LINUX_SYSCALL6:
                            out.write("    ;; --- linux syscall 5 --- \n")
                            out.write("    pop rax\n")
                            out.write("    pop rdi\n")
                            out.write("    pop rsi\n")
                            out.write("    pop rdx\n")
                            out.write("    pop r10\n")
                            out.write("    pop r8\n")
                            out.write("    pop r9\n")
                            out.write("    syscall\n")
                            out.write("    push rax\n")
                        else:
                            compilation_trap(op.token.loc, "Invalid intrinsic operation found when generating assembly.")
            else:
                compilation_trap(op.token.loc, "Invalid operation found when generating assembly.")

        out.write("    ;; --- default exit routine in linux ---\n")
        out.write("    mov rax, 60 ;; sys exit\n")
        out.write("    mov rdi, 0\n")
        out.write("    syscall\n")
        out.write("    ret\n")

        out.write(";; --- static data ---\n")
        out.write("segment readable writable\n")
        for i, s in enumerate(strs):
            out.write("str_%d:\n" % i)
            out.write("db %s\n" % ",".join(map(hex, list(bytes(s, "utf-8")))))
        for name, size in memories.items():
            out.write("memory_%s:\n" % name)
            out.write("rb %d\n" % size)
        out.write("mem: rb %d\n" % YOR_MEM_CAPACITY)

def usage():
    print("USAGE: %s SUBCOMMANDS <ARGS> [FLAGS]" % YOR_PROGRAM_NAME)
    print("SUBCOMMANDS:")
    print("    version                          Get the current version of compiler")
    print("    help                             Get this help messages")
    print("    com <file> <output?> [FLAGS]     Compile program into platform binary. `output` is optional")
    print("        -I           Add include path")
    print("        -no-check    Do type checking")
    print("        -asm         Save the generated assembly")
    print("        -void        Compile program without supporting any host platform")
    print("        -r           Run the program if compilation success")
    print("        -rc          Run the program if compilation success and remove it after execution")
    print("        -s           Don't show any information from Yor about compilation")

def shift(argv: List[str], error_msg: str) -> Tuple[str, list[str]]:
    if len(argv) < 1:
        usage()
        fatal("[ERROR]", error_msg)
    return (argv[0], argv[1:],)

def terminal_call(commands: List[str]):
    if not YOR_SILENT:
        print("++ " + " ".join(commands))
    subprocess.call(commands)

if __name__ == "__main__":
    YOR_PROGRAM_NAME, argv = shift(sys.argv, "[Unreachable] There's no program name??")
    subcommand, argv = shift(argv, "Please provide a subcommand")
    YOR_INCLUDE_DIRS += str(os.environ.get("YOR_INCLUDE_DIRS")).split(":")

    if subcommand == "com":
        run_after_compilation = False
        save_transpiled_assembly = False
        remove_after_run = False
        do_type_checking = True

        source_path = ""
        output = ""
        while len(argv) > 0:
            item, argv = shift(argv, "[Unreachable] Expecting an item since len(argv) > 0")
            if item == "-r":
                run_after_compilation = True
            elif item == "-no-check":
                do_type_checking = True
            elif item == "-rc":
                run_after_compilation = True
                remove_after_run = True
            elif item == "-asm":
                save_transpiled_assembly = True
            elif item == "-s":
                YOR_SILENT = True
            elif item == "-void":
                YOR_HOST_PLATFORM = "void"
            elif item == "-I":
                path, argv = shift(argv, "Please provide a value for include flag")
                if not os.path.exists(path):
                    fatal("Include path `%s` is not exist" % path)
                YOR_INCLUDE_DIRS.append(path)
            elif len(source_path) == 0:
                source_path = item
            elif len(output) == 0:
                output = item

        if len(source_path) == 0:
            usage()
            fatal("[ERROR] Please provide the source file")
        if len(output) == 0:
            output, _ = os.path.splitext(os.path.basename(source_path))

        program = compile_source(source_path)
        if do_type_checking:
            type_check_program(program)
        generate_fasm_linux_x86_64(f"{output}.fasm", program)
        terminal_call(["fasm", f"{output}.fasm", f"{output}"])

        if not save_transpiled_assembly:
            terminal_call(["rm", f"{output}.fasm"])
        if run_after_compilation:
            terminal_call([os.path.abspath(output)])
            if remove_after_run:
                terminal_call(["rm", os.path.abspath(output)])

    elif subcommand == "version":
        print(YOR_VERSION)
    elif subcommand == "help":
        usage()
    else:
        usage()
        print("[ERROR]", "Invalid subcommand provided")
        sys.exit(1)
