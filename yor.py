#!/usr/bin/env python3

import os
import sys 
import subprocess
from typing import Any, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum, auto

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

class Opcode(Enum):
    PUSH_INT = auto()
    PUSH_STR = auto()
    DUP = auto()
    SWAP = auto()
    DROP = auto()
    OVER = auto()
    ROT = auto()

    ADD = auto()
    SUB = auto()
    EQ = auto() 
    NE = auto()
    GT = auto()
    GE = auto()
    LT = auto()
    LE = auto()

    BSL = auto()
    BSR = auto()

    IF = auto()
    ELSE = auto()
    END = auto()
    WHILE = auto()
    DO = auto()

    MEM = auto()
    LOAD8 = auto()
    STORE8 = auto()
    LOAD64 = auto()
    STORE64 = auto()

    # Linux
    LINUX_SYSCALL0 = auto()
    LINUX_SYSCALL1 = auto()
    LINUX_SYSCALL2 = auto()
    LINUX_SYSCALL3 = auto()
    LINUX_SYSCALL4 = auto()
    LINUX_SYSCALL5 = auto()
    LINUX_SYSCALL6 = auto()

    # Debug
    DUMP = auto()

Loc = Tuple[str, int, int] # file_path, row, col

@dataclass
class Instruction:
    kind: Opcode
    value: Any
    loc: Loc

class TokenKind(Enum):
    SYMBOL = auto()
    INT = auto()
    STRING = auto()
    NATIVE = auto()

@dataclass
class Token:
    loc: Loc
    value: str
    kind: TokenKind

def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)

def compilation_trap(loc: Loc, *args: str):
    print("Compilation error at line %d position %d in file %s:" % (loc[1], loc[2], loc[0]), file=sys.stderr)
    fatal(*args)

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
            while i < source_len and not source[i].isspace():
                i, row, col = lex_advance(source, i, row, col)
            symbol = source[start:i]
            result.append(Token((source_path, row + 1, col + 1), symbol, TokenKind.SYMBOL))

    return result

map_of_builtin_symbols_and_opkind = {
    "+": Opcode.ADD,
    "-": Opcode.SUB,
    "=": Opcode.EQ,
    "!": Opcode.NE,
    "<": Opcode.LT,
    "<=": Opcode.LE,
    ">": Opcode.GT,
    ">=": Opcode.GE,

    ">>": Opcode.BSR,
    "<<": Opcode.BSL,

    "!8": Opcode.STORE8,
    "?8": Opcode.LOAD8,
    "!64": Opcode.STORE64,
    "?64": Opcode.LOAD64,

    "if": Opcode.IF,
    "else": Opcode.ELSE,
    "while": Opcode.WHILE,
    "do": Opcode.DO,
    "end": Opcode.END,

    "dup": Opcode.DUP,
    "drop": Opcode.DROP,
    "over": Opcode.OVER,
    "swap": Opcode.SWAP,
    "rot": Opcode.ROT,
    "mem": Opcode.MEM,

    "linux:syscall(0)": Opcode.LINUX_SYSCALL1,
    "linux:syscall(1)": Opcode.LINUX_SYSCALL1,
    "linux:syscall(2)": Opcode.LINUX_SYSCALL2,
    "linux:syscall(3)": Opcode.LINUX_SYSCALL3,
    "linux:syscall(4)": Opcode.LINUX_SYSCALL4,
    "linux:syscall(5)": Opcode.LINUX_SYSCALL5,
    "linux:syscall(6)": Opcode.LINUX_SYSCALL6,

    "dump": Opcode.DUMP,
}

list_of_linux_only_symbols = [ 
    "linux:syscall(0)", "linux:syscall(1)", "linux:syscall(2)", 
    "linux:syscall(3)", "linux:syscall(4)", "linux:syscall(5)",
    "linux:syscall(6)",
]

list_of_debug_only_symbols = [ "dump", ]

def compile_token_to_op(token: Token) -> Instruction:
    assert len(Opcode) == 35, "There's unhandled ops in `translate_token()`"
    if token.kind == TokenKind.SYMBOL:
        if token.value not in map_of_builtin_symbols_and_opkind:
            compilation_trap(token.loc, "Invalid syntax `%s`" % token.value)
        if YOR_HOST_PLATFORM != "linux" and token.value in list_of_linux_only_symbols:
            compilation_trap(token.loc, "Syntax `%s` is not supported on `%s` platform" % (token.value, YOR_HOST_PLATFORM))
        return Instruction(map_of_builtin_symbols_and_opkind[token.value], -1, token.loc)
    if token.kind == TokenKind.INT:
        return Instruction(Opcode.PUSH_INT, int(token.value), token.loc)
    elif token.kind == TokenKind.STRING:
        return Instruction(Opcode.PUSH_STR, token.value, token.loc)
    else:
        compilation_trap(token.loc, "Unreachable token `%s` with type `%s` in translate_token()" % (token.value, token.kind))
        return Instruction(Opcode.PUSH_INT, -1, token.loc)

def compile_tokens_to_program(tokens: List[Token]) -> List[Instruction]:
    addresses = []
    program = [compile_token_to_op(token) for token in tokens]
    for ip, op in enumerate(program):
        assert len(Opcode) == 35, "There's unhandled op in `compile_tokens_to_program()`. Please handle if it creates a block"
        if op.kind == Opcode.IF:
            addresses.append(ip)
        elif op.kind == Opcode.ELSE:
            if_ip = addresses.pop()
            if program[if_ip].kind != Opcode.IF:
                compilation_trap(op.loc, "`else` should only be used in `if` blocks")
            program[if_ip].value = ip
            addresses.append(ip)
        elif op.kind == Opcode.WHILE:
            addresses.append(ip)
        elif op.kind == Opcode.DO:
            while_ip = addresses.pop()
            program[ip].value = while_ip
            addresses.append(ip)
        elif op.kind == Opcode.END:
            block_ip = addresses.pop()
            if program[block_ip].kind == Opcode.IF or program[block_ip].kind == Opcode.ELSE:
                program[block_ip].value = ip
            elif program[block_ip].kind == Opcode.DO:
                if program[block_ip].value < 0:
                    compilation_trap(op.loc, "Invalid usage of `do`")
                program[ip].value = program[block_ip].value
                program[block_ip].value = ip
            else:
                compilation_trap(op.loc, "`end` should only be used to close `if`, `do`, or `else` blocks")
    return program

def preprocess_tokens(tokens: List[Token]) -> Tuple[List[Token], Dict[str, List[Token]]]:
    macros: Dict[str, List[Token]] = {}
    tokens_without_macro_definition: List[Token] = []
    results: List[Token] = []

    i = 0
    blocks = []
    tokens_amount = len(tokens)
    while i < tokens_amount:
        if tokens[i].kind == TokenKind.SYMBOL and tokens[i].value == "def":
            macro_loc = tokens[i].loc
            macro_name = ""
            i += 1

            if i < tokens_amount:
                macro_name = tokens[i].value
            else:
                compilation_trap(macro_loc, "Invalid macro definition that immediately find end of source")

            if macro_name in map_of_builtin_symbols_and_opkind:
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
                    elif tokens[i].value in macros.keys():
                        macro_tokens.extend(macros[tokens[i].value])
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
            if not os.path.exists(absolute_include_path):
                for include_dir in YOR_INCLUDE_DIRS:
                    check_path = os.path.join(include_dir, included_path_token.value)
                    if os.path.exists(check_path):
                        absolute_include_path = check_path
            loaded_tokens, loaded_macros = preprocess_tokens(lex_file(absolute_include_path))
            tokens.extend(loaded_tokens)
            macros.update(loaded_macros)
            tokens_amount = len(tokens)
        else:
            tokens_without_macro_definition.append(tokens[i])
            i += 1

    del tokens
    for token in tokens_without_macro_definition:
        if token.kind == TokenKind.SYMBOL and token.value in macros.keys():
            results.extend(macros[token.value])
        else:
            results.append(token)

    return results, macros

def lex_file(file_path: str) -> List[Token]:
    if not os.path.exists(file_path):
        fatal("Could not open file `%s`" % file_path)
    with open(file_path, "r") as file:
        return lex_source(file_path, file.read())

def compile_source(file_path: str) -> List[Instruction]:
    preprocessed_tokens, _ = preprocess_tokens(lex_file(file_path))
    return compile_tokens_to_program(preprocessed_tokens)

def generate_fasm_linux_x86_64(output_path: str, program: List[Instruction]):
    strs: List[str] = []
    with open(output_path, "w") as out:
        out.write("format ELF64 executable\n")
        out.write("segment readable executable\n")
        out.write("dump:\n")
        out.write("    mov     r9, -3689348814741910323\n")
        out.write("    sub     rsp, 40\n")
        out.write("    mov     BYTE [rsp+31], 10\n")
        out.write("    lea     rcx, [rsp+30]\n")
        out.write(".L2:\n")
        out.write("    mov     rax, rdi\n")
        out.write("    lea     r8, [rsp+32]\n")
        out.write("    mul     r9\n")
        out.write("    mov     rax, rdi\n")
        out.write("    sub     r8, rcx\n")
        out.write("    shr     rdx, 3\n")
        out.write("    lea     rsi, [rdx+rdx*4]\n")
        out.write("    add     rsi, rsi\n")
        out.write("    sub     rax, rsi\n")
        out.write("    add     eax, 48\n")
        out.write("    mov     BYTE [rcx], al\n")
        out.write("    mov     rax, rdi\n")
        out.write("    mov     rdi, rdx\n")
        out.write("    mov     rdx, rcx\n")
        out.write("    sub     rcx, 1\n")
        out.write("    cmp     rax, 9\n")
        out.write("    ja      .L2\n")
        out.write("    lea     rax, [rsp+32]\n")
        out.write("    mov     edi, 1\n")
        out.write("    sub     rdx, rax\n")
        out.write("    xor     eax, eax\n")
        out.write("    lea     rsi, [rsp+32+rdx]\n")
        out.write("    mov     rdx, r8\n")
        out.write("    mov     rax, 1\n")
        out.write("    syscall\n")
        out.write("    add     rsp, 40\n")
        out.write("    ret\n")
        out.write("entry _start\n")
        out.write("_start:\n")
        for ip, op in enumerate(program):
            assert len(Opcode) == 35, "There's unhandled op in `compile_program()`"
            if op.kind == Opcode.PUSH_INT:
                out.write("    ;; --- push int %d --- \n" % op.value)
                out.write("    push %d\n" % op.value)
            elif op.kind == Opcode.PUSH_STR:
                out.write("    ;; --- push str --- \n")
                out.write("    push %d\n" % len(op.value))
                out.write("    push str_%d\n" % len(strs))
                strs.append(op.value)
            elif op.kind == Opcode.DUP:
                out.write("    ;; --- dup --- \n")
                out.write("    pop rax\n")
                out.write("    push rax\n")
                out.write("    push rax\n")
            elif op.kind == Opcode.OVER:
                out.write("    ;; --- over --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    push rax\n")
                out.write("    push rbx\n")
                out.write("    push rax\n")
            elif op.kind == Opcode.DROP:
                out.write("    ;; --- drop --- \n")
                out.write("    pop rax\n")
            elif op.kind == Opcode.SWAP:
                out.write("    ;; --- swap --- \n")
                out.write("    pop rax\n")
                out.write("    pop rbx\n")
                out.write("    push rax\n")
                out.write("    push rbx\n")
            elif op.kind == Opcode.ROT:
                out.write("    ;; --- rotate --- \n")
                out.write("    pop rcx\n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    push rax\n")
                out.write("    push rcx\n")
                out.write("    push rbx\n")
            elif op.kind == Opcode.ADD:
                out.write("    ;; --- add --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    add rax, rbx\n")
                out.write("    push rax\n")
            elif op.kind == Opcode.SUB:
                out.write("    ;; --- sub --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    sub rax, rbx\n")
                out.write("    push rax\n")
            elif op.kind == Opcode.EQ:
                out.write("    ;; --- eq --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmove rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == Opcode.NE:
                out.write("    ;; --- ne --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovne rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == Opcode.GT:
                out.write("    ;; --- gt --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovg rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == Opcode.GE:
                out.write("    ;; --- ge --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovge rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == Opcode.LT:
                out.write("    ;; --- lt --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovl rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == Opcode.LE:
                out.write("    ;; --- lt --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovle rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == Opcode.BSL:
                out.write("    ;; --- bsl --- \n")
                out.write("    pop rcx\n")
                out.write("    pop rbx\n")
                out.write("    shl rbx, cl\n")
                out.write("    push rbx\n")
            elif op.kind == Opcode.BSR:
                out.write("    ;; --- bsr --- \n")
                out.write("    pop rcx\n")
                out.write("    pop rbx\n")
                out.write("    shr rbx, cl\n")
                out.write("    push rbx\n")

            elif op.kind == Opcode.MEM:
                out.write("    ;; --- mem --- \n")
                out.write("    push mem\n")
            elif op.kind == Opcode.STORE8:
                out.write("    ;; --- store8 --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    mov [rax], bl\n")
            elif op.kind == Opcode.LOAD8:
                out.write("    ;; --- load8 --- \n")
                out.write("    pop rax\n")
                out.write("    xor rbx, rbx\n")
                out.write("    mov bl, BYTE [rax]\n")
                out.write("    push rbx\n")
            elif op.kind == Opcode.STORE64:
                out.write("    ;; --- store64 --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    mov [rax], rbx\n")
            elif op.kind == Opcode.LOAD64:
                out.write("    ;; --- load8 --- \n")
                out.write("    pop rax\n")
                out.write("    xor rbx, rbx\n")
                out.write("    mov rbx, QWORD [rax]\n")
                out.write("    push rbx\n")

            elif op.kind == Opcode.IF:
                out.write("    ;; --- if --- \n")
                out.write("    pop rax\n")
                out.write("    cmp rax, 1\n")
                if op.value < 0:
                    compilation_trap(op.loc, 
                        "`if` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the crossreference_program() function" 
                            if YOR_DEBUG else "")
                out.write("    jne addr_%d\n" % op.value)
            elif op.kind == Opcode.ELSE:
                out.write("    ;; --- else --- \n")
                if op.value < 0:
                    compilation_trap(op.loc, 
                        "`else` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the crossreference_program() function" 
                            if YOR_DEBUG else "")
                out.write("    jmp addr_%d\n" % op.value)
                out.write("addr_%d:\n" % ip)
            elif op.kind == Opcode.WHILE:
                out.write("    ;; --- while --- \n")
                out.write("addr_%d:\n" % ip)
            elif op.kind == Opcode.DO:
                out.write("    ;; --- do --- \n")
                out.write("    pop rax\n")
                out.write("    cmp rax, 1\n")
                if op.value < 0:
                    compilation_trap(op.loc, 
                        "`do` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the crossreference_program() function" 
                            if YOR_DEBUG else "")
                out.write("    jne addr_%d\n" % op.value)
            elif op.kind == Opcode.END:
                out.write("    ;; --- end --- \n")
                if op.value >= 0:
                    out.write("    jmp addr_%d\n" % op.value)
                out.write("addr_%d:\n" % ip)

            elif op.kind == Opcode.DUMP:
                out.write("    ;; --- debug dump --- \n")
                out.write("    pop rdi\n")
                out.write("    call dump\n")

            else:
                if YOR_HOST_PLATFORM == "linux":
                    if op.kind == Opcode.LINUX_SYSCALL0:
                        out.write("    ;; --- linux syscall 0 --- \n")
                        out.write("    pop rax\n")
                        out.write("    syscall\n")
                        out.write("    push rax\n")
                    elif op.kind == Opcode.LINUX_SYSCALL1:
                        out.write("    ;; --- linux syscall 1 --- \n")
                        out.write("    pop rax\n")
                        out.write("    pop rdi\n")
                        out.write("    syscall\n")
                        out.write("    push rax\n")
                    elif op.kind == Opcode.LINUX_SYSCALL2:
                        out.write("    ;; --- linux syscall 2 --- \n")
                        out.write("    pop rax\n")
                        out.write("    pop rdi\n")
                        out.write("    pop rsi\n")
                        out.write("    syscall\n")
                        out.write("    push rax\n")
                    elif op.kind == Opcode.LINUX_SYSCALL3:
                        out.write("    ;; --- linux syscall 3 --- \n")
                        out.write("    pop rax\n")
                        out.write("    pop rdi\n")
                        out.write("    pop rsi\n")
                        out.write("    pop rdx\n")
                        out.write("    syscall\n")
                        out.write("    push rax\n")
                    elif op.kind == Opcode.LINUX_SYSCALL4:
                        out.write("    ;; --- linux syscall 4 --- \n")
                        out.write("    pop rax\n")
                        out.write("    pop rdi\n")
                        out.write("    pop rsi\n")
                        out.write("    pop rdx\n")
                        out.write("    pop r10\n")
                        out.write("    syscall\n")
                        out.write("    push rax\n")
                    elif op.kind == Opcode.LINUX_SYSCALL5:
                        out.write("    ;; --- linux syscall 5 --- \n")
                        out.write("    pop rax\n")
                        out.write("    pop rdi\n")
                        out.write("    pop rsi\n")
                        out.write("    pop rdx\n")
                        out.write("    pop r10\n")
                        out.write("    pop r8\n")
                        out.write("    syscall\n")
                        out.write("    push rax\n")
                    elif op.kind == Opcode.LINUX_SYSCALL6:
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
                    compilation_trap(op.loc, "Invalid instruction found.")

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
        out.write("mem: rb %d\n" % YOR_MEM_CAPACITY)

def usage():
    print("USAGE: %s SUBCOMMANDS <ARGS> [FLAGS]" % YOR_PROGRAM_NAME)
    print("SUBCOMMANDS:")
    print("    com <file> <output?> [FLAGS]     Compile program into platform binary. `output` is optional")
    print("        -r        Run the program if compilation success")
    print("        -rc       Run the program if compilation success and remove it after execution")
    print("        -asm      Save the generated assembly")
    print("        -I        Add include path")
    print("        -void     Compile program without supporting any host platform")
    print("        -silent   Don't show any information from Yor about compilation")
    print("    version                          Get the current version of compiler")
    print("    help                             Get this help messages")

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

    if subcommand == "com":
        run_after_compilation = False
        save_transpiled_assembly = False
        remove_after_run = False

        source_path = ""
        output = ""
        while len(argv) > 0:
            item, argv = shift(argv, "[Unreachable] Expecting an item since len(argv) > 0")
            if item == "-r":
                run_after_compilation = True
            elif item == "-rc":
                run_after_compilation = True
                remove_after_run = True
            elif item == "-asm":
                save_transpiled_assembly = True
            elif item == "-silent":
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
