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

if sys.platform == "linux" or sys.platform == "linux2":
    YOR_HOST_PLATFORM = "linux"
elif sys.platform == "win32":
    YOR_HOST_PLATFORM = "win32"
else:
    YOR_HOST_PLATFORM = "void"

class OpKind(Enum):
    OP_PUSH_INT = auto()
    OP_PUSH_STR = auto()
    OP_DUP = auto()
    OP_SWAP = auto()
    OP_DROP = auto()
    OP_OVER = auto()
    OP_ROT = auto()
    OP_ADD = auto()
    OP_SUB = auto()
    OP_EQ = auto() 
    OP_NE = auto()
    OP_GT = auto()
    OP_GE = auto()
    OP_LT = auto()
    OP_LE = auto()

    OP_IF = auto()
    OP_ELSE = auto()
    OP_END = auto()
    OP_WHILE = auto()
    OP_DO = auto()

    OP_MEM = auto()
    OP_LOAD = auto()
    OP_STORE = auto()

    # Linux
    OP_SYSCALL1 = auto()
    OP_SYSCALL3 = auto()

    # Debug
    OP_DUMP = auto()

class TokenKind(Enum):
    TOK_SYMBOL = auto()
    TOK_INT = auto()
    TOK_STRING = auto()

@dataclass
class Token:
    loc: Tuple[str, int, int] # file, row, col
    value: str
    kind: TokenKind

@dataclass
class Loc:
    file_path: str
    row: int
    col: int

@dataclass
class Op:
    kind: OpKind
    value: Any
    loc: Tuple[str, int, int]

def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)

def compilation_trap(loc: Tuple[str, int, int], *args: str):
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
            result.append(Token((source_path, row + 1, col + 1), source[start:i], TokenKind.TOK_INT))
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
            result.append(Token((source_path, row + 1, col + 1), translate_unescaped_str(source[start:i]), TokenKind.TOK_STRING))
            i, row, col = lex_advance(source, i, row, col)
            if i < source_len and not source[i].isspace():
                compilation_trap((source_path, row + 1, col + 1), "Expecting whitespace after a string")
        else:
            start = i
            while i < source_len and not source[i].isspace():
                i, row, col = lex_advance(source, i, row, col)
            result.append(Token((source_path, row + 1, col + 1), source[start:i], TokenKind.TOK_SYMBOL))

    return result

map_of_builtin_symbols_and_opkind = {
    "+": OpKind.OP_ADD,
    "-": OpKind.OP_SUB,
    "=": OpKind.OP_EQ,
    "!": OpKind.OP_NE,
    "<": OpKind.OP_LT,
    "<=": OpKind.OP_LE,
    ">": OpKind.OP_GT,
    ">=": OpKind.OP_GE,

    "if": OpKind.OP_IF,
    "else": OpKind.OP_ELSE,
    "while": OpKind.OP_WHILE,
    "do": OpKind.OP_DO,
    "end": OpKind.OP_END,

    "dup": OpKind.OP_DUP,
    "drop": OpKind.OP_DROP,
    "over": OpKind.OP_OVER,
    "swap": OpKind.OP_SWAP,
    "rot": OpKind.OP_ROT,
    "mem": OpKind.OP_MEM,
    "store": OpKind.OP_STORE,
    "load": OpKind.OP_LOAD,

    "syscall1": OpKind.OP_SYSCALL1,
    "syscall3": OpKind.OP_SYSCALL3,
    "dump": OpKind.OP_DUMP,
}

list_of_linux_only_symbols = [ "syscall1", "syscall2", "syscall3" ]
list_of_debug_only_symbols = [ "dump", ]

def compile_token_to_op(token: Token) -> Op:
    assert len(OpKind) == 26, "There's unhandled ops in `translate_token()`"
    if token.kind == TokenKind.TOK_SYMBOL:
        if token.value not in map_of_builtin_symbols_and_opkind:
            compilation_trap(token.loc, "Invalid syntax `%s`" % token.value)
        if YOR_HOST_PLATFORM != "linux" and token.value in list_of_linux_only_symbols:
            compilation_trap(token.loc, "Syntax `%s` is not supported on `%s` platform" % (token.value, YOR_HOST_PLATFORM))
        return Op(map_of_builtin_symbols_and_opkind[token.value], -1, token.loc)
    if token.kind == TokenKind.TOK_INT:
        return Op(OpKind.OP_PUSH_INT, int(token.value), token.loc)
    elif token.kind == TokenKind.TOK_STRING:
        return Op(OpKind.OP_PUSH_STR, token.value, token.loc)
    else:
        compilation_trap(token.loc, "Unreachable token `%s` with type `%s` in translate_token()" % (token.value, token.kind))
        return Op(OpKind.OP_PUSH_INT, -1, token.loc)

def compile_tokens_to_program(tokens: List[Token]) -> List[Op]:
    addresses = []
    program = [compile_token_to_op(token) for token in tokens]
    for ip, op in enumerate(program):
        assert len(OpKind) == 26, "There's unhandled op in `compile_tokens_to_program()`. Please handle if it creates a block"
        if op.kind == OpKind.OP_IF:
            addresses.append(ip)
        elif op.kind == OpKind.OP_ELSE:
            if_ip = addresses.pop()
            if program[if_ip].kind != OpKind.OP_IF:
                compilation_trap(op.loc, "`else` should only be used in `if` blocks")
            program[if_ip].value = ip
            addresses.append(ip)
        elif op.kind == OpKind.OP_WHILE:
            addresses.append(ip)
        elif op.kind == OpKind.OP_DO:
            while_ip = addresses.pop()
            program[ip].value = while_ip
            addresses.append(ip)
        elif op.kind == OpKind.OP_END:
            block_ip = addresses.pop()
            if program[block_ip].kind == OpKind.OP_IF or program[block_ip].kind == OpKind.OP_ELSE:
                program[block_ip].value = ip
            elif program[block_ip].kind == OpKind.OP_DO:
                if program[block_ip].value < 0:
                    compilation_trap(op.loc, "Invalid usage of `do`")
                program[ip].value = program[block_ip].value
                program[block_ip].value = ip
            else:
                compilation_trap(op.loc, "`end` should only be used to close `if`, `do`, or `else` blocks")
    return program

def preprocess_tokens(tokens: List[Token]) -> List[Token]:
    macros: Dict[str, List[Token]] = {}
    tokens_without_macro_definition: List[Token] = []
    results: List[Token] = []

    i = 0
    blocks = []
    tokens_amount = len(tokens)
    while i < tokens_amount:
        if tokens[i].kind == TokenKind.TOK_SYMBOL and tokens[i].value == "def":
            macro_loc = tokens[i].loc
            macro_name = ""
            i += 1

            if i < tokens_amount:
                macro_name = tokens[i].value
            else:
                compilation_trap(macro_loc, "Invalid macro definition that immediately find end of source")

            if macro_name in map_of_builtin_symbols_and_opkind:
                compilation_trap(macro_loc, "Redefinition of builtin keyword `%s` as a prerocessing symbols" % macro_name)
            if macro_name in macros.keys():
                compilation_trap(macro_loc, "Redefinition of existing macro `%s`" % macro_name)
            i += 1
            
            blocks.append(macro_name)
            macro_tokens = []
            while i < tokens_amount and len(blocks) > 0:
                if tokens[i].kind == TokenKind.TOK_SYMBOL:
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
        else:
            tokens_without_macro_definition.append(tokens[i])
            i += 1
    for token in tokens_without_macro_definition:
        if token.kind == TokenKind.TOK_SYMBOL and token.value in macros.keys():
            results.extend(macros[token.value])
        else:
            results.append(token)

    return results

def compile_source(file_path: str, source: str) -> List[Op]:
    preprocessed_tokens = preprocess_tokens(lex_source(file_path, source))
    return compile_tokens_to_program(preprocessed_tokens)

def generate_fasm_linux_x86_64(output_path: str, program: List[Op]):
    strs = []
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
            assert len(OpKind) == 26, "There's unhandled op in `compile_program()`"
            if op.kind == OpKind.OP_PUSH_INT:
                out.write("    ;; --- push int %d --- \n" % op.value)
                out.write("    push %d\n" % op.value)
            elif op.kind == OpKind.OP_PUSH_STR:
                out.write("    ;; --- push str --- \n")
                out.write("    push %d\n" % len(op.value))
                out.write("    push str_%d\n" % len(strs))
                strs.append(op.value)
            elif op.kind == OpKind.OP_DUP:
                out.write("    ;; --- dup --- \n")
                out.write("    pop rax\n")
                out.write("    push rax\n")
                out.write("    push rax\n")
            elif op.kind == OpKind.OP_OVER:
                out.write("    ;; --- over --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    push rax\n")
                out.write("    push rbx\n")
                out.write("    push rax\n")
            elif op.kind == OpKind.OP_DROP:
                out.write("    ;; --- drop --- \n")
                out.write("    pop rax\n")
            elif op.kind == OpKind.OP_SWAP:
                out.write("    ;; --- swap --- \n")
                out.write("    pop rax\n")
                out.write("    pop rbx\n")
                out.write("    push rax\n")
                out.write("    push rbx\n")
            elif op.kind == OpKind.OP_ROT:
                out.write("    ;; --- rotate --- \n")
                out.write("    pop rcx\n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    push rax\n")
                out.write("    push rcx\n")
                out.write("    push rbx\n")
            elif op.kind == OpKind.OP_ADD:
                out.write("    ;; --- add --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    add rax, rbx\n")
                out.write("    push rax\n")
            elif op.kind == OpKind.OP_SUB:
                out.write("    ;; --- sub --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    sub rax, rbx\n")
                out.write("    push rax\n")
            elif op.kind == OpKind.OP_EQ:
                out.write("    ;; --- eq --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmove rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == OpKind.OP_NE:
                out.write("    ;; --- ne --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovne rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == OpKind.OP_GT:
                out.write("    ;; --- gt --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovg rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == OpKind.OP_GE:
                out.write("    ;; --- ge --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovge rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == OpKind.OP_LT:
                out.write("    ;; --- lt --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovl rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == OpKind.OP_LE:
                out.write("    ;; --- lt --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    cmp rax, rbx\n")
                out.write("    mov rcx, 0\n")
                out.write("    mov rdx, 1\n")
                out.write("    cmovle rcx, rdx\n")
                out.write("    push rcx\n")
            elif op.kind == OpKind.OP_IF:
                out.write("    ;; --- if --- \n")
                out.write("    pop rax\n")
                out.write("    cmp rax, 1\n")
                if op.value < 0:
                    compilation_trap(op.loc, 
                        "`if` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the crossreference_program() function" 
                            if YOR_DEBUG else "")
                out.write("    jne addr_%d\n" % op.value)
            elif op.kind == OpKind.OP_ELSE:
                out.write("    ;; --- else --- \n")
                if op.value < 0:
                    compilation_trap(op.loc, 
                        "`else` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the crossreference_program() function" 
                            if YOR_DEBUG else "")
                out.write("    jmp addr_%d\n" % op.value)
                out.write("addr_%d:\n" % ip)
            elif op.kind == OpKind.OP_WHILE:
                out.write("    ;; --- while --- \n")
                out.write("addr_%d:\n" % ip)
            elif op.kind == OpKind.OP_DO:
                out.write("    ;; --- do --- \n")
                out.write("    pop rax\n")
                out.write("    cmp rax, 1\n")
                if op.value < 0:
                    compilation_trap(op.loc, 
                        "`do` instruction has no reference to the end of its block."
                        "This might me crossreference issues. Please check the crossreference_program() function" 
                            if YOR_DEBUG else "")
                out.write("    jne addr_%d\n" % op.value)
            elif op.kind == OpKind.OP_END:
                out.write("    ;; --- end --- \n")
                if op.value >= 0:
                    out.write("    jmp addr_%d\n" % op.value)
                out.write("addr_%d:\n" % ip)
            elif op.kind == OpKind.OP_MEM:
                out.write("    ;; --- mem --- \n")
                out.write("    push mem\n")
            elif op.kind == OpKind.OP_STORE:
                out.write("    ;; --- store --- \n")
                out.write("    pop rbx\n")
                out.write("    pop rax\n")
                out.write("    mov [rax], bl\n")
            elif op.kind == OpKind.OP_LOAD:
                out.write("    ;; --- load --- \n")
                out.write("    pop rax\n")
                out.write("    xor rbx, rbx\n")
                out.write("    mov bl, BYTE [rax]\n")
                out.write("    push rbx\n")

            elif op.kind == OpKind.OP_SYSCALL1:
                out.write("    ;; --- linux syscall1 --- \n")
                if YOR_HOST_PLATFORM != "linux":
                    compilation_trap(op.loc, "Could not do `%s` instruction at `%s` platform" % (op.kind, YOR_HOST_PLATFORM))
                out.write("    pop rax\n")
                out.write("    pop rdi\n")
                out.write("    syscall\n")
            elif op.kind == OpKind.OP_SYSCALL3:
                out.write("    ;; --- linux syscall3 --- \n")
                if YOR_HOST_PLATFORM != "linux":
                    compilation_trap(op.loc, "Could not do `%s` instruction at `%s` platform" % (op.kind, YOR_HOST_PLATFORM))
                out.write("    pop rax\n")
                out.write("    pop rdi\n")
                out.write("    pop rsi\n")
                out.write("    pop rdx\n")
                out.write("    syscall\n")

            elif op.kind == OpKind.OP_DUMP:
                out.write("    ;; --- debug dump --- \n")
                out.write("    pop rdi\n")
                out.write("    call dump\n")
            else:
                compilation_trap(op.loc, "Invalid instruction found.")
        out.write("    mov rax, 60 ;; sys exit\n")
        out.write("    mov rdi, 0\n")
        out.write("    syscall\n")
        out.write("    ret\n")
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
    print("        -void     Compile program with without supporting any host platform")
    print("    version                          Get the current version of compiler")
    print("    help                             Get this help messages")

def shift(argv: List[str], error_msg: str) -> Tuple[str, list[str]]:
    if len(argv) < 1:
        usage()
        fatal("[ERROR]", error_msg)
    return (argv[0], argv[1:],)

def terminal_call(commands: List[str]):
    print("++ " + " ".join(commands))
    subprocess.call(commands)

if __name__ == "__main__":
    YOR_PROGRAM_NAME, argv = shift(sys.argv, "[Unreachable] there's no program name??")
    subcommand, argv = shift(argv, "Please provide a subcommand")

    if subcommand == "com":
        run_after_compilation = False
        save_transpiled_assembly = False
        remove_after_run = False

        source_path = ""
        output = ""
        while len(argv) > 0:
            item, argv = shift(argv, "[Unreachable]")
            if item == "-r":
                run_after_compilation = True
            elif item == "-rc":
                run_after_compilation = True
                remove_after_run = True
            elif item == "-asm":
                save_transpiled_assembly = True
            elif item == "-void":
                YOR_HOST_PLATFORM = "void"
            elif len(source_path) == 0:
                source_path = item
            elif len(output) == 0:
                output = item

        if len(source_path) == 0:
            usage()
            fatal("[ERROR] Please provide the source file")
        if len(output) == 0:
            output, _ = os.path.splitext(os.path.basename(source_path))

        with open(source_path, "r") as source_file:
            program = compile_source(source_path, source_file.read())
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
