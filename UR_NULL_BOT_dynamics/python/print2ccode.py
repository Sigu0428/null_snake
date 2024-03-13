#!/usr/bin/env python3
import symengine as se
import sympy as sp


def expr2ccode(expr, assign_to=None, optimize=True):
    if not isinstance(assign_to, (se.Basic, type(None), str)):
        raise TypeError(f"Cannot assign to object of type {type(assign_to)}")

    assign_to = str(assign_to)

    # def symbol_name():
    #     num = 1
    #     while True:
    #         # yield f"x{num:03d}"
    #         yield se.Symbol(f"x{num}")
    #         num += 1
    
    # symgen = symbol_name()
    # symgen = sp.numbered_symbols("x_")

    def get_ccode(expr, assign_to="tmp", optimize=True):
        if optimize:
            # symbols, simple = sp.cse([expr], symbols = symgen, optimizations="basic", order='none')
            symbols, simple = se.cse([expr])
            # new_symbols = [(f"{tmp_prefix}_{s[0]}", s[1]) for s in symbols]
            # symbol_names = [s[0] for s in symbols]
            # new_symbols_names = [f"{assign_to}_{sn}" for sn in symbol_names]
            # for i in range(len(symbols)):

            out = ""
            
            if simple[0] != 0:
                for s in symbols:
                    out += f"{s[0]} = {se.ccode(s[1])};\n"
            out += f"{assign_to} = {se.ccode(simple[0])};\n"
            return out
        else:
            return se.ccode(expr)

    if not expr.is_Matrix:
        return f"{assign_to} = {get_ccode(expr, assign_to=assign_to, optimize=optimize)}"

    code_lines = []
    for i, element in enumerate(expr):
        code_line = get_ccode(element, assign_to=f"{assign_to}[{i}]", optimize=optimize)
        code_lines.append(code_line)

    max_lines = max([x.count('\n') for x in code_lines]) - 1 # -1 for the return line
    definitions = [f"double x{i};\n" for i in range(max_lines)]
    
    return ''.join(definitions + code_lines)