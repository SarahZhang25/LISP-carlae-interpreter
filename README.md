# 'LISP carlae' interpreter
Turing-complete implementation of an interpreter for 'carlae', a dialect of LISP, in Python. Completed for MIT 6.009 Spring 2022 lab

The syntax of carlae is made up of numbers, symbols, and S-expressions. The interpreter is made up of three main components:
1. Tokenizer, which breaks up input strings into a list of parseable syntax units (i.e. tokens)
2. Parser, which takes the output of the tokenizer and output an abstract syntax tree representation of the program
3. Evaluator, which takes the output of the parser and executes the program.

The interpreter also has a REPL for interactive use: to access it, run the `lab.py` file, and to exit the REPL, enter 'exit'. This implementation supports variables and variable-binding manipulation, arithmetic operations, functions, conditionals and comparisons, arrays, and array operations. 
