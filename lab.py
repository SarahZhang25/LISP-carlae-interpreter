#!/usr/bin/env python3
"""6.009 Lab 9: Carlae Interpreter Part 2"""

import sys
sys.setrecursionlimit(10_000)
import doctest

# KEEP THE ABOVE LINES INTACT, BUT REPLACE THIS COMMENT WITH YOUR lab.py FROM
# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.


# NO ADDITIONAL IMPORTS!


###########################
# Carlae-related Exceptions #
###########################

# all_envs = []
# num_evals = 0

class CarlaeError(Exception):
    """
    A type of exception to be raised if there is an error with a Carlae
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class CarlaeSyntaxError(CarlaeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class CarlaeNameError(CarlaeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class CarlaeEvaluationError(CarlaeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    CarlaeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(x):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def is_boolean(x):
    """
    Helper function: given a string, convert it to True if @t or False if @f
    otherwise return the string itself
    """
    if x == carlae_true:
        return True
    elif x == carlae_false:
        return False
    else:
        return x

def tokenize(expr):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Carlae
                      expression
    """
    tokens = [expr[0]]
    i = 1
    while i < len(expr):
        # until next space char
        while i < len(expr) and expr[i] not in (" "):
            if expr[i] in ("\t"): # ignore indentation
                pass
            elif expr[i] == "#": # comment: discard rest of line, i.e. skip i to next line
                try:
                    i = i + expr[i:].index('\n')
                except:
                    i = len(expr)
            elif expr[i] in "()" or expr[i-1] in "( \n": # current parens or prev parens/space/newline indicates new word
                tokens.append(expr[i])
            elif expr[i] != "\n": # keep adding to prev token if not newline
                tokens[-1] += expr[i]
            i += 1
        i += 1
    return tokens

def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    def parse_expression(tokens, ind):
        """
        Recursively parses a list of tokens
        """
        if tokens[:ind+1].count("(") < tokens[:ind+1].count(")"): # parens mismatched
            # should never have more closing than open at any point
            raise CarlaeSyntaxError("Mismatched parentheses")
        if tokens[ind] not in "()":
            return number_or_symbol(tokens[ind]), ind+1
        
        elif tokens[ind] == "(" and ind < len(tokens)-1:
            toks = []
            ind += 1
            while tokens[ind] != ")":
                symb, ind = parse_expression(tokens, ind)
                toks.append(symb)
            return toks, ind+1

    if (len(tokens) > 1 and tokens.count("(") == 0 and tokens.count(")") == 0) or \
        tokens.count("(") != tokens.count(")"): # mismatched/missing parens
        raise CarlaeSyntaxError("Mismatched parentheses")
    else:
        parsed, _ = parse_expression(tokens, 0)
        return parsed 

######################
# Built-in Functions #
######################

def mult(*args):
    """
    Implementation of the multiplication function. Accepts elements to be multiplied either
    as a list of values or 2 different arguments.
    """
    if len(args) == 1: # handles both list and 2 number argument inputs
        args = args[0]
    
    if len(args) == 0:
        return 1
    elif len(args) == 1:
        return args[0]
    else:
        return args[0] * mult(args[1:])

def div(*args):
    """
    Implementation of the division function. Accepts elements to be divided either
    as a list of values or 2 different arguments. Requires at least 1 argument.
    """
    if len(args) == 1: # handles both list and 2 number argument inputs
        args = args[0]

    if len(args) == 0:
        raise TypeError("Not enough arguments")
    if len(args) == 1:
        return 1/args[0]
    else:
        return args[0] / mult(args[1:]) 


## Comparisons ##

class CarlaeBool():
    """
    Object that represents booleans in Carlae
    """
    def __init__(self, val):
        self.val = val
    
    def __repr__(self):
        return self.val

    def carlae_not(self):
        """
        Method that applies the python-equivalent 'not' operator to itself
        """
        if self is carlae_true:
            return carlae_false
        elif self is carlae_false:
            return carlae_true

carlae_true = CarlaeBool('@t')
carlae_false = CarlaeBool('@f')

def comparison(args, operator):
    """
    Applies a given comparison operator to a given lists of inputs

    Arguments:
        args: list of arguments to apply comparison operator to
        operator: function that performs binary comparison operations

    Returns:
        carlae_true or carlae_false depending on the evaluation of the 
            comparison operator on the arguments
    """
    if len(args) == 1: # handles both list and 2 number argument inputs
        args = args[0]

    if len(args) == 1:
        return carlae_true
    else:
        for i in range(len(args)-1):
            if not operator(args[i], args[i+1]):
                return carlae_false
        return carlae_true

def eq(*args):
    """
    Equal operator
    """
    return comparison(args, lambda a, b: a == b)

def gt(*args):
    """
    Greater-than operator
    """
    return comparison(args, lambda a, b: a > b)

def geq(*args):
    """
    Greater-than-or-equal-to operator
    """
    return comparison(args, lambda a, b: a >= b)
    
def lt(*args):
    """
    Less-than operator
    """
    return comparison(args, lambda a, b: a < b)

def leq(*args):
    """
    Less-than-or-equal-to operator
    """
    return comparison(args, lambda a, b: a <= b)

def not_func(args):
    """
    Takes a single argument and evaluates to carlae_false if its argument is true
    and carlae_true if its argument is false
    """
    if len(args) != 1:
        raise CarlaeEvaluationError("Expected 1 argument")
    res = args[0]
    if not isinstance(res, CarlaeBool):
        raise TypeError("Expression must evaluate to a CarlaeBool")
    
    return res.carlae_not() # apply not operator to given carlae bool

comparison_operators = ("=?", ">", ">=", "<", "<=")

## Lists ##
class Pair():
    """
    Class that represents the pair object, which contains a head and tail
    """
    def __init__(self, args):
        self.head = args[0]
        self.tail = args[1]
    
    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def set_head(self, new_head):
        self.head = new_head

    def set_tail(self, new_tail):
        self.tail = new_tail

    def __repr__(self):
        return "(pair "+repr(self.head)+" "+repr(self.tail)+")"

    def __str__(self):
        return "("+str(self.head)+", "+str(self.tail)+")"


def make_linked_list(args):
    """
    Makes a linked list out of Pairs given a list of values
    """
    if len(args) == 0:
        return None
    else:
        return Pair([args[0], make_linked_list(args[1:])])

def copy_linked_list(ls):
    """
    Makes a copy of a linked list
    """
    if ls == None: # empty list
        return None
    else: # recursively copy
        return Pair([ls.get_head(), copy_linked_list(ls.get_tail())])


def is_linked_list(args):
    """
    Checks if a given input is a valid linked list (i.e. made of Pairs with the last 
    node's tail pointing to None). Returns carlae_true if valid and carlae_false elsewise.
    """
    # acceptable if list is given directly or in a singleton list
    if isinstance(args, list) and len(args) == 1:
        obj = args[0]
    else:
        obj = args

    if obj == None: # empty list
        return carlae_true
    elif not isinstance(obj, Pair): # not a Pair
        return carlae_false

    # get last node
    tail = obj.get_tail()
    while isinstance(tail, Pair):
        tail = tail.get_tail()

    if tail == None:
        return carlae_true
    else:
        return carlae_false


def length_list(args):
    """
    Returns the number of nodes in a given linked list
    """
    # acceptable if list is given directly or in a singleton list
    if isinstance(args, list) and len(args) == 1:
        ls = args[0]
    else:
        ls = args
        
    if is_linked_list(ls) != carlae_true:
        raise CarlaeEvaluationError("Function expects argument to be a linked list")
    
    
    if ls == None: # empty list
        return 0
    else:
        # get last node
        tail = ls.get_tail()
        length = 1
        while isinstance(tail, Pair):
            tail = tail.get_tail()
            length += 1
        return length


def get_nth_linked_list_index(args):
    """
    Returns the n-th index of a list, indexing from 0. Input expected as a list in [ls, index] format
    """
    ls, index = args
    if is_linked_list(ls) != carlae_true:
        if isinstance(ls, Pair):
            if index == 0:
                return ls.get_head()
            else:
                raise CarlaeEvaluationError("Pair does not support indexing greater than 0")
        raise CarlaeEvaluationError("Function expects argument to be a linked list")
    
    if ls == None:
        raise CarlaeEvaluationError("Cannot index into empty list")
    elif index == 0:
        return ls.get_head()
    else: # iterate through list
        tail = ls.get_tail()
        length = 1
        while isinstance(tail, Pair):
            if length == index:
                return tail.get_head()
            tail = tail.get_tail()
            length += 1
        raise CarlaeEvaluationError("Index out of bounds")


def concat_linked_list(lists): 
    """
    Returns a new list that represents the concatenation of an arbitrary number of linked lists given. 
    Input expected as a list in [linked_list1, linked_list2, ....] format
    """
    if lists != [] and is_linked_list(lists[0]) != carlae_true:
        raise CarlaeEvaluationError("Function expects arguments to be linked lists")

    if len(lists) == 0: # empty list
        return None
    elif len(lists) == 1: # return copy of self
        return copy_linked_list(lists[0])
    else:
        if lists[0] == None: # empty list
            # skip to next list
            return concat_linked_list(lists[1:])

        # connect tail of first list to the head of next list
        new_list = copy_linked_list(lists[0])

        if not isinstance(new_list.get_tail(), Pair): # list has 1 item
            new_list_tail = new_list
        else: 
            # get to tail
            new_list_tail = new_list.get_tail()
            while isinstance(new_list_tail.get_tail(), Pair): # will terminate on last pair
                new_list_tail = new_list_tail.get_tail()

        new_list_tail.set_tail(concat_linked_list(lists[1:]))

        return new_list


def map_linked_list(args):
    """
    Returns a new list containing the results of applying the given function to each element of the given list.
    Input expected as a list in [function, linked_list] format
    """
    fxn, ls = args

    if type(ls) in (int, float):
        mapped_val = fxn.evaluate([ls])
        return mapped_val

    else: # type(ls) is linked list
        new_list = copy_linked_list(ls)
        ptr = new_list

        # apply map to each value by evaluating the function on each value
        while ptr != None:
            if fxn in carlae_builtins.values(): # built in
                mapped_val = fxn([ptr.get_head()])
            else: # user defined
                mapped_val = fxn.evaluate([ptr.get_head()])
            
            ptr.set_head(mapped_val)
            ptr = ptr.get_tail()
        return new_list


def filter_linked_list(args):
    """
    Returns a new list containing only the elements of the given list for which the given function returns true.
    Input expected as a list in [function, linked_list] format
    """
    fxn, ls = args

    if type(ls) in (int, float):
        filter_result = fxn.evaluate([ls])
        return filter_result

    else: # type(ls) is linked list
        new_list = copy_linked_list(ls)
        ptr = new_list
        prev_ptr = ptr # trailing ptr at latest index that passes filter

        # apply filter to each value by skipping over nodes that do not pass
        while ptr != None:
            if fxn in carlae_builtins.values(): # built in
                filter_result = fxn([ptr.get_head()])
            else: # user defined
                filter_result = fxn.evaluate([ptr.get_head()])
            
            if filter_result == carlae_false: # skip current node if does not pass filter
                prev_ptr.set_tail(ptr.get_tail())

                # if first element fails, move up the trailing ptr and chop off the front of the list
                if ptr == new_list: # ptr == new_list only on first element
                    prev_ptr = prev_ptr.get_tail()
                    new_list = new_list.get_tail()        
            else: # if passes filter, update the trailing ptr
                prev_ptr = ptr

            ptr = ptr.get_tail()
        
        return new_list


def reduce_linked_list(args):
    """
    Returns the result of successively applying a given function to the elements in a given list, 
    starting from a given initial value and maintaining an intermediate result along the way.
    Input expected as a list in [function, linked_list, inital_value] format
    """
    fxn, ls, init_val = args

    result = init_val

    if type(ls) in (int, float):
        filter_result = fxn.evaluate([ls])
        return filter_result

    else: # type(ls) is linked list
        ptr = ls
        # apply function to each value successive value and the running result
        while ptr != None:
            if fxn in carlae_builtins.values(): # built in
                result = fxn([result, ptr.get_head()])
            else: # user defined
                result = fxn.evaluate([result, ptr.get_head()])
            
            ptr = ptr.get_tail()

        return result


### File evaluation ###
def begin(args):
    """
    Returns the last argument of a given list of items. useful structure for running commands successively
    allowing ability to run arbitrary expressions sequentially
    """
    return args[-1]

def evaluate_file(fname, envir=None):
    if not envir:
        envir = Environment(parent_env=builtins_env, bindings={})

    with open(fname, 'r') as f:
        expr = f.read()
        return evaluate(parse(tokenize(expr)), envir)

# list of all built in functions
carlae_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mult,
    "/": div,
    "=?": eq,
    ">": gt,
    ">=": geq,
    "<": lt,
    "<=": leq,
    "not": not_func,
    '@t': carlae_true,
    '@f': carlae_false,
    'nil': None,
    "pair": Pair,
    "head": Pair.get_head,
    "tail": Pair.get_tail,
    "list": make_linked_list,
    "list?": is_linked_list, # (list? OBJECT)
    "length": length_list, # (length LIST)
    "nth": get_nth_linked_list_index, # (nth LIST INDEX)
    "concat": concat_linked_list, # (concat LIST1 LIST2 LIST3 ...)
    "map": map_linked_list, # (map FUNCTION LIST)
    "filter": filter_linked_list, # (filter FUNCTION LIST)
    "reduce": reduce_linked_list, # (reduce FUNCTION LIST INITVAL)
    "begin": begin
}


################
# Environments #
################

class Function:
    """
    Class that represents a user-defined function
    """
    def __init__(self, params, body_expr, enclosing_env):
        """
        Executes the definition of a function

        Arguments:
            params: parameters to the function (list)
            body_expr: expression which represents the body of the function
            enclosing_env: the enclosing environment which the function was defined in (Environment)
        """
        self.params = params
        self.body_expr = body_expr
        self.enclosing_env = enclosing_env

    def evaluate(self, args):
        """
        Evaluates given a list of values to use as arguments to the body expression
        """
        func_env = Environment(parent_env= self.enclosing_env, bindings={})
        # all_envs.append(func_env)
        for i in range(len(args)): # bind args in function env
            func_env.set_binding(self.params[i], args[i])
        
        res = evaluate(self.body_expr, func_env)
        return res


class Environment:
    """
    Class that represents an environments which operations are performed in
    """
    def __init__(self, parent_env = None, bindings = None):
        """
        Arguments:
            parent_env: parent environment of current environment. Default None
            bindings: dictionary representing all bindings made and stored in this environment 
        """
        self.parent_env = parent_env # has type Environment

        if not bindings:
            self.bindings = {}
        else:
            self.bindings = bindings

    def set_binding(self, name, assignment):
        """
        Creates a binding of assignment to name in current environment
        """
        self.bindings[name] = assignment

    def get_binding(self, name):
        """
        Retreives the assignment of a name from the current environment or a parent environment. 
        If the name does not exist in the current or a parent environment, raises CarlaeNameError 
        """
        try: # try to retrieve from current env
            return self.bindings[name]
        except:
            try: # try to retrieve from parent env
                return self.parent_env.get_binding(name)
            except: # failed, raise error
                raise CarlaeNameError("Given name does not exist")

    def get_binding_current_env(self, name):
        """
        Retreives the assignment of a name from only the current environment. 
        If the name does not exist in the current or a parent environment, raises CarlaeNameError 
        """
        try: # try to retrieve from current env, else raise error
            return self.bindings[name]
        except:
            raise CarlaeNameError("Given name does not exist in current environment")

    def del_binding(self, name):
        """
        Deletes the binding of a given name from the current environment and raises
        CarlaeNameError is the name is not bound locally
        """
        try:
            return self.bindings.pop(name)
        except:
            raise CarlaeNameError(name + " is not bound locally, cannot be removed")
            
    def __str__(self):
        return "Environment bindings: " + str(self.bindings)

# represents the built-in environment, which only contains the built-in functions and booleans
builtins_env = Environment(bindings=carlae_builtins)
globals = Environment(parent_env=builtins_env, bindings={}) # global frame

# all_envs = [builtins_env, globals]


##############
# Evaluation #
##############


def evaluate(tree, envir = None):
    """
    Evaluate the given syntax tree according to the rules of the Carlae language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the parse function
    """
    # print("evaluating:", tree)
    # num_evals += 1

    def is_func_check(string):
        """
        helper, check type is Function or in carlae_builtins
        """
        return isinstance(envir.get_binding(string), Function) or (envir.get_binding(string) in carlae_builtins.values())


    if not envir:
        envir = Environment(parent_env=builtins_env, bindings={})

    if isinstance(tree, list) and len(tree) == 0:
        raise CarlaeEvaluationError("Empty expression")

    # handle if input is list and is not a var definition or valid function
    if isinstance(tree, list) and tree[0] != ":=":
        is_func = False
        try: # check type is Function or in carlae_builtins
            is_func = is_func_check(tree[0])
            # is_func = isinstance(envir.get_binding(tree[0]), Function) or (envir.get_binding(tree[0]) in carlae_builtins.values())
        except:
            pass
        
        special_forms = ("function", "and", "or", "if", "list", "del", "let", "set!")
        if not (isinstance(tree[0], list) or (tree[0] in special_forms) or is_func):
            # raise CarlaeNameError
            if isinstance(tree[0], str) and not is_func:
                raise CarlaeNameError("Function " + tree[0] + " is not defined")
            raise CarlaeEvaluationError("Evaluation unable to be complete, input format not acceptable")

    # evaluate
    if isinstance(tree, int) or isinstance(tree, float) or isinstance(tree, CarlaeBool): # is number or bool
        return tree
    elif isinstance(tree, str):
        try: # try to retrieve variable or function
            return envir.get_binding(tree)
        except:
            raise CarlaeNameError("Var does not exist")
    else: #type(tree) is list: special form or function
        # recurse
        if tree[0] == ":=": # assignment
            if isinstance(tree[1], list): # short function definition
                name = tree[1][0]

                if len(tree[1]) > 1:
                    func_params = tree[1][1:]
                else:
                    func_params = []

                assignment = Function(params=func_params, body_expr=tree[2] , enclosing_env=envir)
            else: # define new variable assignment
                if type(number_or_symbol(tree[1])) != str:
                    raise TypeError("Var name cannot be a number")
                name = tree[1] # should be a symbol
            
                assignment = evaluate(tree[2], envir) # evaluate expr
                
            envir.set_binding(name, assignment)
            return assignment
        elif tree[0] == "function": # long function definition
            # expecting: (function (PARAM1 PARAM2 ...) EXPR)
            new_func = Function(params=tree[1], body_expr=tree[2], enclosing_env=envir)
            return new_func
        elif tree[0] == "and":
            for item in tree[1:]: # expecting: (and <arbitrary num elements>)
                if evaluate(item, envir) != carlae_true:
                    return carlae_false
            return carlae_true
        elif tree[0] == "or": # expecting: (or <arbitrary num elements>)
            for item in tree[1:]:
                if evaluate(item, envir) == carlae_true:
                    return carlae_true
            return carlae_false
        elif tree[0] == "if": # expecting: (if COND TRUE_EXP FALSE_EXP)
            if evaluate(tree[1], envir) == carlae_true:
                return evaluate(tree[2], envir)
            else:
                return evaluate(tree[3], envir)
        elif tree[0] == "del": # expecting: (del VAR)
            return envir.del_binding(tree[1])
        elif tree[0] == "let": # expecting: (let ((VAR1 VAL1) (VAR2 VAL2) (VAR3 VAL3) ...) BODY)
            if len(tree[1:]) != 2:
                raise CarlaeEvaluationError("Argument error: expected input format of (let ((VAR1 VAL1) (VAR2 VAL2) (VAR3 VAL3) ...) BODY)")
            
            var_vals = tree[1]
            body = tree[2]
            
            # evaluate arguments
            args = [evaluate(item[1], envir) for item in var_vals]

            let_env = Environment(parent_env= envir, bindings={})
            # all_envs.append(let_env)
            for i in range(len(args)): # bind args in a local env
                let_env.set_binding(var_vals[i][0], args[i])
            
            res = evaluate(body, let_env)
            return res
        elif tree[0] == "set!": # expecting: (set! VAR EXPR)
            if len(tree[1:]) != 2:
                raise CarlaeEvaluationError("Argument error: expected input format of (set! VAR EXPR)")

            # find nearest enclosing environment where VAR is defined
            var_env = envir
            env_found = False
            while not (env_found or var_env == None):
                try:
                    var_env.get_binding_current_env(tree[1])
                    env_found = True
                except:
                    var_env = var_env.parent_env
                    if var_env == None:
                        raise CarlaeNameError("Var " + tree[1] + " not defined")
            
            # set the var in that env
            expr = evaluate(tree[2], envir)
            var_env.set_binding(tree[1], expr)

            return expr
        else: # evaluate the function
            # error check for functions assoc'd with Pairs
            func = evaluate(tree[0], envir)

            if tree[0] in ("list?", "length"): # take in 1 arg
                if len(tree[1:]) != 1:
                    raise CarlaeEvaluationError("Argument error: expected one argument")
                args = [evaluate(tree[1], envir)]
            elif tree[0] in ("head", "tail"): # take in 1 arg of type Pair
                if len(tree[1:]) != 1 or not isinstance(evaluate(tree[1], envir), Pair):
                    raise CarlaeEvaluationError("Argument error: expected one argument of type Pair")
                args = evaluate(tree[1], envir)
            elif tree[0] in ("pair", "nth"):#, "map", "filter"): # take in 2 args
                if len(tree[1:]) != 2:
                    raise CarlaeEvaluationError("Argument error: expected two arguments")
                args = [evaluate(item, envir) for item in tree[1:]]
            elif tree[0] in ("map", "filter"): # take in 2 args
                if len(tree[1:]) != 2:
                    raise CarlaeEvaluationError("Argument error: expected two arguments")
                if not (tree[1][0] == "function" or is_func_check(tree[1])) or \
                    is_linked_list(evaluate(tree[2], envir)) != carlae_true:
                    raise CarlaeEvaluationError("Argument error: expected a function and list")
                args = [evaluate(item, envir) for item in tree[1:]]
            elif tree[0] == "reduce": # take in 3 args
                if len(tree[1:]) != 3:
                    raise CarlaeEvaluationError("Argument error: expected three arguments")
                if not (tree[1][0] == "function" or is_func_check(tree[1])) or \
                    is_linked_list(evaluate(tree[2], envir)) != carlae_true:# or \
                    # not (type(evaluate(tree[3], envir)) in (int, float)):
                    raise CarlaeEvaluationError("Argument error: expected a function, list and number")
                
                args = [evaluate(item, envir) for item in tree[1:]]
            else: # take in arbitrary args
                args = [evaluate(item, envir) for item in tree[1:]]

            
            if func in carlae_builtins.values(): # function is built-in
                res = func(args)
            else: # function is user-defined
                # check if correct num args supplied
                if len(args) != len(func.params):
                    raise CarlaeEvaluationError("Too few or too many arguments supplied")

                res = func.evaluate(args)
                return res
            return res

def result_and_env(tree, envir = None):
    """
    Evaluate the given syntax tree according to the rules of the Carlae language.
    Returns the result and the environment the expression was evaluated in.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the parse function
        envir: environment to execute the expression in
    """
    if not envir:
        envir = Environment(parent_env=builtins_env, bindings={})
    return evaluate(tree, envir), envir

def run_repl():
    """
    REPL for carlae which takes in user input in the form of one-line carlae prompts. Exit the REPL by inputting "exit"
    """
    # globals = Environment(parent_env=builtins_env, bindings={}) # global frame
    
    while True:
        user_input = input("in>>> ")
        if user_input == "exit":
            break


        # output = evaluate(parse(tokenize(user_input)), globals)
        try: # attempt evaluation
            output = evaluate(parse(tokenize(user_input)), globals)
        except Exception as e: # if error, save and display exception
            output = "Exception: "+ str(type(e)) + "; Message: " + str(e)
        
        print("    out>>>", output)

if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()

    # run files first
    files = sys.argv[1:]
    for fname in files:
        evaluate_file(fname, globals)
    run_repl()

    # pass
    # x = evaluate(parse(tokenize("(list 1 2 3 4)")))
    # print(type(x))
    # print(evaluate(parse(tokenize("(list 1 2 3 4)"))))
    # print(result_and_env([":=", "x", 2]))
    # print(evaluate([":=", "spam", "x"]))
    # print(result_and_env([":=", "spam", "x"]))
