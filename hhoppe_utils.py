#!/usr/bin/env python3
"""Python library for Hugues Hoppe.

# pylint: disable=line-too-long
Useful commands:

python3 -m doctest -v hh.py  # Test doc string examples.
python3 hh.py  # Also run _self_test().
mypy --strict ~/bin/hh.py
gpylint ~/bin/hh.py
pydoc3 ~/bin/hh.py  # Print module help information (like "help(hh)" within Python)
autopep8 --aggressive --max-line-length 80 --indent-size 2 --diff ~/bin/hh.py
autopep8 --aggressive --max-line-length 80 --indent-size 2 ~/bin/hh.py >/tmp/v && ediff ~/bin/hh.py /tmp/v
autopep8 --aggressive --max-line-length 80 --indent-size 2 --inplace ~/bin/hh.py

# pylint: enable=line-too-long
"""

import doctest
import os
import re
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Sequence, Union

import numpy as np  # type: ignore


def check(condition: bool, message: Any = '') -> None:
  """Raises an informative exception unless condition.

  Args:
    condition: expression whose value should be true.
    message: string or object reported in exception if condition is false.
  Raises:
    RuntimeError: if condition is false.

  >>> check(1)

  >>> check(False, (1, 2))
  Traceback (most recent call last):
  ...
  RuntimeError: Check fails: (1, 2)
  """
  if not condition:
    if not isinstance(message, str):
      message = repr(message)
    raise RuntimeError(f'Check fails: {message}')


def check_eq(a: Any, b: Any, message: Any = None) -> None:
  """Raises an informative exception unless a == b.

  Args:
    a: first expression.
    b: second expression.
    message: string or object reported in exception if condition is false.
  Raises:
    RuntimeError: if condition is false.

  >>> check_eq('a', 'a')

  >>> check_eq(1 + 2, 4)
  Traceback (most recent call last):
  ...
  RuntimeError: Check fails: 3 == 4
  """
  message = '' if message is None else f' ({message})'
  if isinstance(a, np.ndarray):
    check(np.all(a == b), f'{a!r} == {b!r}{message}')
  else:
    check(a == b, f'{a!r} == {b!r}{message}')


def eprint(*args: str, **kwargs: Any) -> None:
  """Print to stderr."""
  kwargs = {**dict(file=sys.stderr, flush=True), **kwargs}
  print(*args, **kwargs)


def run(args: Union[str, Sequence[str]]) -> None:
  """Execute command, printing output from stdout and stderr.

  Args:
    args: command to execute, which can be either a string or a sequence of
      word strings, as in subprocess.run().  If args is a string, the shell is
      invoked to interpret it.

  Raises:
    RuntimeError: (with verbose string) if the command's exit code is nonzero.

  >>> run('echo a')
  a

  >>> run(['sh', '-c', 'echo b'])
  b
  """
  proc = subprocess.run(
      args,
      shell=isinstance(args, str),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
      universal_newlines=True)
  print(proc.stdout, end='', flush=True)
  if proc.returncode:
    raise RuntimeError(
        f'Command {proc.args} failed with code {proc.returncode}')


def matching_parenthesis(text: str) -> int:
  """Returns the index of ')' matching '(' in text[0].

  Args:
    text: string whose first character must be a left parenthesis.
  Raises:
    RuntimeError: if there is no matching right parenthesis in the text string.

  >>> matching_parenthesis('(hello (there) and here) again')
  23
  """
  assert text[0] == '('
  num_open = 0
  for i, c in enumerate(text):
    if c == '(':
      num_open += 1
    elif c == ')':
      num_open -= 1
      if num_open == 0:
        return i
  raise RuntimeError(f'No matching parenthesis in "{text}"')


def dump_vars(*args: Any) -> str:
  """Returns a string showing the values of each expression.

  Specifically, converts each expression (contributed by the caller to the
  variable-parameter list *args) into a substring f'expression = {expression}'
  and joins these substrings separated by ', '.

  If the caller itself provided a variable-parameter list (*args),
  the search continues in its callers.  The approach examines a stack trace,
  so it is fragile and non-portable.

  Args:
    *args: expressions.
  Raises:
    Exception: if the dump_vars(...) does not fit on a single source line.

  >>> a = 45
  >>> b = 'Hello'
  >>> dump_vars(a, b, (a * 2) + 5, b + ' there')
  "a = 45, b = Hello, (a * 2) + 5 = 95, b + ' there' = Hello there"
  """
  # Adapted from make_dict() in https://stackoverflow.com/a/2553524/1190077.
  stack = traceback.extract_stack()
  this_function_name = stack[-1][2]
  for stackframe in stack[-2::-1]:
    (filename, unused_line_number, function_name, text) = stackframe  # caller
    # https://docs.python.org/3/tutorial/errors.html:
    # "however, it will not display lines read from standard input."
    if filename == '<stdin>':
      assert text == ''
      return ', '.join(str(e) for e in args)
    prefix = this_function_name + '('
    begin = text.find(prefix)
    if begin < 0:
      raise Exception(f'Cannot find "{prefix}" in line "{text}"')
    begin += len(this_function_name)
    end = begin + matching_parenthesis(text[begin:])
    parameter_string = text[begin + 1:end]
    if re.fullmatch(r'\*[\w]+', parameter_string):
      this_function_name = function_name
      # Because the call is made using a *args, we continue to
      # the earlier caller in the stack trace.
    else:
      text = [name.strip() for name in parameter_string.split(',')]
      return ', '.join([t + ' = ' + str(e) for (t, e) in zip(text, args)])
  assert False


def show(*args: Any, **kwargs: Any) -> None:
  """Prints expressions and their values on stderr.

  Args:
    *args: expressions to show.
    **kwargs: keyword arguments passed to eprint().

  >>> show(4 * 3, file=sys.stdout)
  4 * 3 = 12

  >>> a ='<string>'
  >>> show(a, a * 2, 34 // 3, file=sys.stdout)
  a = <string>, a * 2 = <string><string>, 34 // 3 = 11
  """
  eprint(dump_vars(*args), **kwargs)


def _self_test() -> None:
  """Runs doctest and other tests."""
  doctest.testmod()

  with tempfile.TemporaryDirectory() as tmpdir:
    filename = os.path.join(tmpdir, 'test.txt')
    run(f'echo a >{filename}')
    with open(filename) as f:
      check_eq(f.read(), 'a\n')


if __name__ == '__main__':
  _self_test()
