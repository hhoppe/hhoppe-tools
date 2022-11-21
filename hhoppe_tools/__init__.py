#!/usr/bin/env python3
# -*- fill-column: 80; -*-
"""Library of Python tools -- Hugues Hoppe.

Useful commands to test and polish this file:

c:/windows/sysnative/wsl -e bash -lc 'cd ..; echo flake8; flake8; echo mypy; mypy . | grep -Ev " errors? in . file|Any from function declared to return .(ndarray|Tensor)"; echo autopep8; autopep8 -j8 -d .; echo pylint; pylint -j8 .; echo pytest; pytest -qq; echo All ran.'

env python3 -m doctest -v __init__.py | perl -ne 'print if /had no tests/../passed all/' | tail -n +2 | head -n -1
"""

from __future__ import annotations

__docformat__ = 'google'
__version__ = '1.0.2'
__version_info__ = tuple(int(num) for num in __version__.split('.'))

import ast
import collections.abc
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import contextlib
import cProfile
import dataclasses
import doctest
import enum
import functools
import gc
import io  # pylint: disable=unused-import
import importlib.util
import itertools
import math
import os  # pylint: disable=unused-import
import pathlib
import pstats
import re
import stat
import subprocess
import sys
import tempfile  # pylint:disable=unused-import # noqa
import time
import traceback
import typing
from typing import Any, Generic, TypeVar, Union
import unittest.mock  # pylint: disable=unused-import # noqa

import numpy as np
import numpy.typing as npt

# For np.broadcast_to(), etc.
# mypy: allow-untyped-calls

_T = TypeVar('_T')
_F = TypeVar('_F', bound='Callable[..., Any]')

_UNDEFINED = object()

_NDArray = npt.NDArray[Any]
_DTypeLike = npt.DTypeLike
_ArrayLike = npt.ArrayLike

# https://github.com/python/mypy/issues/5667
if typing.TYPE_CHECKING:
  _Path = Union[str, 'os.PathLike[str]']
else:
  _Path = Union[str, os.PathLike]


# ** Language extensions


def assert_value(value: _T | None) -> _T:
  """Return value after asserting it, as in a functional assert.

  >>> assert_value('word')
  'word'

  >>> assert_value(33)
  33

  >>> assert_value(0.0)
  Traceback (most recent call last):
  ...
  AssertionError: 0.0

  >>> assert_value([])
  Traceback (most recent call last):
  ...
  AssertionError: []

  >>> assert_value(None)
  Traceback (most recent call last):
  ...
  AssertionError: None

  >>> assert_value(False)
  Traceback (most recent call last):
  ...
  AssertionError: False
  """
  assert value, value
  return value


def assert_not_none(value: _T | None) -> _T:
  """Return value after asserting that it is not None.

  >>> assert_not_none('word')
  'word'

  >>> assert_not_none(0)
  0

  >>> assert_not_none('')
  ''

  >>> assert_not_none(())
  ()

  >>> assert_not_none(False)
  False

  >>> assert_not_none(None)
  Traceback (most recent call last):
  ...
  AssertionError
  """
  assert value is not None
  return value


# ** Debugging output


def check_eq(a: Any, b: Any) -> None:
  """Assert that two values are equal or raise exception with a useful message.

  Args:
    a: First expression.
    b: Second expression.
  Raises:
    RuntimeError: If a != b (or np.any(a != b) if np.ndarray).

  >>> check_eq('a' + 'b', 'ab')

  >>> check_eq(1 + 2, 4)
  Traceback (most recent call last):
  ...
  AssertionError: 3 == 4
  """
  are_equal = np.all(a == b) if isinstance(a, np.ndarray) else a == b
  if not are_equal:
    raise AssertionError(f'{a!r} == {b!r}')


def print_err(*args: str, **kwargs: Any) -> None:
  r"""Prints arguments to stderr immediately.

  >>> with unittest.mock.patch('sys.stderr', new_callable=io.StringIO) as m:
  ...   print_err('hello')
  ...   print(repr(m.getvalue()))
  'hello\n'
  """
  kwargs = {**dict(file=sys.stderr, flush=True), **kwargs}
  print(*args, **kwargs)


def dump_vars(*args: Any) -> str:
  """Return a string showing the values of each expression.

  Specifically, convert each expression (contributed by the caller to the
  variable-parameter list *args) into a substring f'expression = {expression}'
  and join these substrings separated by ', '.

  If the caller itself provided a variable-parameter list (*args),
  the search continues in its callers.  The approach examines a stack trace,
  so it is fragile and non-portable.

  Args:
    *args: Expressions to show.
  Raises:
    Exception: If the dump_vars(...) does not fit on a single source line.

  >>> a = 45
  >>> b = 'Hello'
  >>> dump_vars(a)
  'a = 45'
  >>> dump_vars(b)
  'b = Hello'
  >>> dump_vars(a, b, (a * 2) + 5, b + ' there')
  "a = 45, b = Hello, (a * 2) + 5 = 95, b + ' there' = Hello there"
  >>> dump_vars([3, 4, 5][1])
  '[3, 4, 5][1] = 4'
  """

  def matching_parenthesis(text: str) -> int:
    """Return the index of ')' matching '(' in text[0]."""
    check_eq(text[0], '(')
    num_open = 0
    for i, c in enumerate(text):
      if c == '(':
        num_open += 1
      elif c == ')':
        num_open -= 1
        if num_open == 0:
          return i
    raise RuntimeError(f'No matching right parenthesis in "{text}"')

  # Adapted from make_dict() in https://stackoverflow.com/a/2553524 .
  stack = traceback.extract_stack()
  this_function_name = stack[-1][2]  # i.e. initially 'dump_vars'.
  for stackframe in stack[-2::-1]:
    (filename, unused_line_number, function_name, text) = stackframe  # Caller.
    # https://docs.python.org/3/tutorial/errors.html:
    # "however, it will not display lines read from standard input."
    if filename == '<stdin>':
      check_eq(text, '')
      return ', '.join(str(e) for e in args)  # Unfortunate fallback.
    prefix = this_function_name + '('
    begin = text.find(prefix)
    if begin < 0:
      raise Exception(f'dump_vars: cannot find "{prefix}" in line "{text}"')
    begin += len(this_function_name)
    end = begin + matching_parenthesis(text[begin:])
    parameter_string = text[begin + 1:end].strip()
    if re.fullmatch(r'\*[\w]+', parameter_string):
      this_function_name = function_name
      # Because the call is made using a *args, we continue to
      # the earlier caller in the stack trace.
    else:
      if len(args) == 1:
        expressions = [parameter_string.strip()]
      elif hasattr(ast, 'get_source_segment'):  # Python 3.8.
        node = ast.parse(parameter_string)
        # print(ast.dump(node))  # ", indent=2" requires Python 3.9.
        value = getattr(node.body[0], 'value', '?')
        elements = getattr(value, 'elts', [value])

        def get_text(element: Any) -> str:
          text = ast.get_source_segment(parameter_string, element)
          return '?' if text is None else text

        expressions = [get_text(element) for element in elements]
      else:
        expressions = [name.strip() for name in parameter_string.split(',')]
      l = []
      for (expr, value) in zip(expressions, args):
        l.append(f'{expr} = {value}' if expr[0] not in '"\'' else str(value))
      return ', '.join(l)
  raise AssertionError


def show(*args: Any) -> None:
  r"""Prints expressions and their values on stdout.

  Args:
    *args: Expressions to show.

  >>> with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as m:
  ...   show(4 * 3)
  ...   check_eq(m.getvalue(), '4 * 3 = 12\n')

  >>> with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as m:
  ...   a ='<string>'
  ...   show(a, 'literal_string', "s", a * 2, 34 // 3)
  ...   s = m.getvalue()
  >>> s
  'a = <string>, literal_string, s, a * 2 = <string><string>, 34 // 3 = 11\n'
  """
  print(dump_vars(*args), flush=True)


def analyze_lru_caches(variables: Mapping[str, Any]) -> None:
  """Report on usage and efficiency of memoization caches.

  Args:
    variables: Dictionary, which is usually `globals()`.

  >>> @functools.lru_cache(maxsize=None)
  ... def func(i: int) -> int:
  ...   return i**2

  >>> [func(i) for i in [1, 2, 1, 3, 1]]
  [1, 4, 1, 9, 1]
  >>> analyze_lru_caches(globals())
  ...  # doctest:+ELLIPSIS
  # func ...  3/inf        0.400 hit=            2 miss=            3
  """
  for name, value in variables.items():
    try:
      info = value.cache_info()
    except AttributeError:
      continue

    hit_ratio = info.hits / (info.hits + info.misses + 1e-30)
    s_max_size = 'inf' if info.maxsize is None else f'{info.maxsize:_}'
    name2 = f'{name:31}' if len(name) <= 31 else f'{name[:15]}..{name[-14:]}'
    print(f'# {name2} {info.currsize:11_}/{s_max_size:<10}'
          f' {hit_ratio:5.3f} hit={info.hits:13_} miss={info.misses:13_}')


def clear_lru_caches(variables: Mapping[str, Any], *,
                     verbose: bool = False) -> None:
  """Clear all the function memoization caches.

  Args:
    variables: Dictionary, which is usually `globals()`.
    verbose: If True, show names of cleared caches.

  >>> @functools.lru_cache(maxsize=None)
  ... def func(i: int) -> int:
  ...   return i**2

  >>> [func(i) for i in [1, 2, 1, 3, 1]]
  [1, 4, 1, 9, 1]
  >>> check_eq(func.cache_info().currsize, 3)
  >>> clear_lru_caches(globals())
  >>> check_eq(func.cache_info().hits, 0)
  """
  for name, value in variables.items():
    try:
      value.cache_clear()
      if verbose:
        print(f'Cleared lru_cache for {name}().')
    except AttributeError:
      pass


# ** Jupyter/IPython notebook functionality


def in_colab() -> bool:
  """Return True if running inside Google Colab.

  >>> in_colab()
  False
  """
  try:
    import google.colab  # pylint: disable=unused-import # noqa
    return True
  except ModuleNotFoundError:
    return False


def adjust_jupyterlab_markdown_width(width: int = 1016) -> None:
  """Set the Markdown cell width in Jupyterlab to the value used by Colab."""
  # https://stackoverflow.com/a/66278615.
  import IPython.display
  style = f'{{max-width: {width}px!important;}}'
  text = f'<style>.jp-Cell.jp-MarkdownCell {style}</style>'
  IPython.display.display(IPython.display.HTML(text))


class _CellTimer:
  """Record timings of all notebook cells and show top entries at the end."""
  # Inspired from https://github.com/cpcloud/ipython-autotime.

  instance: _CellTimer | None = None

  def __init__(self) -> None:
    import IPython
    self.elapsed_times: dict[int, float] = {}
    self.pre_run(None)
    IPython.get_ipython().events.register('pre_run_cell', self.pre_run)
    IPython.get_ipython().events.register('post_run_cell', self.post_run)

  def close(self) -> None:
    """Destroy the _CellTimer and its notebook callbacks."""
    import IPython
    IPython.get_ipython().events.unregister('pre_run_cell', self.pre_run)
    IPython.get_ipython().events.unregister('post_run_cell', self.post_run)

  def pre_run(self, unused_info: Any) -> None:
    """Start a timer for the notebook cell execution."""
    self.start_time = time.monotonic()

  def post_run(self, unused_result: Any) -> None:
    """Start the timer for the notebook cell execution."""
    import IPython
    elapsed_time = time.monotonic() - self.start_time
    input_index = IPython.get_ipython().execution_count - 1
    self.elapsed_times[input_index] = elapsed_time

  def show_times(self, n: int | None = None, sort: bool = False) -> None:
    """Print notebook cell timings."""
    import IPython
    print(f'# Total time: {sum(self.elapsed_times.values()):.2f} s')
    times = list(self.elapsed_times.items())
    times = sorted(times, key=lambda x: x[sort], reverse=sort)
    # https://github.com/ipython/ipython/blob/master/IPython/core/history.py
    # https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.history.html
    session = IPython.get_ipython().history_manager.session_number
    history_range = IPython.get_ipython().history_manager.get_range(session)
    inputs = {index: text for unused_session, index, text in history_range}
    for input_index, elapsed_time in itertools.islice(times, n):
      cell_input = inputs[input_index]
      text = repr(cell_input)[1:-1][:59]
      text = re.sub(r'# pylint:', '# pylinX:', text)
      text = re.sub(r'[A-Za-z\\]+$', '', text)
      print(f'# In[{input_index:3}] {text:59} {elapsed_time:6.3f} s')


def start_timing_notebook_cells() -> None:
  """Start timing of Jupyter notebook cells.

  Place in an early notebook cell.  See also `show_notebook_cell_top_times`.
  """
  import IPython
  if IPython.get_ipython():
    if _CellTimer.instance:
      _CellTimer.instance.close()
    _CellTimer.instance = _CellTimer()


def show_notebook_cell_top_times() -> None:
  """Print summary of timings for Jupyter notebook cells.

  Place in a late notebook cell.  See also `start_timing_notebook_cells`.
  """
  if _CellTimer.instance:
    _CellTimer.instance.show_times(n=20, sort=True)


# ** Timing


def get_time_and_result(func: Callable[[], _T], *,
                        max_repeat: int = 10,
                        max_time: float = 2.0) -> tuple[float, _T]:
  """Call the function repeatedly to determine its minimum run time.

  If the measured run time is small, more precise time estimates are obtained
  by considering batches of function calls (with automatically increasing
  batch size).

  Args:
    func: Function to time.
    max_repeat: Maximum number of batch measurements across which to compute
      the minimum value.
    max_time: Desired maximum total time in obtaining timing measurements.
      If set to zero, a single timing measurement is taken.

  Returns:
    minimum_time: The smallest time (in seconds) measured across the repeated
      calls to `func` (divided by batch size if batches are introduced).
    result: The value returned by the last call to `func`.

  >>> elapsed, result = get_time_and_result(lambda: 11 + 22)
  >>> assert elapsed < 0.01, elapsed
  >>> result
  33
  """
  assert callable(func) and max_repeat > 0 and max_time >= 0.0
  result = None
  gc_was_enabled = gc.isenabled()
  batch_size = 1
  smallest_acceptable_batch_time = 0.01  # All times are in seconds.

  try:
    gc.disable()
    while True:
      num_repeat = 0
      sum_time = 0.0
      min_time = math.inf
      start = time.monotonic()
      while num_repeat < max_repeat and sum_time <= max_time:
        for _ in range(batch_size):
          result = func()
        stop = time.monotonic()
        elapsed = stop - start
        start = stop
        num_repeat += 1
        sum_time += elapsed
        min_time = min(min_time, elapsed)
      if min_time >= min(smallest_acceptable_batch_time, max_time):
        break
      batch_size *= 10

  finally:
    if gc_was_enabled:
      gc.enable()

  return min_time / batch_size, typing.cast(_T, result)


def get_time(func: Callable[[], Any], **kwargs: Any) -> float:
  """Return the minimum execution time when repeatedly calling `func`.

  >>> elapsed = get_time(lambda: time.sleep(0.2), max_repeat=1)
  >>> assert 0.15 < elapsed < 0.25, elapsed
  """
  return get_time_and_result(func, **kwargs)[0]


def format_float(value: float, precision: int) -> str:
  """Return non-scientific repr. of value with specified digits of precision.

  >>> format_float(1234, 3)
  '1230'

  >>> format_float(0.1234, 3)
  '0.123'

  >>> format_float(0.1230, 3)
  '0.123'

  >>> format_float(0.01236, 3)
  '0.0124'

  >>> format_float(123, 3)
  '123'

  >>> format_float(120, 3)
  '120'
  """
  text = np.format_float_positional(
      value, fractional=False, unique=False, precision=precision)
  return text.rstrip('.')


def print_time(func: Callable[[], Any], **kwargs: Any) -> None:
  r"""Print the minimum execution time when repeatedly calling `func`.

  >>> with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as m:
  ...   print_time(lambda: 1)
  ...   assert re.fullmatch(r'[0-9.]+ .?s\n', m.getvalue()), m.getvalue()

  """
  min_time = get_time(func, **kwargs)
  # print(f'{min_time:.3f} s', flush=True)
  text = (f'{format_float(min_time, 3)} s' if min_time >= 1.0 else
          f'{format_float(min_time*1e3, 3)} ms' if min_time > 1.0 / 1e3 else
          f'{format_float(min_time*1e6, 3)} µs' if min_time > 1.0 / 1e6 else
          f'{format_float(min_time*1e6, 2)} µs')
  print(text, flush=True)


# ** Profiling


def prun(func: Callable[[], Any],
         mode: str = 'tottime',
         top: int | None = None) -> None:
  """Profile the function call and print reformatted statistics.

  >>> with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as m:
  ...   prun(lambda: np.linalg.qr(np.ones((400, 400))))
  ...   lines = m.getvalue().splitlines()
  >>> assert lines[0].startswith('# Prun: tottime ')
  >>> assert 'overall_cumtime' in lines[0]
  """
  assert callable(func)
  assert mode in ('original', 'full', 'tottime'), mode
  site_packages = np.__path__[0][:-5]
  assert site_packages.endswith(('/site-packages/', '/dist-packages/'))

  profile = cProfile.Profile()
  try:
    profile.enable()
    func()
  finally:
    profile.disable()

  with io.StringIO() as string_io:
    args = (top,) if top is not None else ()
    pstats.Stats(profile, stream=string_io).sort_stats(
        'tottime').print_stats(*args)
    lines = string_io.getvalue().strip('\n').splitlines()

  if mode == 'original':
    print('\n'.join(lines))
    return

  def beautify_function_name(name: str) -> str:
    name = re.sub(r'^\{built-in method (.*)\}$', r'\1 (built-in)', name)
    name = re.sub(r"^\{method '(.*)' of '(.*)' objects\}$", r'\2.\1', name)
    name = re.sub(r'^\{function (\S+) at (0x\w+)\}$', r'\1', name)
    name = re.sub(r'^<ipython-input[-\w]+>:\d+\((.*)\)$', r'\1', name)
    name = re.sub(r'^([^:()]+):(\d+)\((.+)\)$', r'\3 (\1:\2)', name)
    name = re.sub(r'^\{(\S+)\}$', r'\1', name)
    name = re.sub(r' \(/tmp/ipykernel.*\.py:', r' (/tmp/ipykernel:', name)
    name = name.replace(site_packages, '')
    return name

  output = []
  overall_time = 0.0
  post_header = False
  for line in lines:
    if post_header:
      tottime_str, cumtime_str, name = assert_value(re.fullmatch(
          r'\s*\S+\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+(\S.*)', line)).groups()
      tottime, cumtime = float(tottime_str), float(cumtime_str)
      beautified_name = beautify_function_name(name)
      overall_time += 1e-6
      significant_time = (tottime / overall_time > 0.05 or
                          0.05 < cumtime / overall_time < 0.95)
      if top is not None or significant_time:
        if mode == 'tottime':
          output.append(f'     {tottime:8.3f} {cumtime:8.3f} {beautified_name}')
        else:  # mode == 'full'
          output.append(line.replace(name, beautified_name))
    elif ' function calls ' in line:
      overall_time = float(
          assert_value(re.search(r' in (\S+) seconds', line)).group(1))
      output.append(f'Prun: tottime {overall_time:8.3f} overall_cumtime')
    elif line.lstrip().startswith('ncalls '):
      if mode == 'full':
        output.append(line)
      post_header = True

  print('\n'.join([f'#{" " * bool(line)}' + line for line in output]))


# ** Class objects


class OrderedEnum(enum.Enum):
  """Enum supporting comparisons, sort, and max.

  >>> class MyEnum(OrderedEnum):
  ...   VALUE1 = enum.auto()
  >>> assert hash(MyEnum.VALUE1)
  >>> assert MyEnum.VALUE1 <= MyEnum.VALUE1
  """
  # https://docs.python.org/3/library/enum.html

  def __ge__(self, other: OrderedEnum) -> bool:
    if self.__class__ is not other.__class__:
      return NotImplemented
    return typing.cast(bool, self.value >= other.value)

  def __gt__(self, other: OrderedEnum) -> bool:
    if self.__class__ is not other.__class__:
      return NotImplemented
    return typing.cast(bool, self.value > other.value)

  def __le__(self, other: OrderedEnum) -> bool:
    if self.__class__ is not other.__class__:
      return NotImplemented
    return typing.cast(bool, self.value <= other.value)

  def __lt__(self, other: OrderedEnum) -> bool:
    if self.__class__ is not other.__class__:
      return NotImplemented
    return typing.cast(bool, self.value < other.value)


# ** Operations on iterables


def repeat_each(iterable: Iterable[_T], n: int) -> Iterator[_T]:
  """Repeat each element of iterable 'n' times.

  >>> list(repeat_each(list('abc'), 2))
  ['a', 'a', 'b', 'b', 'c', 'c']

  >>> ''.join(itertools.islice(repeat_each(itertools.cycle('abcd'), 4), 30))
  'aaaabbbbccccddddaaaabbbbccccdd'
  """
  # https://stackoverflow.com/a/65071833
  return itertools.chain.from_iterable(zip(*itertools.tee(iterable, n)))


def only(iterable: Iterable[_T]) -> _T:
  """Return the first element and asserts that there are no more.

  >>> only(range(1))
  0

  >>> only(range(2))
  Traceback (most recent call last):
  ...
  ValueError: [0, 1, ...] has more than one element

  >>> only(range(0))
  Traceback (most recent call last):
  ...
  StopIteration
  """
  # Or use: return (lambda x: x)(*iterable)
  iterator = iter(iterable)
  first = next(iterator)
  missing = object()
  second = next(iterator, missing)
  if second != missing:
    raise ValueError(f'[{first}, {second}, ...] has more than one element')
  return first


def grouped(iterable: Iterable[_T],
            n: int,
            fillvalue: _T | None = None,
            ) -> Iterator[tuple[_T | None, ...]]:
  """Return elements collected into fixed-length chunks.

  >>> list(grouped('ABCDEFG', 3, 'x'))
  [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]

  >>> list(grouped(range(5), 3))
  [(0, 1, 2), (3, 4, None)]

  >>> list(grouped(range(5), 3, fillvalue=9))
  [(0, 1, 2), (3, 4, 9)]

  >>> list(grouped(range(6), 3))
  [(0, 1, 2), (3, 4, 5)]

  >>> list(grouped([], 2))
  []
  """
  # From grouper() in https://docs.python.org/3/library/itertools.html.
  iters = [iter(iterable)] * n
  return itertools.zip_longest(*iters, fillvalue=fillvalue)


def chunked(iterable: Iterable[_T],
            n: int | None = None,
            ) -> Iterator[tuple[_T, ...]]:
  """Return elements collected as tuples of length at most 'n' if not None.

  >>> list(chunked('ABCDEFG', 3))
  [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

  >>> list(chunked(range(5), 3))
  [(0, 1, 2), (3, 4)]

  >>> list(chunked(range(5)))
  [(0, 1, 2, 3, 4)]

  >>> list(chunked([]))
  []
  """

  def take(n: int, iterable: Iterable[_T]) -> tuple[_T, ...]:
    return tuple(itertools.islice(iterable, n))

  return iter(functools.partial(take, n, iter(iterable)), ())


def sliding_window(iterable: Iterable[_T], n: int) -> Iterator[tuple[_T, ...]]:
  """Return overlapping tuples of length `n` from `iterable`.

  >>> list(sliding_window('ABCDEF', 4))
  [('A', 'B', 'C', 'D'), ('B', 'C', 'D', 'E'), ('C', 'D', 'E', 'F')]

  >>> list(sliding_window('ABCDE', 1))
  [('A',), ('B',), ('C',), ('D',), ('E',)]

  >>> list(sliding_window('ABCDE', 8))
  []
  >>> list(sliding_window('A', 2))
  []
  >>> list(sliding_window('', 1))
  []
  """
  # From https://docs.python.org/3/library/itertools.html.
  it = iter(iterable)
  window = collections.deque(itertools.islice(it, n), maxlen=n)
  if len(window) == n:
    yield tuple(window)
  for x in it:
    window.append(x)
    yield tuple(window)


def powerset(iterable: Iterable[_T]) -> Iterator[tuple[_T, ...]]:
  """Return all subsets of iterable.

  >>> list(powerset([1, 2, 3]))
  [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

  >>> list(powerset([]))
  [()]
  """
  # From https://docs.python.org/3/library/itertools.html.
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s) + 1))


def unique_permutations(elements: Sequence[_T]) -> Iterator[tuple[_T, ...]]:
  """Yields unique permutations; see https://stackoverflow.com/a/30558049 .

  >>> list(unique_permutations([1, 2, 1]))
  [(1, 1, 2), (1, 2, 1), (2, 1, 1)]
  """
  if len(elements) == 1:
    yield (elements[0],)
  else:
    unique_elements = set(elements)
    for first_element in unique_elements:
      remaining_elements = list(elements)
      remaining_elements.remove(first_element)
      for sub_permutation in unique_permutations(remaining_elements):
        yield (first_element, *sub_permutation)


def peek_first(iterator: Iterable[_T]) -> tuple[_T, Iterable[_T]]:
  """Given an iterator, return first element and re-initialized iterator.

  Example:
    first_image, images = peek_first(images)

  Args:
    iterator: An input iterator or iterable.

  Returns:
    A tuple (first_element, iterator_reinitialized) containing:
      first_element: The first element of the input.
      iterator_reinitialized: A clone of the original iterator/iterable.

  >>> value, iter = peek_first(range(5))
  >>> value
  0
  >>> list(iter)
  [0, 1, 2, 3, 4]
  """
  # Inspired from https://stackoverflow.com/a/12059829
  peeker, iterator_reinitialized = itertools.tee(iterator)
  first = next(peeker)
  return first, iterator_reinitialized


# ** Temporary variable assignment


@contextlib.contextmanager
def temporary_assignment(variables: dict[str, Any], name: str,
                         value: Any) -> Iterator[None]:
  """Temporarily assign `value` to the variable named `name` in `variables`.

  Args:
    variables: Usually the `globals()` of the caller module.  Note that a
      function-scope `locals()` does not work as it should not be modified.
    name: Name of the variable in `variables` to temporarily assign.
    value: Value assigned to `name` in the lifetime of the context.

  >>> var = 1
  >>> with temporary_assignment(globals(), 'var', 2):
  ...   check_eq(var, 2)
  >>> check_eq(var, 1)

  >>> assert 'var2' not in globals()
  >>> with temporary_assignment(globals(), 'var2', '1'):
  ...   check_eq(var2, '1')  # noqa
  >>> assert 'var2' not in globals()
  """
  # https://stackoverflow.com/a/57226721.
  old_value = variables.get(name, _UNDEFINED)
  variables[name] = value
  yield
  if old_value is _UNDEFINED:
    del variables[name]
  else:
    variables[name] = old_value


# ** Meta programming


@typing.overload  # Bare decorator.
def noop_decorator(func: _F) -> _F:
  ...


@typing.overload  # Decorator with arguments.
def noop_decorator(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
  ...


def noop_decorator(*args: Any, **kwargs: Any) -> Any:
  """Return function decorated with no-op; invokable with or without args.

  >>> @noop_decorator
  ... def func1(x): return x * 10

  >>> @noop_decorator()
  ... def func2(x): return x * 10

  >>> @noop_decorator(2, 3)
  ... def func3(x): return x * 10

  >>> @noop_decorator(keyword=True)
  ... def func4(x): return x * 10

  >>> check_eq(func1(1) + func2(1) + func3(1) + func4(1), 40)
  """
  if len(args) != 1 or not callable(args[0]) or kwargs:
    return noop_decorator  # Decorator is invoked with arguments; ignore them.
  func: Callable[..., Any] = args[0]
  return func


def terse_str(cls: type) -> type:
  """Decorator for a dataclasses.dataclass, which defines a custom str().

  >>> @terse_str
  ... @dataclasses.dataclass
  ... class TestTerseStr:
  ...   a: int = 3
  ...   b: list[str] = dataclasses.field(default_factory=lambda: ['g', 'h'])

  >>> str(TestTerseStr())
  'TestTerseStr()'

  >>> str(TestTerseStr(a=4))
  'TestTerseStr(a=4)'

  >>> str(TestTerseStr(b=['i', 'j']))
  "TestTerseStr(b=['i', 'j'])"
  """
  assert isinstance(cls, type)
  default_for_field = {
      field.name: (field.default_factory() if callable(field.default_factory)
                   else field.default)
      for field in dataclasses.fields(cls)
      if field.repr
  }

  def __str__(self: Any) -> str:
    """Return a string containing only the non-default field values."""
    text = ', '.join(f'{name}={getattr(self, name)!r}'
                     for name, default in default_for_field.items()
                     if getattr(self, name) != default)
    return f'{type(self).__name__}({text})'

  setattr(cls, '__str__', __str__)
  return cls


# ** Memoization


def selective_lru_cache(*args: Any, ignore_kwargs: tuple[str, ...] = (),
                        **kwargs: Any) -> Callable[[_F], _F]:
  """Like `functools.lru_cache` but memoization can ignore specified kwargs.

  Because `lru_cache` is unaware of default keyword values, it is recommended
  that the parameters named in `ignore_kwargs` not have defaults in the
  decorated function.  Inspired by https://stackoverflow.com/a/30738279

  >>> @selective_lru_cache(ignore_kwargs=('kw1'))
  ... def func(arg1: int, *, kw1: bool):
  ...   print(arg1, kw1)
  ...   return arg1

  >>> func(1, kw1=True)
  1 True
  1

  >>> func(2, kw1=False)
  2 False
  2

  >>> func(1, kw1=False)
  1
  """
  lru_decorator: Callable[[_F], _F] = functools.lru_cache(*args, **kwargs)

  class Equals:
    """Wraps an object to replace its equality test and hash function."""

    def __init__(self, o: Any) -> None:
      self.obj = o

    def __eq__(self, other: Any) -> bool:
      return True

    def __hash__(self) -> int:
      return 0

  def decorator(func: _F) -> _F:
    @lru_decorator
    def helper(*args: Any, **kwargs: Any) -> Any:
      kwargs = {k: (v.obj if k in ignore_kwargs else v)
                for k, v in kwargs.items()}
      return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      kwargs = {k: (Equals(v) if k in ignore_kwargs else v)
                for k, v in kwargs.items()}
      return helper(*args, **kwargs)

    for attribute in ['cache_clear', 'cache_info', 'cache_parameters']:
      value = getattr(helper, attribute, None)
      if value:  # 'cache_parameters' added only in Python 3.9.
        setattr(wrapper, attribute, value)
    return typing.cast(_F, wrapper)

  return decorator


# ** Imports and modules


def create_module(module_name: str, elements: Iterable[Any] = ()) -> Any:
  """Return a new empty module (not associated with any file).

  >>> def some_function(*args, **kwargs): return 'success'
  >>> class Node:
  ...   def __init__(self): self.attrib = 2
  >>> test_module = create_module('test_module', [some_function, Node])
  >>> test_module.some_function(10)
  'success'
  >>> assert 'some_function' in dir(test_module)
  >>> help(test_module.some_function)
  Help on function some_function in module test_module:
  <BLANKLINE>
  some_function(*args, **kwargs)
  <BLANKLINE>
  >>> node = test_module.Node()
  >>> type(node)
  <class 'test_module.Node'>
  >>> node.attrib
  2
  """
  # https://stackoverflow.com/a/53080237
  module = sys.modules.get(module_name)
  if not module:
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    assert spec
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

  for element in elements:
    setattr(module, element.__name__, element)
    element.__module__ = module_name

  return module


# ** System functions


@contextlib.contextmanager
def timing(description: str = 'Timing') -> Iterator[None]:
  """Context that reports elapsed time.

  Example:
    with timing('List comprehension example'):
      _ = [i for i in range(10_000_000)]

  Args:
    description: A string to print before the elapsed time.

  Yields:
    None.

  >>> with timing('List comprehension example'):
  ...   _ = [i for i in range(10_000)]  # doctest:+ELLIPSIS
  List comprehension example: 0.00...
  """
  start = time.monotonic()
  yield
  elapsed_time = time.monotonic() - start
  print(f'{description}: {elapsed_time:.6f}')


def typename(o: Any) -> str:
  """Return the full name (including module) of the type of o.

  >>> typename(5)
  'int'

  >>> typename('text')
  'str'

  >>> typename(np.array([1]))
  'numpy.ndarray'
  """
  # https://stackoverflow.com/a/2020083
  name: str = o.__class__.__qualname__
  module = o.__class__.__module__
  return name if module in (None, 'builtins') else f'{module}.{name}'


def show_biggest_vars(variables: Mapping[str, Any], n: int = 10) -> None:
  """Print the variables with the largest memory allocation (in bytes).

  Usage:
    show_biggest_vars(globals())

  Args:
    variables: Dictionary of variables (often, `globals()`).
    n: The number of largest variables to list.

  >>> show_biggest_vars({'i': 12, 's': 'text', 'ar': np.ones((1000, 1000))})
  ... # doctest:+ELLIPSIS
  ar                       numpy.ndarray        ...
  s                        str                  ...
  i                        int                  ...
  """
  var = variables
  infos = [(name, sys.getsizeof(value), typename(value))
           for name, value in var.items()]
  infos.sort(key=lambda pair: pair[1], reverse=True)
  for name, size, vartype in infos[:n]:
    print(f'{name:24} {vartype:20} {size:_}')


# ** Mathematics


def as_float(a: _ArrayLike) -> _NDArray:
  """Convert non-floating-point array to floating-point type.

  Args:
    a: Input array.

  Returns:
    Array 'a' if it is already floating-point (np.float32 or np.float64),
    else 'a' converted to type np.float32 or np.float64 based on the necessary
    precision.  Note that 64-bit integers cannot be represented exactly.

  >>> as_float(np.array([1.0, 2.0]))
  array([1., 2.])

  >>> as_float(np.array([1.0, 2.0], dtype=np.float32))
  array([1., 2.], dtype=float32)

  >>> as_float(np.array([1.0, 2.0], dtype='float64'))
  array([1., 2.])

  >>> as_float(np.array([1, 2], dtype=np.uint8))
  array([1., 2.], dtype=float32)

  >>> as_float(np.array([1, 2], dtype=np.uint16))
  array([1., 2.], dtype=float32)

  >>> as_float(np.array([1, 2]))
  array([1., 2.])
  """
  a = np.asarray(a)
  if issubclass(a.dtype.type, np.floating):
    return a
  dtype = np.float64 if np.iinfo(a.dtype).bits >= 32 else np.float32
  return a.astype(dtype)


def normalize(a: _ArrayLike, axis: int | None = None) -> _NDArray:
  """Return array 'a' scaled such that its elements have unit 2-norm.

  Args:
    a: Input array.
    axis: If None, normalizes the entire matrix.  Otherwise, normalizes each
      element along the specified axis.

  Returns:
    An array such that its elements along 'axis' are rescaled to have L2 norm
    equal to 1.  Any element with zero norm is replaced by nan values.

  >>> normalize(np.array([10, 10, 0]))
  array([0.70710678, 0.70710678, 0.        ])

  >>> normalize([[0, 10], [10, 10]], axis=-1)
  array([[0.        , 1.        ],
         [0.70710678, 0.70710678]])

  >>> normalize([[0, 10], [10, 10]], axis=0)
  array([[0.        , 0.70710678],
         [1.        , 0.70710678]])

  >>> normalize([[0, 10], [10, 10]])
  array([[0.        , 0.57735027],
         [0.57735027, 0.57735027]])
  """
  a = np.asarray(a)
  norm = np.linalg.norm(a, axis=axis)
  if axis is not None:
    norm = np.expand_dims(norm, axis)
  with np.errstate(invalid='ignore'):
    return a / norm


def rms(a: _ArrayLike, axis: int | None = None) -> _NDArray:
  """Return the root mean square of the array values.

  >>> rms([3.0])
  3.0

  >>> rms([-3.0, 4.0])
  3.5355339059327378

  >>> rms([10, 11, 12])
  11.030261405182864

  >>> rms([[-1.0, 1.0], [0.0, -2.0]])
  1.224744871391589

  >>> rms([[-1.0, 1.0], [0.0, -2.0]], axis=-1)
  array([1.        , 1.41421356])
  """
  return np.sqrt(np.mean(np.square(as_float(a)), axis, dtype=np.float64))


def lenient_subtract(a: _ArrayLike, b: _ArrayLike) -> _NDArray:
  """Return a - b, but returns 0 where a and b are the same signed infinity.

  >>> inf = math.inf
  >>> lenient_subtract([3., 3., inf, inf, -inf, -inf],
  ...                  [1., inf, inf, -inf, inf, -inf])
  array([  2., -inf,   0.,  inf, -inf,   0.])
  """
  a = np.asarray(a)
  b = np.asarray(b)
  same_infinity = ((np.isposinf(a) & np.isposinf(b)) |
                   (np.isneginf(a) & np.isneginf(b)))
  return np.subtract(a, b, out=np.zeros_like(a), where=~same_infinity)


def print_array(a: _ArrayLike, **kwargs: Any) -> None:
  """Print the array.

  >>> print_array(np.arange(6).reshape(2, 3), file=sys.stdout)
  array([[0, 1, 2],
         [3, 4, 5]]) shape=(2, 3) dtype=int64
  """
  x = np.asarray(a)
  print_err(f'{repr(x)} shape={x.shape} dtype={x.dtype}', **kwargs)


def prime_factors(n: int) -> list[int]:
  """Return an ascending list of the (greather-than-one) prime factors of n.

  >>> prime_factors(1)
  []

  >>> prime_factors(2)
  [2]

  >>> prime_factors(4)
  [2, 2]

  >>> prime_factors(60)
  [2, 2, 3, 5]
  """
  factors = []
  d = 2
  while d * d <= n:
    while (n % d) == 0:
      factors.append(d)
      n //= d
    d += 1
  if n > 1:
    factors.append(n)
  return factors


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
  """Find the greatest common divisor using the extended Euclidean algorithm.

  Returns:
    A tuple (gcd, x, y) with the property that a * x + b * y = gcd.

  >>> extended_gcd(29, 71)
  (1, -22, 9)
  >>> (29 * -22) % 71
  1
  """
  prev_x, x = 1, 0
  prev_y, y = 0, 1
  while b:
    q = a // b
    x, prev_x = prev_x - q * x, x
    y, prev_y = prev_y - q * y, y
    a, b = b, a % b
  x, y = prev_x, prev_y
  return a, x, y


def modular_inverse(a: int, b: int) -> int:
  """Return the multiplicative inverse of 'a' with respect to the modulus 'b'.

  With the extended Euclidean algorithm, for the case that a and b are coprime,
  i.e. gcd(a, b) = 1, applying "modulo b" to both sides of a * x + b * y = 1
  results in (a * x) % b = 1, and hence 'x' is a modular multiplicative inverse
  of 'a' with respect to the modulus 'b'.
  See https://en.wikipedia.org/wiki/Modular_multiplicative_inverse

  >>> modular_inverse(29, 71)
  49
  >>> (29 * 49) % 71
  1
  """
  # Note: This becomes available as "pow(a, -1, mod=b)" in Python 3.8.
  gcd, x, unused_y = extended_gcd(a, b)
  check_eq(gcd, 1)
  return x % b


def diagnostic(a: _ArrayLike) -> str:
  """Return a diagnostic string summarizing the values in 'a' for debugging.

  Args:
    a: Input values; must be convertible to an np.ndarray of scalars.

  Returns:
    A string summarizing the different types of arithmetic values.

  >>> import textwrap
  >>> print(textwrap.fill(diagnostic(
  ...     [[math.nan, math.inf, -math.inf, -math.inf], [0, -1, 2, -0]])))
  shape=(2, 4) dtype=float64 size=8 nan=1 posinf=1 neginf=2 finite=4,
  min=-1.0, max=2.0, avg=0.25, sdv=1.25831) zero=2
  """
  a = np.asarray(a)
  dtype = a.dtype
  if dtype == bool:
    a = a.astype(np.uint8)
  finite = a[np.isfinite(a)]
  return (f'shape={a.shape} dtype={dtype} size={a.size}'
          f' nan={np.isnan(a).sum()}'
          f' posinf={np.isposinf(a).sum()}'
          f' neginf={np.isneginf(a).sum()}'
          f' finite{repr(Stats(finite))[10:]}'
          f' zero={(finite == 0).sum()}')


# ** Statistics


class Stats:
  r"""Statistics computed from numbers in an iterable.

  >>> Stats()
  Stats(size=0, min=inf, max=-inf, avg=nan, sdv=nan)

  >>> Stats([1.5])
  Stats(size=1, min=1.5, max=1.5, avg=1.5, sdv=0.0)

  >>> Stats(range(3, 5))
  Stats(size=2, min=3, max=4, avg=3.5, sdv=0.707107)

  >>> Stats([3.0, 4.0])
  Stats(size=2, min=3.0, max=4.0, avg=3.5, sdv=0.707107)

  >>> Stats([-12345., 2.0**20])
  Stats(size=2, min=-12345.0, max=1.04858e+06, avg=5.18116e+05, sdv=7.50184e+05)

  >>> print(Stats(range(55)))
  (         55)            0 : 54           av=27.0000      sd=16.0208

  >>> print(Stats())
  (          0)          inf : -inf         av=nan          sd=nan

  >>> str(Stats() + Stats([3.0]))
  '(          1)      3.00000 : 3.00000      av=3.00000      sd=0.00000'

  >>> print(f'{Stats([-12345., 2.0**20]):14.9}')
  (          2)       -12345.0 : 1048576.0      av=518115.5       sd=750184.433

  >>> print(f'{Stats([-12345., 2.0**20]):#10.4}')
  (          2) -1.234e+04 : 1.049e+06  av=5.181e+05  sd=7.502e+05

  >>> len(Stats([1, 2]))
  2
  >>> Stats([-2, 2]).rms()
  2.0

  >>> a = Stats([1, 2])
  >>> a.min(), a.max(), a.avg()
  (1, 2, 1.5)

  >>> stats1 = Stats([-3, 7])
  >>> stats2 = Stats([1.25e11 / 3, -1_234_567_890])
  >>> stats3 = stats1 + stats2 * 20_000_000
  >>> print(stats1, f'{stats2}', format(stats3), sep='\n')
  (          2)           -3 : 7            av=2.00000      sd=7.07107
  (          2) -1.23457e+09 : 4.16667e+10  av=2.02160e+10  sd=3.03358e+10
  ( 40_000_002) -1.23457e+09 : 4.16667e+10  av=2.02160e+10  sd=2.14506e+10

  >>> fmt = '9.3'
  >>> print(f'{stats1:{fmt}}', f'{stats2:{fmt}}', f'{stats3:{fmt}}', sep='\n')
  (          2)        -3 : 7         av=2.0       sd=7.07
  (          2) -1.23e+09 : 4.17e+10  av=2.02e+10  sd=3.03e+10
  ( 40_000_002) -1.23e+09 : 4.17e+10  av=2.02e+10  sd=2.15e+10
  """

  _size: int
  _sum: float
  _sum2: float
  _min: float
  _max: float

  def __init__(self, *args: Any) -> None:
    if not args:
      self._size = 0
      self._sum = 0.0
      self._sum2 = 0.0
      self._min = math.inf
      self._max = -math.inf
    elif len(args) == 1:
      a = array_always(args[0])
      self._size = a.size
      self._sum = a.sum()
      self._sum2 = np.square(a).sum()
      self._min = a.min() if a.size > 0 else math.inf
      self._max = a.max() if a.size > 0 else -math.inf
    else:
      (self._size, self._sum, self._sum2, self._min, self._max) = args

  def sum(self) -> float:
    """Return the sum of the values.

    >>> f'{Stats([3.5, 2.2, 4.4]).sum():.8g}'
    '10.1'
    """
    return self._sum

  def min(self) -> float:
    """Return the minimum value.

    >>> Stats([3.5, 2.2, 4.4]).min()
    2.2
    """
    return self._min

  def max(self) -> float:
    """Return the maximum value.

    >>> Stats([3.5, 2.2, 4.4]).max()
    4.4
    """
    return self._max

  def avg(self) -> float:
    """Return the average.

    >>> Stats([1, 1, 4]).avg()
    2.0
    """
    return math.nan if self._size == 0 else self._sum / self._size

  def ssd(self) -> float:
    """Return the sum of squared deviations.

    >>> Stats([1, 1, 4]).ssd()
    6.0
    """
    return (math.nan if self._size == 0 else
            max(self._sum2 - self._sum**2 / self._size, 0))

  def var(self) -> float:
    """Return the unbiased estimate of variance, as in np.var(a, ddof=1).

    >>> Stats([1, 1, 4]).var()
    3.0
    """
    return (math.nan if self._size == 0 else
            0.0 if self._size == 1 else
            self.ssd() / (self._size - 1))

  def sdv(self) -> float:
    """Return the unbiased standard deviation as in np.std(a, ddof=1).

    >>> Stats([1, 1, 4]).sdv()
    1.7320508075688772
    """
    return self.var()**0.5  # type: ignore[no-any-return]

  def rms(self) -> float:
    """Return the root-mean-square.

    >>> Stats([1, 1, 4]).rms()
    2.449489742783178
    >>> Stats([-1, 1]).rms()
    1.0
    """
    if self._size == 0:
      return 0.0
    return (self._sum2 / self._size)**0.5  # type: ignore[no-any-return]

  def __format__(self, format_spec: str = '') -> str:
    """Return a summary of the statistics (size, min, max, avg, sdv)."""
    fmt = format_spec if format_spec else '#12.6'
    fmt_int = fmt[:fmt.find('.')] if fmt.find('.') >= 0 else ''
    fmt_min = fmt if isinstance(self._min, np.floating) else fmt_int
    fmt_max = fmt if isinstance(self._max, np.floating) else fmt_int
    return (f'({self._size:11_})'
            f' {self._min:{fmt_min}} :'
            f' {self._max:<{fmt_max}}'
            f' av={self.avg():<{fmt}}'
            f' sd={self.sdv():<{fmt}}').rstrip()

  def __str__(self) -> str:
    return self.__format__()

  def __repr__(self) -> str:
    fmt = '.6'
    fmt_int = ''
    fmt_min = fmt if isinstance(self._min, np.floating) else fmt_int
    fmt_max = fmt if isinstance(self._max, np.floating) else fmt_int
    return (f'Stats(size={self._size}, '
            f'min={self._min:{fmt_min}}, '
            f'max={self._max:{fmt_max}}, '
            f'avg={self.avg():{fmt}}, '
            f'sdv={self.sdv():{fmt}})')

  def __len__(self) -> int:
    return self._size

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Stats):
      return NotImplemented
    return ((self._size, self._sum, self._sum2, self._min, self._max) ==
            (other._size, other._sum, other._sum2, other._min, other._max))

  def __add__(self, other: Stats) -> Stats:
    """Return combined statistics.

    >>> Stats([2, -1]) + Stats([7, 5]) == Stats([-1, 2, 5, 7])
    True
    """
    return Stats(self._size + other._size, self._sum + other._sum,
                 self._sum2 + other._sum2, min(self._min, other._min),
                 max(self._max, other._max))

  def __mul__(self, n: int) -> Stats:
    """Return statistics whereby each element appears 'n' times.

    >>> Stats([4, -2]) * 3 == Stats([-2, -2, -2, 4, 4, 4])
    True
    """
    return Stats(
        self._size * n, self._sum * n, self._sum2 * n, self._min, self._max)


# ** Numpy operations


def array_always(a: _ArrayLike | Iterable[_ArrayLike]) -> _NDArray:
  """Return a numpy array even if a is an iterator of subarrays.

  >>> array_always(np.array([[1, 2], [3, 4]]))
  array([[1, 2],
         [3, 4]])

  >>> array_always(range(3) for _ in range(2))
  array([[0, 1, 2],
         [0, 1, 2]])

  >>> array_always(np.array([[1, 2], [3, 4]]))
  array([[1, 2],
         [3, 4]])
  """
  if isinstance(a, collections.abc.Iterator):
    return np.array(tuple(a))
  return np.asarray(a)


def bounding_slices(a: _ArrayLike) -> tuple[slice, ...]:
  """Return the slices that bound the nonzero elements of array.

  >>> bounding_slices(())
  (slice(0, 0, None),)

  >>> bounding_slices(np.ones(0))
  (slice(0, 0, None),)

  >>> bounding_slices(np.ones((0, 10)))
  (slice(0, 0, None), slice(0, 0, None))

  >>> bounding_slices(32.0)
  (slice(0, 1, None),)

  >>> bounding_slices([0.0, 0.0, 0.0, 0.5, 1.5, 0.0, 2.5, 0.0, 0.0])
  (slice(3, 7, None),)

  >>> a = np.array([0, 0, 6, 7, 0, 0])
  >>> a[bounding_slices(a)]
  array([6, 7])

  >>> a = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
  >>> a[bounding_slices(a)]
  array([[1, 1]])

  >>> bounding_slices([[[0, 0], [0, 1]], [[0, 0], [0, 0]]])
  (slice(0, 1, None), slice(1, 2, None), slice(1, 2, None))
  """
  a = np.atleast_1d(a)
  slices = []
  for dim in range(a.ndim):
    line = a.any(axis=tuple(i for i in range(a.ndim) if i != dim))
    indices = line.nonzero()[0]
    if indices.size:
      vmin, vmax = indices[[0, -1]]
      slices.append(slice(vmin, vmax + 1))
    else:
      slices.append(slice(0, 0))  # Empty slice.
  return tuple(slices)


def broadcast_block(a: _NDArray,
                    block_shape: int | tuple[int, ...]) -> _NDArray:
  """Return an array view where each element of 'a' is repeated as a block.

  Args:
    a: input array of any dimension.
    block_shape: shape for the block that each element of 'a' becomes.  If a
      scalar value, all block dimensions are assigned this value.

  Returns:
    an array view with shape "a.shape * block_shape".

  >>> print(broadcast_block(np.arange(8).reshape(2, 4), (2, 3)))
  [[0 0 0 1 1 1 2 2 2 3 3 3]
   [0 0 0 1 1 1 2 2 2 3 3 3]
   [4 4 4 5 5 5 6 6 6 7 7 7]
   [4 4 4 5 5 5 6 6 6 7 7 7]]

  >>> a = np.arange(6).reshape(2, 3)
  >>> result = broadcast_block(a, (2, 3))
  >>> result.shape
  (4, 9)
  >>> np.all(result == np.kron(a, np.ones((2, 3), dtype=a.dtype)))
  True
  """
  block_shape2 = np.broadcast_to(np.asarray(block_shape), (a.ndim,))
  # Inspired from https://stackoverflow.com/a/52339952
  # and https://stackoverflow.com/a/52346065
  shape1 = tuple(v for pair in zip(a.shape, (1,) * a.ndim) for v in pair)
  shape2 = tuple(v for pair in zip(a.shape, block_shape2) for v in pair)
  final_shape = a.shape * block_shape2
  return np.broadcast_to(a.reshape(shape1), shape2).reshape(final_shape)


def np_int_from_ch(a: _ArrayLike, int_from_ch: Mapping[str, int],
                   dtype: _DTypeLike = None) -> _NDArray:
  """Return array of integers by mapping from array of characters.

  >>> np_int_from_ch(np.array(list('abcab')), {'a': 0, 'b': 1, 'c': 2})
  array([0, 1, 2, 0, 1])
  """
  # Adapted from https://stackoverflow.com/a/49566980
  a = np.asarray(a).view(np.int32)
  max_ch = max(a.max(), max(ord(ch) for ch in int_from_ch))
  lookup = np.zeros(max_ch + 1, dtype=dtype or np.int64)
  for ch, value in int_from_ch.items():
    lookup[ord(ch)] = value
  return lookup[a]


def grid_from_string(string: str,
                     int_from_ch: Mapping[str, int] | None = None,
                     dtype: _DTypeLike = None) -> _NDArray:
  r"""Return a 2D array created from a multiline string.

  Args:
    string: Nonempty lines correspond to the rows of the grid, with one chr
      per grid element.
    int_from_ch: Mapping from the chr in string to integers in the resulting
      grid; if None, the grid contains chr elements (dtype='<U1').
    dtype: Integer element type for the result of int_from_ch.

  >>> string = '..B\nB.A\n'
  >>> g = grid_from_string(string)
  >>> g, g.nbytes
  (array([['.', '.', 'B'],
         ['B', '.', 'A']], dtype='<U1'), 24)

  >>> g = grid_from_string(string, {'.': 0, 'A': 1, 'B': 2})
  >>> g, g.nbytes
  (array([[0, 0, 2],
         [2, 0, 1]]), 48)

  >>> g = grid_from_string(string, {'.': 0, 'A': 1, 'B': 2}, dtype=np.uint8)
  >>> g, g.nbytes
  (array([[0, 0, 2],
         [2, 0, 1]], dtype=uint8), 6)
  """
  # grid = np.array(list(map(list, string.strip('\n').split('\n'))))  # Slow.
  lines = string.strip('\n').splitlines()
  height, width = len(lines), len(lines[0])
  grid = np.empty((height, width), dtype='U1')
  dtype_for_row = f'U{width}'
  for i, line in enumerate(lines):
    grid[i].view(dtype_for_row)[0] = line

  if int_from_ch is None:
    assert dtype is None
  else:
    grid = np_int_from_ch(grid, int_from_ch, dtype=dtype)
  return grid


def string_from_grid(grid: _ArrayLike,
                     ch_from_int: Mapping[int, str] | None = None) -> str:
  r"""Return a multiline string created from a 2D array.

  Args:
    grid: 2D array-like data containing either chr or integers.
    ch_from_int: Mapping from each integer in grid to the chr in the resulting
      string; if None, the grid must contain str or byte elements.

  >>> string_from_grid([[0, 1], [0, 0]], {0: '.', 1: '#'})
  '.#\n..'

  >>> string_from_grid([['a', 'b', 'c'], ['d', 'e', 'f']])
  'abc\ndef'

  >>> string_from_grid([[b'A', b'B'], [b'C', b'D']])
  'AB\nCD'
  """
  grid = np.asarray(grid)
  check_eq(grid.ndim, 2)
  lines = []
  for y in range(grid.shape[0]):
    if ch_from_int is None:
      if grid.dtype.kind == 'S':  # or dtype.type == np.bytes_
        line = b''.join(grid[y]).decode('ascii')
      else:
        line = ''.join(grid[y])
    else:
      line = ''.join(ch_from_int[elem] for elem in grid[y])
    lines.append(line)
  return '\n'.join(lines)


def grid_from_indices(
    iterable_or_map: Iterable[Sequence[int]] | Mapping[Sequence[int], Any],
    background: Any = 0,
    foreground: Any = 1,
    indices_min: int | Sequence[int] | None = None,
    indices_max: int | Sequence[int] | None = None,
    pad: int | Sequence[int] = 0,
    dtype: _DTypeLike = None,
) -> _NDArray:
  r"""Return an array from (sparse) indices or from a map {index: value}.

  Indices are sequences of integers with some length D, which determines the
  dimensionality of the output array.  The array shape is computed by bounding
  the range of index coordinates in each dimension (which may be overriden by
  'indices_min' and 'indices_max') and is adjusted by the 'pad' parameter.

  Args:
    iterable_or_map: Iterable of indices or a mapping from indices to values.
    background: Value assigned to the array elements not in 'iterable_or_map'.
    foreground: If 'iterable_or_map' is an iterable, the array value assigned to
      its indices.
    indices_min: For each dimension, the index coordinate that gets mapped to
      coordinate zero in the array.  Replicated if an integer.
    indices_max: For each dimension, the index coordinate that gets mapped to
      the last coordinate in the array.  Replicated if an integer.
    pad: For each dimension d, number of additional slices of 'background'
      values before and after the range [indices_min[d], indices_max[d]].
    dtype: Data type of the output array.

  Returns:
    A D-dimensional numpy array initialized with the value 'background' and
    then sparsely assigned the elements in the parameter 'iterable_or_map'
    (using 'foreground' value if an iterable, or the map values if a map).
    By default, array spans a tight bounding box of the indices, but these
    bounds can be overridden using 'indices_min', 'indices_max', and 'pad'.

  >>> l = [(-1, -2), (-1, 1), (1, 0)]
  >>> grid_from_indices(l)
  array([[1, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 1, 0]])

  >>> grid_from_indices(l, indices_max=(1, 2))
  array([[1, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0]])

  >>> grid_from_indices(l, foreground='#', background='.')
  array([['#', '.', '.', '#'],
         ['.', '.', '.', '.'],
         ['.', '.', '#', '.']], dtype='<U1')

  >>> l = [5, -2, 1]
  >>> grid_from_indices(l, pad=1)
  array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])

  >>> grid_from_indices(l, indices_min=-4, indices_max=5)
  array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])

  >>> l = [(1, 1, 1), (2, 2, 2), (2, 1, 1)]
  >>> repr(grid_from_indices(l))
  'array([[[1, 0],\n        [0, 0]],\n\n       [[1, 0],\n        [0, 1]]])'

  >>> m = {(-1, 0): 'A', (0, 2): 'B', (1, 1): 'C'}
  >>> grid_from_indices(m, background=' ')
  array([['A', ' ', ' '],
         [' ', ' ', 'B'],
         [' ', 'C', ' ']], dtype='<U1')

  >>> grid_from_indices(m, background=' ', dtype='S1')
  array([[b'A', b' ', b' '],
         [b' ', b' ', b'B'],
         [b' ', b'C', b' ']], dtype='|S1')

  >>> grid_from_indices({(0, 0): (255, 1, 2), (1, 2): (3, 255, 4)})
  array([[[255,   1,   2],
          [  0,   0,   0],
          [  0,   0,   0]],
  <BLANKLINE>
         [[  0,   0,   0],
          [  0,   0,   0],
          [  3, 255,   4]]])
  """
  assert isinstance(iterable_or_map, collections.abc.Iterable)
  is_map = False
  if isinstance(iterable_or_map, collections.abc.Mapping):  # Help mypy.
    is_map = True
    mapping: Mapping[Sequence[int], Any] = iterable_or_map

  indices = np.array(list(iterable_or_map))
  if indices.ndim == 1:
    indices = indices[:, None]
  assert indices.ndim == 2 and np.issubdtype(indices.dtype, np.integer)
  i_min = (np.min(indices, axis=0) if indices_min is None else
           np.full(indices.shape[1], indices_min))
  i_max = (np.max(indices, axis=0) if indices_max is None else
           np.full(indices.shape[1], indices_max))
  a_pad = np.asarray(pad)
  shape = i_max - i_min + 2 * a_pad + 1
  offset = -i_min + a_pad
  elems = [next(iter(mapping.values()))] if is_map and mapping else []
  elems += [background, foreground]
  shape2 = (*shape, *np.broadcast(*elems).shape)
  del shape
  dtype = np.array(elems[0], dtype=dtype).dtype
  grid = np.full(shape2, background, dtype=dtype)
  indices += offset
  grid[tuple(indices.T)] = list(mapping.values()) if is_map else foreground
  return grid


def image_from_yx_map(map_yx_value: Mapping[tuple[int, int], Any],
                      background: Any,
                      cmap: Mapping[Any, tuple[int, int, int]],
                      pad: int | Sequence[int] = 0) -> _NDArray:
  """Return image from mapping {yx: value} and cmap = {value: rgb}.

  >>> m = {(2, 2): 'A', (2, 4): 'B', (1, 3): 'A'}
  >>> cmap = {'A': (100, 1, 2), 'B': (3, 200, 4), ' ': (235, 235, 235)}
  >>> image_from_yx_map(m, background=' ', cmap=cmap)
  array([[[235, 235, 235],
          [100,   1,   2],
          [235, 235, 235]],
  <BLANKLINE>
         [[100,   1,   2],
          [235, 235, 235],
          [  3, 200,   4]]], dtype=uint8)
  """
  array = grid_from_indices(map_yx_value, background=background, pad=pad)
  image = np.array([
      cmap[e] for e in array.flat  # pylint: disable=not-an-iterable
  ], dtype=np.uint8).reshape(*array.shape, 3)
  return image


def fit_shape(shape: Sequence[int], num: int) -> tuple[int, ...]:
  """Given 'shape' with one optional -1 dimension, make it fit 'num' elements.

  Args:
    shape: Input dimensions.  These must be positive, except that one dimension
      may be -1 to indicate that it should be computed.  If all dimensions are
      positive, these must satisfy np.prod(shape) >= num.
    num: Number of elements to fit into the output shape.

  Returns:
    The original 'shape' if all its dimensions are positive.  Otherwise, a
    new_shape where the unique dimension with value -1 is replaced by the
    smallest number such that np.prod(new_shape) >= num.

  >>> fit_shape((3, 4), 10)
  (3, 4)

  >>> fit_shape((5, 2), 11)
  Traceback (most recent call last):
  ...
  ValueError: (5, 2) is insufficiently large for 11 elements.

  >>> fit_shape((3, -1), 10)
  (3, 4)

  >>> fit_shape((-1, 10), 51)
  (6, 10)
  """
  shape = tuple(shape)
  if not all(dim > 0 for dim in shape if dim != -1):
    raise ValueError(f'Shape {shape} has non-positive dimensions.')
  if sum(dim == -1 for dim in shape) > 1:
    raise ValueError(f'More than one dimension in {shape} is -1.')
  if -1 in shape:
    slice_size = int(np.prod([dim for dim in shape if dim != -1]))
    shape = tuple((num + slice_size - 1) // slice_size if dim == -1 else dim
                  for dim in shape)
  elif np.prod(shape) < num:
    raise ValueError(f'{shape} is insufficiently large for {num} elements.')
  return shape


def assemble_arrays(arrays: Sequence[_NDArray],
                    shape: Sequence[int],
                    background: _ArrayLike = 0,
                    *,
                    align: str = 'center',
                    spacing: _ArrayLike = 0,
                    round_to_even: _ArrayLike = False) -> _NDArray:
  """Return an output array formed as a packed grid of input arrays.

  Args:
    arrays: Sequence of input arrays with the same data type and rank.  The
      arrays must have the same trailing dimensions arrays[].shape[len(shape):].
      The leading dimensions arrays[].shape[:len(shape)] may be different and
      these are packed together as a grid to form output.shape[:len(shape)].
    shape: Dimensions of the grid used to unravel the arrays before packing. The
      dimensions must be positive, with prod(shape) >= len(arrays).  One
      dimension of shape may be -1, in which case it is computed automatically
      as the smallest value such that prod(shape) >= len(arrays).
    background: Broadcastable value used for the unassigned elements of the
      output array.
    align: Relative position ('center', 'start', or 'stop') for each input array
      and for each axis within its output grid cell.  The value must be
      broadcastable onto the shape [len(arrays), len(shape)].
    spacing: Extra space between grid elements.  The value may be specified
      per-axis, i.e., it must be broadcastable onto the shape [len(shape)].
    round_to_even: If True, ensure that the final output dimension of each axis
      is even.  The value must be broadcastable onto the shape [len(shape)].

  Returns:
    A numpy output array of the same type as the input 'arrays', with
    output.shape = packed_shape + arrays[0].shape[len(shape):], where
    packed_shape is obtained by packing arrays[:].shape[:len(shape)] into a
    grid of the specified 'shape'.

  >>> assemble_arrays(
  ...    [np.array([[1, 2, 3]]), np.array([[5], [6]]), np.array([[7]]),
  ...     np.array([[8, 9]]), np.array([[3, 4, 5]])],
  ...    shape=(2, 3))
  array([[1, 2, 3, 0, 5, 0, 7],
         [0, 0, 0, 0, 6, 0, 0],
         [8, 9, 0, 3, 4, 5, 0]])
  """
  num = len(arrays)
  if num == 0:
    raise ValueError('There must be at least one input array.')
  shape = fit_shape(shape, num)
  if any(array.dtype != arrays[0].dtype for array in arrays):
    raise ValueError(f'Arrays {arrays} have different types.')
  tail_dims = arrays[0].shape[len(shape):]
  if any(array.shape[len(shape):] != tail_dims for array in arrays):
    raise ValueError(f'Shapes of {arrays} do not all end in {tail_dims}')
  align2 = np.broadcast_to(np.asarray(align), (num, len(shape)))
  spacing2 = np.broadcast_to(np.asarray(spacing), len(shape))
  round_to_even2 = np.broadcast_to(np.asarray(round_to_even), len(shape))
  del align, spacing, round_to_even

  # [shape] -> leading dimensions [:len(shape)] of each input array.
  head_dims = np.array([list(array.shape[:len(shape)]) for array in arrays] +
                       [[0] * len(shape)] * (np.prod(shape) - num)).reshape(
                           *shape, len(shape))

  # For each axis, find the length and position of each slice of input arrays.
  axis_lengths, axis_origins = [], []
  for axis, shape_axis in enumerate(shape):
    all_lengths = np.moveaxis(head_dims[..., axis], axis, 0)
    # Find the length of each slice along axis as the max over its arrays.
    lengths = all_lengths.max(axis=tuple(range(1, len(shape))))
    # Compute the dimension of the output axis.
    total_length = lengths.sum() + spacing2[axis] * (shape_axis - 1)
    if round_to_even2[axis] and total_length % 2 == 1:
      lengths[-1] += 1  # Lengthen the last slice so the axis dimension is even.
    axis_lengths.append(lengths)
    # Insert inter-element padding spaces.
    spaced_lengths = np.insert(lengths, 0, 0)
    spaced_lengths[1:-1] += spacing2[axis]
    # Compute slice positions along axis as cumulative sums of slice lengths.
    axis_origins.append(spaced_lengths.cumsum())

  # [shape] -> smallest corner coords in output array.
  origins = np.moveaxis(np.array(np.meshgrid(*axis_origins, indexing='ij')),
                        0, -1)

  # Initialize the output array.
  output_shape = tuple(origins[(-1,) * len(shape)]) + tail_dims
  output_array = np.full(output_shape, background, dtype=arrays[0].dtype)

  def offset(length: int, size: int, align: str) -> int:
    """Return an offset to align element of given size within cell of length."""
    remainder = length - size
    if align not in ('start', 'stop', 'center'):
      raise ValueError(f'Alignment {align} is not recognized.')
    return (0 if align == 'start' else
            remainder if align == 'stop' else remainder // 2)

  # Copy each input array to its packed, aligned location in the output array.
  for i, array in enumerate(arrays):
    coords = np.unravel_index(i, shape)
    slices = []
    for axis in range(len(shape)):
      start = origins[coords][axis]
      length = axis_lengths[axis][coords[axis]]
      extent = array.shape[axis]
      aligned_start = start + offset(length, extent, align2[i][axis])
      slices.append(slice(aligned_start, aligned_start + extent))
    output_array[tuple(slices)] = array

  return output_array


def shift(array: _ArrayLike,
          offset: _ArrayLike,
          constant_values: _ArrayLike = 0) -> _NDArray:
  """Return a copy of the array shifted by offset, with fill using constant.

  >>> array = np.arange(1, 13).reshape(3, 4)

  >>> shift(array, (1, 1))
  array([[0, 0, 0, 0],
         [0, 1, 2, 3],
         [0, 5, 6, 7]])

  >>> shift(array, (-1, -2), constant_values=-1)
  array([[ 7,  8, -1, -1],
         [11, 12, -1, -1],
         [-1, -1, -1, -1]])
  """
  array = np.asarray(array)
  offset = np.atleast_1d(offset)
  assert offset.shape == (array.ndim,)
  new_array = np.empty_like(array)

  def slice_axis(o: int) -> slice:
    return slice(o, None) if o >= 0 else slice(0, o)

  new_array[tuple(slice_axis(o) for o in offset)] = (
      array[tuple(slice_axis(-o) for o in offset)])

  for axis, o in enumerate(offset):
    new_array[(slice(None),) * axis +
              (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values

  return new_array


# ** Graph algorithms


class UnionFind(Generic[_T]):
  """Union-find is an efficient technique for tracking equivalence classes as
  pairs of elements are incrementally unified into the same class.  See
  https://en.wikipedia.org/wiki/Disjoint-set_data_structure .
  The implementation uses path compression but without weight-balancing, so the
  worst case time complexity is O(n*log(n)), but the average case is O(n).

  >>> union_find = UnionFind[str]()
  >>> union_find.find('hello')
  'hello'
  >>> union_find.same('hello', 'hello')
  True
  >>> union_find.same('hello', 'different')
  False
  >>> union_find.union('hello', 'there')
  >>> union_find.find('hello')
  'hello'
  >>> union_find.find('there')
  'hello'
  >>> union_find.same('hello', 'there')
  True
  >>> union_find.union('there', 'here')
  >>> union_find.same('hello', 'here')
  True
  """

  def __init__(self) -> None:
    self._rep: dict[_T, _T] = {}

  def union(self, a: _T, b: _T) -> None:
    """Merge the equivalence class of b into that of a.

    >>> union_find = UnionFind[int]()
    >>> union_find.union(1, 2)
    >>> assert union_find.same(1, 2) and not union_find.same(2, 3)
    """
    rep_a, rep_b = self.find(a), self.find(b)
    self._rep[rep_b] = rep_a

  def same(self, a: _T, b: _T) -> bool:
    """Return whether a and b are in the same equivalence class.

    >>> union_find = UnionFind[int]()
    >>> assert not union_find.same((1, 2), (2, 3))
    >>> union_find.union((1, 2), (2, 3))
    >>> assert union_find.same((1, 2), (2, 3))
    """
    result: bool = self.find(a) == self.find(b)
    return result

  def find(self, a: _T) -> _T:
    """Return a representative for the class of a; valid until next union().

    >>> union_find = UnionFind[str]()
    >>> union_find.union('a', 'b')
    >>> check_eq(union_find.find('a'), 'a')
    >>> check_eq(union_find.find('b'), 'a')
    >>> check_eq(union_find.find('c'), 'c')
    >>> union_find.union('d', 'a')
    >>> check_eq(union_find.find('b'), 'd')
    """
    if a not in self._rep:
      return a
    parents = []
    while True:
      parent = self._rep.setdefault(a, a)
      if parent == a:
        break
      parents.append(a)
      a = parent
    for p in parents:
      self._rep[p] = a
    return a


def topological_sort(graph: Mapping[_T, Sequence[_T]],
                     cycle_check: bool = False) -> list[_T]:
  """Given a dag (directed acyclic graph), return a list of graph nodes such
  that for every directed edge (u, v) in the graph, u is before v in the list.
  See https://en.wikipedia.org/wiki/Topological_sorting and
  https://stackoverflow.com/a/47234034 .

  >>> graph = {2: [3], 3: [4], 1: [2], 4: []}
  >>> topological_sort(graph)
  [1, 2, 3, 4]

  >>> topological_sort({2: [3], 3: [4, 5], 1: [2], 4: [5], 5: []})
  [1, 2, 3, 4, 5]
  """
  if sys.version_info > (3, 9):
    import graphlib  # pylint: disable=import-error
    return list(graphlib.TopologicalSorter(graph).static_order())[::-1]

  result = []
  seen = set()

  def recurse(node: _T) -> None:
    for dependent in reversed(graph[node]):
      if dependent not in seen:
        seen.add(dependent)
        recurse(dependent)
    result.append(node)

  all_dependents: set[_T] = set()
  all_dependents.update(*graph.values())
  for node in reversed(list(graph)):  # (reversed(graph) in Python 3.8).
    if node not in all_dependents:
      recurse(node)

  if cycle_check:
    position = {node: i for i, node in enumerate(result)}
    for node, dependents in graph.items():
      for dependent in dependents:
        if position[node] < position[dependent]:
          raise ValueError('Graph contains a cycle')

  return result[::-1]


# ** Search algorithms


def discrete_binary_search(feval: Callable[[int], float], xl: int, xh: int,
                           y_desired: float) -> int:
  """Return x such that feval(x) <= y_desired < feval(x + 1),

  Parameters must satisfy xl < xh and feval(xl) <= y_desired < feval(xh).

  >>> discrete_binary_search(lambda x: x**2, 0, 20, 15)
  3
  >>> discrete_binary_search(lambda x: x**2, 0, 20, 16)
  4
  >>> discrete_binary_search(lambda x: x**2, 0, 20, 17)
  4
  >>> discrete_binary_search(lambda x: x**2, 0, 20, 24)
  4
  >>> discrete_binary_search(lambda x: x**2, 0, 20, 25)
  5
  """
  assert xl < xh
  while xh - xl > 1:
    xm = (xl + xh) // 2
    ym = feval(xm)
    if y_desired >= ym:
      xl = xm
    else:
      xh = xm
  return xl


# ** General I/O


def is_executable(path: _Path) -> bool:
  """Check if a file is executable.

  >>> with tempfile.TemporaryDirectory() as dir:
  ...   path = pathlib.Path(dir) / 'file'
  ...   _ = path.write_text('test')
  ...   check_eq(is_executable(path), False)
  ...   if sys.platform != 'cygwin':
  ...     # Copy R bits to X bits:
  ...     path.chmod(path.stat().st_mode | ((path.stat().st_mode & 0o444) >> 2))
  ...     check_eq(is_executable(path), True)
  """
  return bool(pathlib.Path(path).stat().st_mode & stat.S_IEXEC)


# ** OS commands


def run(args: str | Sequence[str]) -> None:
  """Execute command, printing output from stdout and stderr.

  Args:
    args: Command to execute, which can be either a string or a sequence of word
      strings, as in `subprocess.run()`.  If `args` is a string, the shell is
      invoked to interpret it.

  Raises:
    RuntimeError: If the command's exit code is nonzero.

  >>> with tempfile.TemporaryDirectory() as dir:
  ...   path = pathlib.Path(dir) / 'file'
  ...   run(f'echo ab >{path}')
  ...   assert path.is_file() and 3 <= path.stat().st_size <= 4
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
        f"Command '{proc.args}' failed with code {proc.returncode}.")


if __name__ == '__main__':
  doctest.testmod()
