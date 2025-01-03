#!/usr/bin/env python3
# -*- fill-column: 100; -*-
"""Library of Python tools by Hugues Hoppe.

Useful commands to lint and test this module:
```shell
cd ..; c:/windows/sysnative/wsl -e bash -lc 'echo autopep8; autopep8 -j8 -d .; echo pyink; pyink --diff .; echo mypy; mypy .; echo pylint; pylint -j8 .; echo pytest; pytest -qq; echo All ran.'

env python3 -m doctest -v __init__.py | perl -ne 'print if /had no tests/../passed all/' | tail -n +2 | head -n -1
```
"""

from __future__ import annotations

__docformat__ = 'google'
__version__ = '1.5.7'
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
import importlib
import inspect
import io
import itertools
import math
import os
import pathlib
import pstats
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import types
import typing
from typing import Any, Generic, Literal, TypeVar, Union
import unittest.mock  # pylint: disable=unused-import # noqa
import uuid
import warnings

import numpy as np
import numpy.typing

if typing.TYPE_CHECKING:
  import PIL.ImageDraw

# For np.broadcast_to(), etc.
# mypy: allow-untyped-calls

_T = TypeVar('_T')
_F = TypeVar('_F', bound='Callable[..., Any]')

_UNDEFINED = object()

_NDArray = numpy.typing.NDArray[Any]
_DTypeLike = numpy.typing.DTypeLike
_ArrayLike = numpy.typing.ArrayLike
_Path = Union[str, os.PathLike[str]]

# ** numba


@typing.overload  # Bare decorator.
def noop_decorator(func: _F, /) -> _F:
  ...


@typing.overload  # Decorator with arguments.
def noop_decorator(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
  ...


def noop_decorator(*args: Any, **kwargs: Any) -> Any:
  """Return function decorated with no-operation; invocable with or without args.

  >>> @noop_decorator
  ... def func1(x): return x * 10

  >>> @noop_decorator()
  ... def func2(x): return x * 10

  >>> @noop_decorator(2, 3)
  ... def func3(x): return x * 10

  >>> @noop_decorator(keyword=True)
  ... def func4(x): return x * 10

  >>> check_eq(func1(1) + func2(2) + func3(4) + func4(8), 15 * 10)
  """
  if len(args) != 1 or not callable(args[0]) or kwargs:
    return noop_decorator  # Decorator is invoked with arguments; ignore them.
  func: Callable[..., Any] = args[0]
  return func


try:
  import numba
except ModuleNotFoundError:
  numba = sys.modules['numba'] = types.ModuleType('numba')
  numba.njit = noop_decorator


# ** Language extensions


def assert_not_none(value: _T | None, /) -> _T:
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


def check_eq(a: Any, b: Any, /) -> None:
  """Assert that two values are equal or raise exception with a useful message.

  Args:
    a: First expression.
    b: Second expression.

  Raises:
    RuntimeError: If `a != b` (or `np.any(a != b) if np.ndarray`).

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
  r"""Prints arguments to `stderr` immediately.

  >>> with unittest.mock.patch('sys.stderr', new_callable=io.StringIO) as m:
  ...   print_err('hello')
  ...   print(repr(m.getvalue()))
  'hello\n'
  """
  kwargs2: Any = dict(file=sys.stderr, flush=True) | kwargs
  print(*args, **kwargs2)


def _dump_vars(*args: Any) -> str:
  """Return a string showing the values of each expression.

  Specifically, convert each expression (contributed by the caller to the variable-parameter
  list `*args`) into a substring `f'expression = {expression}'` and join these substrings
  separated by `', '`.

  If the caller itself provided a variable-parameter list (*args), the search continues in its
  callers.  The approach examines a stack trace, so it is fragile and non-portable.

  Args:
    *args: Expressions to show.

  Raises:
    RuntimeError: If the invoking `_dump_vars(...)` is not contained on a single source line.

  >>> a = 45
  >>> b = 'Hello'

  >>> _dump_vars(a)
  'a = 45'

  >>> _dump_vars(b)
  'b = Hello'

  >>> _dump_vars(a, b, (a * 2) + 5, b + ' there')
  "a = 45, b = Hello, (a * 2) + 5 = 95, b + ' there' = Hello there"

  >>> _dump_vars([3, 4, 5][1])
  '[3, 4, 5][1] = 4'
  """

  def matching_parenthesis(text: str) -> int:
    """Return the index of ')' matching '(' in text[0]."""
    check_eq(text[0], '(')
    num_open = 0
    for i, ch in enumerate(text):
      if ch == '(':
        num_open += 1
      elif ch == ')':
        num_open -= 1
        if num_open == 0:
          return i
    raise RuntimeError(f'No matching right parenthesis in "{text}".')

  # Adapted from make_dict() in https://stackoverflow.com/a/2553524 .
  stack = traceback.extract_stack()
  # for frame in stack:
  #   print(f'{tuple(frame)=}', file=sys.stderr)
  this_function_name = stack[-1][2]  # i.e. initially '_dump_vars'.
  for stackframe in stack[-2::-1]:
    filename, unused_line_number, function_name, text = stackframe  # Caller.
    # https://docs.python.org/3/tutorial/errors.html:
    # "however, it will not display lines read from standard input."
    if filename == '<stdin>':
      check_eq(text, '')
      return ', '.join(str(e) for e in args)  # Unfortunate fallback.
    prefix = this_function_name + '('
    begin = text.find(prefix)
    if begin < 0:
      raise RuntimeError(f'_dump_vars: cannot find "{prefix}" in line "{text}".')
    begin += len(this_function_name)
    end = begin + matching_parenthesis(text[begin:])
    parameter_string = text[begin + 1 : end].strip()
    if re.fullmatch(r'\*[\w]+', parameter_string):
      this_function_name = function_name
      # Because the call is made using a *args, we continue to
      # the earlier caller in the stack trace.
    else:
      if len(args) == 1:
        expressions = [parameter_string.strip()]
      else:
        node = ast.parse(parameter_string)
        # print(ast.dump(node))  # ", indent=2" requires Python 3.9.
        value = getattr(node.body[0], 'value', '?')
        elements = getattr(value, 'elts', [value])

        def get_text(element: Any) -> str:
          text = ast.get_source_segment(parameter_string, element)
          return '?' if text is None else text

        expressions = [get_text(element) for element in elements]
      l = []
      for expr, value in zip(expressions, args):  # Python 3.10: strict=True.
        l.append(f'{expr} = {value}' if expr[0] not in '"\'' else str(value))
      return ', '.join(l)

  raise AssertionError


def show(*args: Any) -> None:
  r"""Prints expressions and their values on stdout.

  Args:
    *args: Expressions to show.

  Raises:
    RuntimeError: If the invoking `show(...)` is not contained on a single source line.

  Gets confused if there are multiple `show` calls on the same line of code.

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
  print(_dump_vars(*args), flush=True)


def analyze_functools_caches(variables: Mapping[str, Any], /) -> None:
  """Report on usage and efficiency of memoization caches.

  Args:
    variables: Dictionary, which is usually `globals()`.

  >>> @functools.cache
  ... def func(i: int) -> int:
  ...   return i**2

  >>> [func(i) for i in [1, 2, 1, 3, 1]]
  [1, 4, 1, 9, 1]
  >>> analyze_functools_caches(globals())  # doctest:+ELLIPSIS
  #... func ...  3/inf        0.400 hit=            2 miss=            3...
  """
  for name, value in variables.items():
    try:
      info = value.cache_info()
    except AttributeError:
      continue

    hit_ratio = info.hits / (info.hits + info.misses + 1e-30)
    s_max_size = 'inf' if info.maxsize is None else f'{info.maxsize:_}'
    name2 = f'{name:31}' if len(name) <= 31 else f'{name[:15]}..{name[-14:]}'
    print(
        f'# {name2} {info.currsize:11_}/{s_max_size:<10}'
        f' {hit_ratio:5.3f} hit={info.hits:13_} miss={info.misses:13_}'
    )


def clear_functools_caches(variables: Mapping[str, Any], /, *, verbose: bool = False) -> None:
  """Clear all the function memoization caches.

  Args:
    variables: Dictionary, which is usually `globals()`.
    verbose: If True, show names of cleared caches.

  >>> @functools.cache
  ... def func(i: int) -> int:
  ...   return i**2

  >>> [func(i) for i in [1, 2, 1, 3, 1]]
  [1, 4, 1, 9, 1]
  >>> check_eq(func.cache_info().currsize, 3)

  >>> clear_functools_caches(globals())
  >>> check_eq(func.cache_info().hits, 0)
  """
  for name, value in variables.items():
    with contextlib.suppress(AttributeError):
      value.cache_clear()
      if verbose:
        print(f'Cleared functools.cache for {name}().')


# ** Iterator functionality


def mirror_loop(sequence: Sequence[_T], duplicate_ends: bool = False) -> Iterator[_T]:
  """Yields elements from 'sequence' alternating forward and backward.

  Examples:
    >>> tuple(itertools.islice(mirror_loop((1, 2, 3, 4)), 10))
    (1, 2, 3, 4, 3, 2, 1, 2, 3, 4)

    >>> tuple(itertools.islice(mirror_loop((1, 2, 3, 4), duplicate_ends=True), 10))
    (1, 2, 3, 4, 4, 3, 2, 1, 1, 2)

    >>> tuple(itertools.islice(mirror_loop((1, 2)), 5))
    (1, 2, 1, 2, 1)

    >>> tuple(itertools.islice(mirror_loop((1, 2), duplicate_ends=True), 5))
    (1, 2, 2, 1, 1)

    >>> tuple(itertools.islice(mirror_loop((1,)), 5))
    (1, 1, 1, 1, 1)

    >>> tuple(itertools.islice(mirror_loop((1,), duplicate_ends=True), 5))
    (1, 1, 1, 1, 1)

  Args:
    sequence: Input elements; must be non-empty.
    duplicate_ends: boolean indicating if the first and last elements are duplicated in the output.
      (True leads to a more uniform usage of the elements but successive identical elements may
      affect inter-element statistics.)
  """
  length = len(sequence)
  if length == 0:
    raise ValueError('Accessing empty sequence.')
  while True:
    yield from sequence
    yield from sequence[::-1] if duplicate_ends else sequence[1:-1][::-1]


def divide_slice(sl: slice, n: int) -> Iterator[slice]:
  """Divide a slice `sl` into `n` subslices.

  Args:
    slice: The slice to divide.
    n: The number of subslices.

  Yields:
    A subslice of the slice.  If the size of `sl` is less than `n`, the last subslices are empty.

  >>> list(divide_slice(slice(0, 10), 2))
  [slice(0, 5, None), slice(5, 10, None)]

  >>> list(divide_slice(slice(0, 10), 3))
  [slice(0, 4, None), slice(4, 8, None), slice(8, 10, None)]

  >>> list(divide_slice(slice(1, 4), 2))
  [slice(1, 3, None), slice(3, 4, None)]

  >>> list(divide_slice(slice(1, 4), 3))
  [slice(1, 2, None), slice(2, 3, None), slice(3, 4, None)]

  >>> list(divide_slice(slice(1, 4), 4))
  [slice(1, 2, None), slice(2, 3, None), slice(3, 4, None), slice(4, 4, None)]

  >>> list(divide_slice(slice(1, 1), 1))
  [slice(1, 1, None)]

  >>> list(divide_slice(slice(1, 1), 2))
  [slice(1, 1, None), slice(1, 1, None)]
  """
  subslice_size = math.ceil((sl.stop - sl.start) / n)
  for i in range(n):
    yield slice(
        min(sl.start + i * subslice_size, sl.stop),
        min(sl.start + (i + 1) * subslice_size, sl.stop),
        sl.step,
    )


# ** Jupyter/IPython notebook functionality


def _get_ipython() -> Any:
  import IPython

  return IPython.get_ipython()  # type: ignore


def in_notebook() -> bool:
  """Return True if running inside an Jupyter/IPython notebook.

  >>> in_notebook()
  False
  """
  return _get_ipython() is not None


def in_colab() -> bool:
  """Return True if running inside Google Colab.

  >>> in_colab()
  False
  """
  try:
    import google.colab  # pylint: disable=unused-import # noqa # pytype: disable=import-error

    return True
  except ModuleNotFoundError:
    return False


def no_vertical_scroll() -> None:
  """If in Colab, omit vertical scrollbar in cell output."""
  if in_colab():
    import google.colab.output  # pylint: disable=import-error,no-name-in-module # pytype:disable=import-error

    google.colab.output.no_vertical_scroll()  # pylint: disable=c-extension-no-member


def display(obj: Any, /) -> None:
  """In a Jupyter notebook, display the object."""
  import IPython.display

  IPython.display.display(obj)


def display_html(text: str, /) -> None:
  """In a Jupyter notebook, display the HTML `text`."""
  import IPython.display

  display(IPython.display.HTML(text))  # type: ignore[attr-defined]


def display_math(text: str, /) -> None:
  """In a Jupyter notebook, display the LaTeX `text`."""
  import IPython.display

  display(IPython.display.Math(text))  # type: ignore[attr-defined]


def adjust_jupyterlab_markdown_width(width: int = 1016, /) -> None:
  """Set the Markdown cell width in Jupyterlab to the value used by Colab."""
  # https://stackoverflow.com/a/66278615.
  style = f'{{max-width: {width}px!important;}}'
  classes = [
      '.jp-Cell.jp-MarkdownCell',
      '.jp-RenderedMarkdown',  # For show_var_docstring().
      '.jp-RenderedLatex',  # For Latex output.
  ]
  s_classes = ', '.join(classes)
  text = f'<style>{s_classes} {style}</style>'
  display_html(text)


class _CellTimer:
  """Record timings of all notebook cells and show top entries at the end."""

  # Inspired from https://github.com/cpcloud/ipython-autotime.

  instance: _CellTimer | None = None

  def __init__(self) -> None:
    self.elapsed_times: dict[int, float] = {}
    self.pre_run(None)
    _get_ipython().events.register('pre_run_cell', self.pre_run)
    _get_ipython().events.register('post_run_cell', self.post_run)

  def close(self) -> None:
    """Destroy the `_CellTimer` and its notebook callbacks."""
    _get_ipython().events.unregister('pre_run_cell', self.pre_run)
    _get_ipython().events.unregister('post_run_cell', self.post_run)

  def pre_run(self, unused_info: Any) -> None:
    """Start a timer for the notebook cell execution."""
    self.start_time = time.perf_counter()

  def post_run(self, unused_result: Any) -> None:
    """Start the timer for the notebook cell execution."""
    elapsed_time = time.perf_counter() - self.start_time
    input_index = _get_ipython().execution_count - 1
    self.elapsed_times[input_index] = elapsed_time

  def show_times(self, n: int | None = None, sort: bool = False) -> None:
    """Print notebook cell timings."""
    print(f'# Total time: {sum(self.elapsed_times.values()):.2f} s')
    times = list(self.elapsed_times.items())
    times = sorted(times, key=lambda x: x[sort], reverse=sort)
    # https://github.com/ipython/ipython/blob/master/IPython/core/history.py
    # https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.history.html
    session = _get_ipython().history_manager.session_number
    history_range = _get_ipython().history_manager.get_range(session)
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
  if in_notebook():
    if _CellTimer.instance:
      _CellTimer.instance.close()
    _CellTimer.instance = _CellTimer()


def show_notebook_cell_top_times() -> None:
  """Print summary of timings for Jupyter notebook cells.

  Place in a late notebook cell.  See also `start_timing_notebook_cells`.
  """
  if _CellTimer.instance:
    _CellTimer.instance.show_times(n=20, sort=True)


class StopExecution(Exception):
  """Exception that will not dump trace; useful to quietly abort a notebook cell computation."""

  # Adapted from https://stackoverflow.com/a/56953105.

  def __init__(self, message: str = '<StopExecution>') -> None:
    self.message = message

  def _render_traceback_(self) -> None:
    if self.message:
      print(self.message)


def pdoc_help(
    thing: Any,
    /,
    *,
    docformat: Literal['markdown', 'google', 'numpy', 'restructuredtext'] = 'google',
) -> None:
  """Display `pdoc` (HTML) documentation for a function or class.

  >>> import IPython.display
  >>> htmls = []
  >>> with unittest.mock.patch('IPython.display.display', htmls.append):
  ...   pdoc_help(dataclasses.dataclass)
  >>> (html,) = htmls
  >>> assert 'View Source' in html.data, html.data
  """
  # Adapted from https://github.com/mitmproxy/pdoc/issues/494.
  import pdoc

  with tempfile.TemporaryDirectory() as temp_dir:
    template_dir = pathlib.Path(temp_dir)
    (template_dir / 'frame.html.jinja2').write_text(
        """\
      <div>
          {% block content %}{% endblock %}
          {% filter minify_css %}
                  <style>{% include "syntax-highlighting.css" %}</style>
                  <style>{% include "theme.css" %}</style>
                  <style>{% include "content.css" %}</style>
          {% endfilter %}
      </div>
      """,
        encoding='utf-8',
    )
    (template_dir / 'module.html.jinja2').write_text(
        """\
      {% extends "default/module.html.jinja2" %}
      {% macro is_public(doc) %}
          {% if should_show(doc.qualname) %}
              {{ default_is_public(doc) }}
          {% endif %}
      {% endmacro %}
      {% block module_info %}{% endblock %}
      """,
        encoding='utf-8',
    )
    module = inspect.getmodule(thing)
    if module is None:
      raise ValueError(f'Cannot identify module for {thing=}.')
    doc = pdoc.doc.Module(module)
    pdoc.render.configure(
        docformat=docformat, math=True, show_source=True, template_directory=template_dir
    )

    def should_show(qualname: str) -> bool:
      thing_name = getattr(thing, '__qualname__', '')
      return all(x == y for x, y in zip(qualname.split('.'), thing_name.split('.')))

    pdoc.render.env.globals['should_show'] = should_show
    with warnings.catch_warnings():
      # Ignore the fact that pdoc cannot parse annotations of the form "x | y" with Python <3.10.
      warnings.filterwarnings(action='ignore', message='Error parsing type annotation')
      html = pdoc.render.html_module(module=doc, all_modules={})

  # Limit the maximum width.
  html = '<style>main.pdoc {max-width: 784px;}</style>\n' + html

  # The <h6> tags would appear in the Jupyterlab table of contents, so change to <div>.
  html = '<style>.myh6 {font-size: 14px; font-weight: 700;}</style>\n' + html
  html = html.replace('<h6', '<div class="myh6"').replace('</h6>', '</div>')

  display_html(html)


# ** Timing


def get_time_and_result(
    func: Callable[[], _T], /, *, max_repeat: int = 10, max_time: float = 2.0
) -> tuple[float, _T]:
  """Call function `func` repeatedly to determine its minimum run time.

  If the measured run time is small, more precise time estimates are obtained
  by considering batches of function calls (with automatically increasing
  batch size).

  Args:
    func: Function to time.
    max_repeat: Maximum number of batch measurements across which to compute the minimum value.
    max_time: Desired maximum total time in obtaining timing measurements.
      If set to zero, a single timing measurement is taken.

  Returns:
    time: The minimum time (in seconds) measured across the repeated calls to `func` (divided
      by the batch size if batches are introduced).
    result: The value returned by the last call to `func`.

  >>> elapsed, result = get_time_and_result(lambda: 11 + 22)
  >>> assert elapsed < 0.01, elapsed
  >>> result
  33
  """
  assert callable(func) and max_repeat > 0 and max_time >= 0.0
  result = None
  batch_size = 1
  smallest_acceptable_batch_time = 0.01  # All times are in seconds.
  gc_was_enabled = gc.isenabled()

  try:
    gc.disable()
    # gc.collect()
    while True:
      num_repeat = 0
      sum_time = 0.0
      min_time = math.inf
      start = time.perf_counter_ns()
      while num_repeat < max_repeat and sum_time <= max_time:
        for _ in range(batch_size):
          result = func()
        stop = time.perf_counter_ns()
        elapsed = (stop - start) / 10**9
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


def get_time(func: Callable[[], Any], /, **kwargs: Any) -> float:
  """Return the minimum execution time when repeatedly calling `func`.

  >>> elapsed = get_time(lambda: time.sleep(0.2), max_repeat=1)
  >>> assert 0.15 < elapsed < 0.25, elapsed
  """
  return get_time_and_result(func, **kwargs)[0]


def print_time(func: Callable[[], Any], /, **kwargs: Any) -> None:
  r"""Print the minimum execution time when repeatedly calling `func`.

  >>> with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as m:
  ...   print_time(lambda: 1)
  ...   assert re.fullmatch(r'[\d.]+ .?s\n', m.getvalue()), m.getvalue()

  """
  min_time = get_time(func, **kwargs)
  # print(f'{min_time:.3f} s', flush=True)
  text = (
      f'{format_float(min_time, 3)} s'
      if min_time >= 1.0
      else f'{format_float(min_time*1e3, 3)} ms'
      if min_time > 1.0 / 1e3
      else f'{format_float(min_time*1e6, 3)} µs'
      if min_time > 1.0 / 1e6
      else f'{format_float(min_time*1e6, 2)} µs'
  )
  print(text, flush=True)


# ** Profiling


def prun(
    func: Callable[[], Any],
    /,
    *,
    mode: Literal['original', 'full', 'tottime'] = 'tottime',
    top: int | None = None,
) -> None:
  """Profile calling the function `func` and print reformatted statistics.

  >>> with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as m:
  ...   prun(lambda: np.linalg.qr(np.ones((400, 400))))
  ...   lines = m.getvalue().splitlines()
  >>> assert lines[0].startswith('# Prun: tottime ')
  >>> assert 'overall_cumtime' in lines[0]
  """
  assert callable(func)
  assert mode in ('original', 'full', 'tottime'), mode
  site_packages = list(np.__path__)[0][:-5].replace('\\', '/')
  assert site_packages.endswith(('/site-packages/', '/dist-packages/'))

  profile = cProfile.Profile()
  try:
    profile.enable()
    func()
  finally:
    profile.disable()

  with io.StringIO() as string_io:
    args = (top,) if top is not None else ()
    pstats.Stats(profile, stream=string_io).sort_stats('tottime').print_stats(*args)
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
      pattern = r'\s*\S+\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+(\S.*)'
      tottime_str, cumtime_str, name = re_groups(pattern, line)
      tottime, cumtime = float(tottime_str), float(cumtime_str)
      beautified_name = beautify_function_name(name)
      overall_time += 1e-6
      significant_time = tottime / overall_time > 0.05 or 0.05 < cumtime / overall_time < 0.95
      if top is not None or significant_time:
        if mode == 'tottime':
          output.append(f'     {tottime:8.3f} {cumtime:8.3f} {beautified_name}')
        else:  # mode == 'full'
          output.append(line.replace(name, beautified_name))
    elif ' function calls ' in line:
      overall_time = float(re_groups(r' in (\S+) seconds', line)[0])
      output.append(f'Prun: tottime {overall_time:8.3f} overall_cumtime')
    elif line.lstrip().startswith('ncalls '):
      if mode == 'full':
        output.append(line)
      post_header = True

  print('\n'.join([f'#{" " * bool(line)}' + line for line in output]))


# ** Class objects


class OrderedEnum(enum.Enum):
  """An Enum supporting comparisons, sort, and max.

  >>> class MyEnum(OrderedEnum):
  ...   VALUE1 = enum.auto()
  >>> assert hash(MyEnum.VALUE1)
  >>> assert MyEnum.VALUE1 <= MyEnum.VALUE1
  """

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


# ** Temporary variable assignment


@contextlib.contextmanager
def temporary_assignment(variables: dict[str, Any], /, **kwargs: Any) -> Iterator[None]:
  """Temporarily assign `**kwargs` to `variables`.

  Args:
    variables: Usually the `globals()` of the caller module.  Note that a function-scope
      `locals()` does not work as it should not be modified.
    **kwargs: Assignments of values to variable names.

  >>> var = 1
  >>> with temporary_assignment(globals(), var=2):
  ...   check_eq(var, 2)
  >>> check_eq(var, 1)

  >>> assert 'var2' not in globals()
  >>> with temporary_assignment(globals(), var2='1'):
  ...   check_eq(var2, '1')  # noqa
  >>> assert 'var2' not in globals()
  """
  # https://stackoverflow.com/a/57226721.
  old_values = {key: variables.get(key, _UNDEFINED) for key in kwargs}
  try:
    variables.update(kwargs)
    yield
  finally:
    for key, old_value in old_values.items():
      if old_value is _UNDEFINED:
        variables.pop(key)
      else:
        variables[key] = old_value


# ** Meta programming


def terse_str(cls: type, /) -> type:
  """Decorator for a `dataclasses.dataclass`, which defines a custom `str()`.

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
      field.name: (field.default_factory() if callable(field.default_factory) else field.default)
      for field in dataclasses.fields(cls)
      if field.repr
  }

  def new_str(self: Any) -> str:
    """Return a string containing only the non-default field values."""
    text = ', '.join(
        f'{name}={getattr(self, name)!r}'
        for name, default in default_for_field.items()
        if getattr(self, name) != default
    )
    return f'{type(self).__name__}({text})'

  setattr(cls, '__str__', new_str)
  return cls


# ** Memoization


def selective_lru_cache(
    *args: Any, ignore_kwargs: tuple[str, ...] = (), **kwargs: Any
) -> Callable[[_F], _F]:
  """Like `functools.lru_cache` but memoization can ignore specified `kwargs`.

  Because `lru_cache` is unaware of default keyword values, it is recommended that the parameters
  named in `ignore_kwargs` not have defaults in the decorated function.
  Inspired by https://stackoverflow.com/a/30738279

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
      kwargs = {k: (v.obj if k in ignore_kwargs else v) for k, v in kwargs.items()}
      return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      kwargs = {k: (Equals(v) if k in ignore_kwargs else v) for k, v in kwargs.items()}
      return helper(*args, **kwargs)

    for attribute in ['cache_clear', 'cache_info', 'cache_parameters']:
      value = getattr(helper, attribute, None)
      if value:  # 'cache_parameters' added only in Python 3.9.
        setattr(wrapper, attribute, value)
    return typing.cast(_F, wrapper)

  return decorator


# ** Imports and modules


def create_module(module_name: str, elements: Iterable[Any] = (), /) -> Any:
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
  module = sys.modules.get(module_name)
  if not module:
    module = sys.modules[module_name] = types.ModuleType(module_name)
  for element in elements:
    setattr(module, element.__name__, element)
    element.__module__ = module_name

  return module


@contextlib.contextmanager
def function_in_temporary_module(
    function: _F,
    *,
    header: str = '',
    funcs: Sequence[Any] = (),
) -> Iterator[_F]:
  """Copies function into a new module backed by a Python file, for multiprocessing.

  Args:
    function: An original function (Callable).
    header: Text to be inserted at the top of the temporary module.  It typically contains imports
      necessary for the definitions of `function` or `funcs`.
    funcs: List of additional functions to include in the temporary module.

  Yields:
    function: The new callable in the temporary module.
  """
  sources = [header] + [inspect.getsource(func) for func in [function] + list(funcs)]
  source = '\n\n\n'.join(textwrap.dedent(text) for text in sources)

  old_sys_path = sys.path
  try:
    salt = uuid.uuid4().hex[-8:]
    module_name = f'temp_module_{salt}'
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      temp_file = temp_path / f'{module_name}.py'
      temp_file.write_text(source, encoding='utf-8')
      sys.path.append(str(temp_path))
      temp_module = importlib.import_module(module_name)
      new_function = getattr(temp_module, function.__name__)
      assert new_function is not None
      yield new_function

  finally:
    sys.path = old_sys_path


# ** System functions


@contextlib.contextmanager
def timing(
    description: str = 'Timing',
    /,
    *,
    enabled: bool = True,
) -> Iterator[None]:
  """Context that reports elapsed time and multithreaded parallelism.

  Args:
    description: A string to print before the elapsed time.

  Yields:
    None.

  >>> with timing('List comprehension example'):
  ...   _ = [i for i in range(10_000)]  # doctest:+ELLIPSIS
  List comprehension example: 0.00...
  """
  if enabled:
    gc_was_enabled = gc.isenabled()

    try:
      gc.disable()
      # gc.collect()
      start = time.perf_counter_ns()
      process_time_start = time.process_time_ns()

      try:
        yield

      finally:
        elapsed_time = (time.perf_counter_ns() - start) / 10**9
        process_time = (time.process_time_ns() - process_time_start) / 10**9
        multithreading = process_time / elapsed_time
        s_parallelism = f'  {multithreading:5.2f}x' if multithreading > 1.05 else ''
        print(f'{description}: {elapsed_time:.6f}{s_parallelism}')

    finally:
      if gc_was_enabled:
        gc.enable()
  else:
    yield


def typename(thing: Any, /) -> str:
  """Return the full name (including module) of the type of `thing`.

  >>> typename(5)
  'int'

  >>> typename('text')
  'str'

  >>> typename(np.array([1]))
  'numpy.ndarray'
  """
  # https://stackoverflow.com/a/2020083
  name: str = thing.__class__.__qualname__
  module = thing.__class__.__module__
  return name if module in (None, 'builtins') else f'{module}.{name}'


def show_biggest_vars(variables: Mapping[str, Any], /, n: int = 10) -> None:
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
  infos = [(name, sys.getsizeof(value), typename(value)) for name, value in var.items()]
  infos.sort(key=lambda t: t[1], reverse=True)
  for name, size, vartype in infos[:n]:
    print(f'{name:24} {vartype:20} {size:_}')


# ** String functions


def format_float(value: float, /, precision: int) -> str:
  """Return the non-scientific representation of `value` with specified `precision` digits.

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
  text = np.format_float_positional(value, fractional=False, unique=False, precision=precision)
  return text.rstrip('.')


def re_groups(pattern: str, string: str, /) -> tuple[str, ...]:
  r"""Like `re.search(...).groups()` but with an assertion that a match is found.

  Args:
    pattern: A regular expression.  It may include a prefix `'^'` or suffix `'$'` to constrain
      the search location.
    string: Text to search.

  Returns:
    A tuple of strings corresponding to the regex groups found in the pattern match within `string`.

  Raises:
    ValueError if `pattern` is not found in `string`.

  >>> re_groups(r'object (\d+).*loc (\w+)', 'The object 13 at loc ABC.')
  ('13', 'ABC')
  >>> re_groups(r'(\d+)', 'Some text.')
  Traceback (most recent call last):
  ...
  ValueError: Did not locate pattern "(\d+)" in "Some text.".
  """
  match = re.search(pattern, string)
  if not match:
    raise ValueError(f'Did not locate pattern "{pattern}" in "{string}".')
  return match.groups()


# ** Discrete mathematics.


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
  """Find the greatest common divisor using the extended Euclidean algorithm.

  Returns: (gcd(a, b), x, y) with the property that a * x + b * y = gcd(a, b).

  >>> extended_gcd(29, 71)
  (1, -22, 9)
  >>> -22 * 29 + 9 * 71, math.gcd(29, 71)
  (1, 1)

  >>> extended_gcd(6, 8)
  (2, -1, 1)
  >>> -1 * 6 + 1 * 8, math.gcd(6, 8)
  (2, 2)
  """
  prev_x, x = 1, 0
  prev_y, y = 0, 1
  while b:
    q, r = divmod(a, b)
    x, prev_x = prev_x - q * x, x
    y, prev_y = prev_y - q * y, y
    a, b = b, r
  return a, prev_x, prev_y


def solve_modulo_congruences(remainders: Sequence[int], moduli: Sequence[int]) -> int:
  """Return `x` satisfying `x % moduli[i] == remainders[i]`; handles non-coprime moduli.

  >>> solve_modulo_congruences([3, 6, 6], [5, 7, 11])  # Coprime moduli.
  83

  >>> solve_modulo_congruences([1, 1, 1, 1, 1], [2, 3, 4, 5, 6])  # Non-coprime moduli.
  1
  """
  if len(remainders) != len(moduli):
    raise ValueError('Number of remainders must match number of moduli.')

  r, m = remainders[0], moduli[0]

  for r2, m2 in zip(remainders[1:], moduli[1:]):  # Python 3.10: strict=True
    g, x, _ = extended_gcd(m, m2)
    assert r % g == r2 % g
    if (r2 - r) % g != 0:
      raise ValueError('No solution exists.')

    # Find a particular solution.
    x0 = (x * ((r2 - r) // g)) % (m2 // g)

    # Update the current solution.
    r = r + m * x0
    m = m * m2 // g

  return r % m


# ** Continuous mathematics.


def as_float(a: _ArrayLike, /) -> _NDArray:
  """If `a` is not floating-point, convert it to floating-point type.

  Args:
    a: Input array.

  Returns:
    Array `a` if it is already floating-point (`np.float32` or `np.float64`), else `a` converted to
    type `np.float32` or `np.float64` based on the necessary precision.  Note that 64-bit integers
    cannot be represented exactly.

  >>> as_float(np.array([1.0, 2.0]))
  array([1., 2.])

  >>> as_float(np.array([1.0, 2.0], np.float32))
  array([1., 2.], dtype=float32)

  >>> as_float(np.array([1.0, 2.0], 'float64'))
  array([1., 2.])

  >>> as_float(np.array([1, 2], np.uint8))
  array([1., 2.], dtype=float32)

  >>> as_float(np.array([1, 2], np.uint16))
  array([1., 2.], dtype=float32)

  >>> as_float(np.array([1, 2]))
  array([1., 2.])
  """
  a = np.asarray(a)
  if issubclass(a.dtype.type, np.floating):
    return a
  dtype = np.float64 if np.iinfo(a.dtype).bits >= 32 else np.float32
  return a.astype(dtype)


def normalize(a: _ArrayLike, /, axis: int | None = None) -> _NDArray:
  """Return array `a` scaled such that its elements have unit 2-norm.

  Args:
    a: Input array.
    axis: If None, normalizes the entire matrix.  Otherwise, normalizes each
      element along the specified axis.

  Returns:
    An array such that its elements along `axis` are rescaled to have L2 norm
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

  >>> normalize([[0, 0], [10, 10]], axis=-1)
  array([[       nan,        nan],
         [0.70710678, 0.70710678]])
  """
  a = np.asarray(a)
  norm = np.linalg.norm(a, axis=axis)
  if axis is not None:
    norm = np.expand_dims(norm, axis)
  with np.errstate(invalid='ignore'):
    return a / norm


def rms(a: _ArrayLike, /, axis: int | None = None) -> _NDArray:
  """Return the root mean square of the array values.

  >>> assert rms([3.0]) == 3.0

  >>> rms([-3.0, 4.0]).item()
  3.5355339059327378

  >>> rms([10, 11, 12]).item()
  11.030261405182864

  >>> rms([[-1.0, 1.0], [0.0, -2.0]]).item()
  1.224744871391589

  >>> rms([[-1.0, 1.0], [0.0, -2.0]], axis=-1)
  array([1.        , 1.41421356])
  """
  return np.sqrt(np.mean(np.square(as_float(a)), axis, np.float64))


def prime_factors(n: int, /) -> list[int]:
  """Return an ascending list of the (greater-than-one) prime factors of `n`.

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


def van_der_corput(n: int, base: int = 2) -> float:
  """Return the nth element of the quasi-random low-discrepancy Van der Corput sequence.

  Args:
    n: Index in the sequence.  Zero is at n=0, and 0.5 is at n=1, so starting at n=1 is useful.
    base: Base for the sequence (typically 2 for binary).

  Return:
    A value in the range [0, 1) with low-discrepancy properties.

  >>> [van_der_corput(i) for i in range(1, 6)]
  [0.5, 0.25, 0.75, 0.125, 0.625]
  """
  vdc = 0.0
  denom = 1
  while n:
    denom *= base
    n, remainder = divmod(n, base)
    vdc += remainder / denom
  return vdc


def van_der_corput_sequence(n: int, base: int = 2) -> _NDArray:
  """Generate a vectorized Van der Corput sequence using efficient bitwise operations.

  Args:
    n: Number of elements to generate in the sequence.
    base: Base for the sequence (typically 2 for binary).

  Returns:
    NumPy array of low-discrepancy sequence elements in the range [0, 1).

  >>> van_der_corput_sequence(5)
  array([0.5  , 0.25 , 0.75 , 0.125, 0.625])
  >>> assert np.all(van_der_corput_sequence(1000) == [van_der_corput(i + 1) for i in range(1000)])
  """
  indices = np.arange(1, n + 1)
  vdc = np.zeros(n, dtype=np.float64)
  current_denom = base
  # Process each bit position
  while np.any(indices):
    remainder = indices % base
    vdc += remainder / current_denom
    indices //= base
    current_denom *= base
  return vdc


def diagnostic(a: _ArrayLike, /) -> str:
  """Return a diagnostic string summarizing the values in `a` for debugging.

  Args:
    a: Input values; must be convertible to an `np.ndarray` of scalars.

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
  return (
      f'shape={a.shape} {dtype=!s} size={a.size}'
      f' nan={np.isnan(a).sum()}'
      f' posinf={np.isposinf(a).sum()}'
      f' neginf={np.isneginf(a).sum()}'
      f' finite{repr(Stats(finite))[10:]}'
      f' zero={(finite == 0).sum()}'
  )


# ** Statistics

# Note that using dataclasses.dataclass(frozen=True) incurs a performance
# penalty # because the initialization must use object.__setattr__() to bypass
# the # disabled __set_attr__() member function.
# Instead, I use an ordinary class with protected class fields.


def _determine_precision(dtype: _DTypeLike) -> np.dtype[Any]:
  """Return a dtype for accurate statistics computations."""
  if np.issubdtype(dtype, np.signedinteger):
    return np.dtype(np.int64)
  if np.issubdtype(dtype, np.unsignedinteger):
    return np.dtype(np.uint64)
  if np.issubdtype(dtype, np.floating):
    return np.dtype(np.float64)
  raise ValueError(f'Array dtype {dtype} is not supported.')


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
  >>> Stats([-2, 2]).rms().item()
  2.0

  >>> a = Stats([1, 2])
  >>> assert a.min() == 1 and a.max() == 2 and a.avg() == 1.5

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
      precision = _determine_precision(a.dtype)
      a = a.astype(precision)
      self._size = a.size
      self._sum = a.sum()
      self._sum2 = np.square(a).sum()
      self._min = a.min() if a.size > 0 else math.inf
      self._max = a.max() if a.size > 0 else -math.inf
    else:
      self._size, self._sum, self._sum2, self._min, self._max = args

  def sum(self) -> float:
    """Return the sum of the values.

    >>> f'{Stats([3.5, 2.2, 4.4]).sum():.8g}'
    '10.1'
    """
    return self._sum

  def min(self) -> float:
    """Return the minimum value.

    >>> assert Stats([3.5, 2.2, 4.4]).min() == 2.2
    """
    return self._min

  def max(self) -> float:
    """Return the maximum value.

    >>> assert Stats([3.5, 2.2, 4.4]).max() == 4.4
    """
    return self._max

  def avg(self) -> float:
    """Return the average.

    >>> assert Stats([1, 1, 4]).avg() == 2.0
    """
    return math.nan if self._size == 0 else self._sum / self._size

  def ssd(self) -> float:
    """Return the sum of squared deviations.

    >>> assert Stats([1, 1, 4]).ssd() == 6.0
    """
    return math.nan if self._size == 0 else max(self._sum2 - self._sum**2 / self._size, 0)

  def var(self) -> float:
    """Return the unbiased estimate of variance, as in `np.var(a, ddof=1)`.

    >>> assert Stats([1, 1, 4]).var() == 3.0
    """
    return (
        math.nan if self._size == 0 else 0.0 if self._size == 1 else self.ssd() / (self._size - 1)
    )

  def sdv(self) -> float:
    """Return the unbiased standard deviation as in `np.std(a, ddof=1)`.

    >>> Stats([1, 1, 4]).sdv().item()
    1.7320508075688772
    """
    return self.var() ** 0.5

  def rms(self) -> float:
    """Return the root-mean-square.

    >>> Stats([1, 1, 4]).rms().item()
    2.449489742783178

    >>> assert Stats([-1, 1]).rms() == 1.0
    """
    if self._size == 0:
      return 0.0
    return (self._sum2 / self._size) ** 0.5

  def __format__(self, format_spec: str = '') -> str:
    """Return a summary of the statistics `(size, min, max, avg, sdv)`."""
    fmt = format_spec if format_spec else '#12.6'
    fmt_int = fmt[: fmt.find('.')] if fmt.find('.') >= 0 else ''
    fmt_min = fmt if isinstance(self._min, np.floating) else fmt_int
    fmt_max = fmt if isinstance(self._max, np.floating) else fmt_int
    return (
        f'({self._size:11_})'
        f' {self._min:{fmt_min}} :'
        f' {self._max:<{fmt_max}}'
        f' av={self.avg():<{fmt}}'
        f' sd={self.sdv():<{fmt}}'
    ).rstrip()

  def __str__(self) -> str:
    return self.__format__()

  def __repr__(self) -> str:
    fmt = '.6'
    fmt_int = ''
    fmt_min = fmt if isinstance(self._min, np.floating) else fmt_int
    fmt_max = fmt if isinstance(self._max, np.floating) else fmt_int
    return (
        f'Stats(size={self._size}, '
        f'min={self._min:{fmt_min}}, '
        f'max={self._max:{fmt_max}}, '
        f'avg={self.avg():{fmt}}, '
        f'sdv={self.sdv():{fmt}})'
    )

  def __len__(self) -> int:
    return self._size

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Stats):
      return NotImplemented
    return (
        self._size == other._size
        and self._sum == other._sum
        and self._sum2 == other._sum2
        and self._min == other._min
        and self._max == other._max
    )

  def __add__(self, other: Stats) -> Stats:
    """Return combined statistics.

    >>> assert Stats([2, -1]) + Stats([7, 5]) == Stats([-1, 2, 5, 7])
    """
    return Stats(
        self._size + other._size,
        self._sum + other._sum,
        self._sum2 + other._sum2,
        min(self._min, other._min),
        max(self._max, other._max),
    )

  def __mul__(self, n: int) -> Stats:
    """Return statistics whereby each element appears `n` times.

    >>> assert Stats([4, -2]) * 3 == Stats([-2, -2, -2, 4, 4, 4])
    """
    return Stats(self._size * n, self._sum * n, self._sum2 * n, self._min, self._max)


# ** Numpy operations


def array_always(a: _ArrayLike | Iterable[_ArrayLike], /) -> _NDArray:
  """Return a numpy array even if `a` is an iterator of subarrays.

  >>> array_always(np.array([[1, 2], [3, 4]]))
  array([[1, 2],
         [3, 4]])

  >>> array_always(range(3) for _ in range(2))
  array([[0, 1, 2],
         [0, 1, 2]])

  >>> array_always([[1, 2], [3, 4]])
  array([[1, 2],
         [3, 4]])
  """
  if isinstance(a, collections.abc.Iterator):
    return np.array(tuple(a))
  return np.asarray(a)


def to_image(a: _ArrayLike, /, color0: _ArrayLike = 0, color1: _ArrayLike = 255) -> _NDArray:
  """Return a 3-channel uint8 image given a boolean array and specified colors.

  >>> to_image([False, True, True], 10, 240)
  array([[ 10,  10,  10],
         [240, 240, 240],
         [240, 240, 240]], dtype=uint8)

  >>> to_image(np.eye(2) == 1, (240, 241, 242), (255, 0, 0))
  array([[[255,   0,   0],
          [240, 241, 242]],
  <BLANKLINE>
         [[240, 241, 242],
          [255,   0,   0]]], dtype=uint8)
  """
  a = np.asarray(a)
  assert a.dtype == bool
  image = np.full((*a.shape, 3), color0, np.uint8)
  image[a] = color1
  return image


def pad_array(array: _ArrayLike, /, pad: _ArrayLike, value: _ArrayLike = 0) -> _NDArray:
  """Return `array` padded along initial dimensions by `value`, which may be an array.

  Args:
    array: Input data.
    pad: Sequence of tuples representing pad widths before and after each desired dimension.
      The length of `pad` may be less than `array.ndim`.  If `pad` is scalar, it is broadcast
      onto the shape `(array.ndim - value.ndim, 2)`.  If `pad` is 1-dim, each entry is duplicated.
    value: Value to use for padding.  It must be scalar if `pad` is scalar; otherwise its shape
      must equal `array.shape[len(pad):]`.

  >>> array1 = np.arange(6).reshape(2, 3)
  >>> pad_array(array1, 1, 9)
  array([[9, 9, 9, 9, 9],
         [9, 0, 1, 2, 9],
         [9, 3, 4, 5, 9],
         [9, 9, 9, 9, 9]])

  >>> pad_array(array1, ((1, 0), (0, 1)), 9)
  array([[9, 9, 9, 9],
         [0, 1, 2, 9],
         [3, 4, 5, 9]])

  >>> pad_array(array1, ((2, 0),), (6, 7, 8))
  array([[6, 7, 8],
         [6, 7, 8],
         [0, 1, 2],
         [3, 4, 5]])

  >>> pad_array(array1, ((0, 1),), (6, 7, 8))
  array([[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]])

  >>> pad_array(array1, (0, 1), 9)
  array([[9, 0, 1, 2, 9],
         [9, 3, 4, 5, 9]])

  >>> pad_array([[[1, 2], [3, 4]],
  ...            [[5, 6], [7, 8]]], ((0, 1), (1, 0)), (9, 0))
  array([[[9, 0],
          [1, 2],
          [3, 4]],
  <BLANKLINE>
         [[9, 0],
          [5, 6],
          [7, 8]],
  <BLANKLINE>
         [[9, 0],
          [9, 0],
          [9, 0]]])

  >>> pad_array([1, 2, 3], 0, 9)
  array([1, 2, 3])

  >>> pad_array([1, 2, 3], 1, 9)
  array([9, 1, 2, 3, 9])

  >>> pad_array([1, 2, 3], ((0, 1),), 9)
  array([1, 2, 3, 9])
  """
  array, pad, value = np.asarray(array), np.asarray(pad), np.asarray(value)
  if pad.ndim == 0:
    pad = np.broadcast_to(pad, (array.ndim - value.ndim, 2))
  elif pad.ndim == 1:
    pad = np.broadcast_to(pad[:, None], (array.ndim - value.ndim, 2))
  check_eq(value.shape, array.shape[len(pad) :])
  # Create a ragged array, so use dtype=object.
  cval = np.array([[value, value]] * len(pad) + [[0, 0]] * (array.ndim - len(pad)), dtype=object)
  if len(pad) < array.ndim:
    pad = np.concatenate([pad, [[0, 0]] * (array.ndim - len(pad))])
  return np.pad(array, pad, constant_values=cval)


def bounding_slices(a: _ArrayLike, /) -> tuple[slice, ...]:
  """Return the tuple of slices that bound the nonzero elements of array `a`.

  >>> bounding_slices(())
  (slice(0, 0, None),)

  >>> bounding_slices(np.ones(0))
  (slice(0, 0, None),)

  >>> bounding_slices(np.ones((0, 10)))
  (slice(0, 0, None), slice(0, 0, None))

  >>> assert bounding_slices(32.0) == (slice(0, 1, None),)

  >>> assert bounding_slices([0.0, 0.0, 0.0, 0.5, 1.5, 0.0, 2.5, 0.0, 0.0]) == (slice(3, 7, None),)

  >>> a = np.array([0, 0, 6, 7, 0, 0])
  >>> a[bounding_slices(a)]
  array([6, 7])

  >>> a = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
  >>> a[bounding_slices(a)]
  array([[1, 1]])

  >>> assert (bounding_slices([[[0, 0], [0, 1]], [[0, 0], [0, 0]]]) ==
  ...         (slice(0, 1, None), slice(1, 2, None), slice(1, 2, None)))
  """
  a = np.atleast_1d(a)
  slices = []
  for dim in range(a.ndim):
    line = a.any(axis=tuple(i for i in range(a.ndim) if i != dim))
    (indices,) = line.nonzero()
    if indices.size:
      vmin, vmax = indices[[0, -1]]
      slices.append(slice(vmin, vmax + 1))
    else:
      slices.append(slice(0, 0))  # Empty slice.
  return tuple(slices)


def bounding_crop(array: _ArrayLike, value: _ArrayLike, /, *, margin: _ArrayLike = 0) -> _NDArray:
  """Return `array` trimmed where its boundaries equal `value` (which may be an array).

  >>> bounding_crop([[1, 0], [2, 0]], 3)
  array([[1, 0],
         [2, 0]])

  >>> bounding_crop([[1, 0], [2, 0]], 0)
  array([[1],
         [2]])

  >>> bounding_crop([[1, 0], [2, 0]], 0, margin=((1, 0), (1, 1)))
  array([[0, 0, 0],
         [0, 1, 0],
         [0, 2, 0]])

  >>> bounding_crop([[1, 2], [-1, -1]], -1)
  array([[1, 2]])

  >>> bounding_crop([[1, 1], [1, 0]], 1)
  array([[0]])

  >>> bounding_crop([0, 0, 1, 0], 0)
  array([1])

  >>> bounding_crop([0, 0, 1, 0], 0, margin=1)
  array([0, 1, 0])

  >>> bounding_crop([0, 0, 0, 0], 0).tolist()  # array([], dtype=int64) in Unix, dtype=int32 in Win.
  []

  >>> bounding_crop([0, 0, 0, 0], 1)
  array([0, 0, 0, 0])

  >>> bounding_crop([[1, 0], [2, 0], [2, 0]], (2, 0))
  array([[1, 0]])

  """
  array, value = np.asarray(array), np.asarray(value)
  sample_dim = array.ndim - value.ndim
  axis = tuple(range(sample_dim, array.ndim))
  mask = (array != value).any(axis)  # Unfortunately this step is the bottleneck.
  array = array[bounding_slices(mask)]
  return pad_array(array, margin, value)


def _np_int_from_ch(
    a: _ArrayLike, /, int_from_ch: Mapping[str, int], dtype: _DTypeLike = None
) -> _NDArray:
  """Return array of integers created by mapping from an array `a` of characters.

  >>> _np_int_from_ch(np.array(list('abcab')), {'a': 0, 'b': 1, 'c': 2})
  array([0, 1, 2, 0, 1])
  """
  # Adapted from https://stackoverflow.com/a/49566980
  a = np.asarray(a).view(np.int32)
  max_ch = max(a.max(), max(ord(ch) for ch in int_from_ch))
  lookup = np.zeros(max_ch + 1, dtype or np.int_)
  for ch, value in int_from_ch.items():
    lookup[ord(ch)] = value
  return lookup[a]


def grid_from_string(
    string: str, /, int_from_ch: Mapping[str, int] | None = None, dtype: _DTypeLike = None
) -> _NDArray:
  r"""Return a 2D array created from a multiline `string`.

  Args:
    string: Nonempty lines correspond to the rows of the grid, with one `ch` per grid element.
    int_from_ch: Mapping from the `ch` in string to integers in the resulting grid; if None,
      the grid contains chr elements (`dtype='<U1'`).
    dtype: Integer element type for the result of `int_from_ch`.

  >>> string = '..B\nB.A\n'
  >>> g = grid_from_string(string)
  >>> g, g.nbytes
  (array([['.', '.', 'B'],
         ['B', '.', 'A']], dtype='<U1'), 24)

  >>> g = grid_from_string(string, {'.': 0, 'A': 1, 'B': 2})
  >>> g, g.size, g.dtype == np.int_
  (array([[0, 0, 2],
         [2, 0, 1]]), 6, True)

  >>> g = grid_from_string(string, {'.': 0, 'A': 1, 'B': 2}, np.uint8)
  >>> g, g.nbytes
  (array([[0, 0, 2],
         [2, 0, 1]], dtype=uint8), 6)
  """
  # grid = np.array(list(map(list, string.splitlines())))  # Slow.
  lines = string.splitlines()
  height, width = len(lines), len(lines[0])
  grid = np.empty((height, width), 'U1')
  dtype_for_row = f'U{width}'
  for i, line in enumerate(lines):
    grid[i].view(dtype_for_row)[0] = line

  if int_from_ch is None:
    assert dtype is None
  else:
    grid = _np_int_from_ch(grid, int_from_ch, dtype)
  return grid


def string_from_grid(grid: _ArrayLike, /, ch_from_int: Mapping[int, str] | None = None) -> str:
  r"""Return a multiline string created from a 2D array `grid`.

  Args:
    grid: 2D array-like data containing either ch or integers.
    ch_from_int: Mapping from each integer in `grid` to the ch in the resulting string; if None,
      the grid must contain str or byte elements.

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
    /,
    *,
    background: Any = 0,
    foreground: Any = 1,
    indices_min: int | Sequence[int] | None = None,
    indices_max: int | Sequence[int] | None = None,
    pad: int | Sequence[int] = 0,
    dtype: _DTypeLike = None,
) -> _NDArray:
  r"""Return an `array` from (sparse) indices or from a map {index: value}.

  Indices are sequences of integers with some length D, which determines the dimensionality of
  the output `array`.  The array shape is computed by bounding the range of index coordinates in
  each dimension (which may be overridden by `indices_min` and `indices_max`) and is adjusted
  by the `pad` parameter.

  Args:
    iterable_or_map: Iterable of indices or a mapping from indices to values.
    background: Value assigned to the array elements not in `iterable_or_map`.
    foreground: If `iterable_or_map` is an iterable, the array value assigned to its indices.
    indices_min: For each dimension, the index coordinate that gets mapped to coordinate zero in
      the array.  Replicated if an integer.
    indices_max: For each dimension, the index coordinate that gets mapped to the last coordinate
      in the array.  Replicated if an integer.
    pad: For each dimension d, number of additional slices of `background` values before and after
      the range `[indices_min[d], indices_max[d]]`.
    dtype: Data type of the output array.

  Returns:
    array: A D-dimensional numpy array initialized with the value `background` and then sparsely
      assigned the elements in the parameter `iterable_or_map` (using `foreground` value if
      an iterable, or the map values if a map).  By default, `array` spans a tight bounding box
      of the indices, but these bounds can be overridden by using `indices_min`, `indices_max`,
      and `pad`.

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
  i_min = np.min(indices, axis=0) if indices_min is None else np.full(indices.shape[1], indices_min)
  i_max = np.max(indices, axis=0) if indices_max is None else np.full(indices.shape[1], indices_max)
  a_pad = np.asarray(pad)
  shape = i_max - i_min + 2 * a_pad + 1
  offset = -i_min + a_pad
  # pylint: disable-next=possibly-used-before-assignment
  elems = [next(iter(mapping.values()))] if is_map and mapping else []
  elems += [background, foreground]
  shape2 = (*shape, *np.broadcast(*elems).shape)
  del shape
  dtype = np.array(elems[0], dtype).dtype
  grid = np.full(shape2, background, dtype)
  indices += offset
  grid[tuple(indices.T)] = list(mapping.values()) if is_map else foreground
  return grid


def rgb_from_hsx(hsx: _ArrayLike, *, is_hsl: bool) -> _NDArray:
  """Convert from HSV/HSL ([0, 360], [0, 1], [0, 1]) to RGB ([0, 1], [0, 1], [0, 1])."""
  hsx = np.asarray(hsx)
  if hsx.shape[-1] != 3:
    raise ValueError(f'The last dimension in {hsx.shape} is not 3.')
  h, s, v_or_l = hsx[..., 0], hsx[..., 1], hsx[..., 2]
  h = np.mod(h, 360)  # Wrap hue to [0, 360).
  if is_hsl:
    c = (1 - np.abs(2 * v_or_l - 1)) * s  # HSL chroma formula.
    m = v_or_l - 0.5 * c  # HSL match formula.
  else:
    c = v_or_l * s  # HSV chroma formula.
    m = v_or_l - c  # HSV match formula.
  x = c * (1 - np.abs((h / 60) % 2 - 1))  # Secondary component.
  rgb = np.empty_like(hsx)

  mask1 = h < 60  # Six hue sectors.
  mask2 = (60 <= h) & (h < 120)
  mask3 = (120 <= h) & (h < 180)
  mask4 = (180 <= h) & (h < 240)
  mask5 = (240 <= h) & (h < 300)
  mask6 = 300 <= h

  rgb[mask1, 0], rgb[mask1, 1], rgb[mask1, 2] = c[mask1], x[mask1], 0
  rgb[mask2, 0], rgb[mask2, 1], rgb[mask2, 2] = x[mask2], c[mask2], 0
  rgb[mask3, 0], rgb[mask3, 1], rgb[mask3, 2] = 0, c[mask3], x[mask3]
  rgb[mask4, 0], rgb[mask4, 1], rgb[mask4, 2] = 0, x[mask4], c[mask4]
  rgb[mask5, 0], rgb[mask5, 1], rgb[mask5, 2] = x[mask5], 0, c[mask5]
  rgb[mask6, 0], rgb[mask6, 1], rgb[mask6, 2] = c[mask6], 0, x[mask6]

  rgb += m[..., None]  # Add the match value to shift all components to the [0, 1] range.
  return rgb


def rgb_from_hsl(hsl: _ArrayLike) -> _NDArray:
  """Convert from HSL ([0, 360], [0, 1], [0, 1]) to RGB ([0, 1], [0, 1], [0, 1]).

  >>> hsl = (np.indices((10, 10, 10)).T.reshape(10, 100, 3) + 0.5) / 10 * [360, 1, 1]
  >>> assert ((hsl_from_rgb(rgb_from_hsl(hsl)) - hsl) ** 2).sum() < 1e-20
  """
  return rgb_from_hsx(hsl, is_hsl=True)


def rgb_from_hsv(hsv: _ArrayLike) -> _NDArray:
  """Convert from HSV ([0, 360], [0, 1], [0, 1]) to RGB ([0, 1], [0, 1], [0, 1]).

  >>> hsv = (np.indices((10, 10, 10)).T.reshape(10, 100, 3) + 0.5) / 10 * [360, 1, 1]
  >>> assert ((hsv_from_rgb(rgb_from_hsv(hsv)) - hsv) ** 2).sum() < 1e-20
  """
  return rgb_from_hsx(hsv, is_hsl=False)


def hsx_from_rgb(rgb: _ArrayLike, *, use_hsl: bool) -> _NDArray:
  """Convert from RGB ([0, 1], [0, 1], [0, 1]) to HSV/HSL ([0, 360], [0, 1], [0, 1])."""
  rgb = np.asarray(rgb)
  if rgb.shape[-1] != 3:
    raise ValueError(f'The last dimension in {rgb.shape} is not 3.')
  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  hsx = np.zeros_like(rgb)
  h, s, v_or_l = hsx[..., 0], hsx[..., 1], hsx[..., 2]
  max_val = np.max(rgb, axis=-1)
  min_val = np.min(rgb, axis=-1)
  c = max_val - min_val  # Chroma.
  mask_r = (max_val == r) & (c > 0)  # Compute hue.
  mask_g = (max_val == g) & (c > 0)
  mask_b = (max_val == b) & (c > 0)
  h[mask_r] = (g[mask_r] - b[mask_r]) / c[mask_r] % 6
  h[mask_g] = (b[mask_g] - r[mask_g]) / c[mask_g] + 2
  h[mask_b] = (r[mask_b] - g[mask_b]) / c[mask_b] + 4
  h[:] = np.mod(h * 60, 360)

  if use_hsl:
    l = v_or_l
    l[:] = (max_val + min_val) / 2
    s[:] = np.where(c == 0, 0, c / (1 - np.abs(2 * l - 1)))
  else:
    v = v_or_l
    v[:] = max_val
    s[:] = np.where(c == 0, 0, c / v)

  return hsx


def hsl_from_rgb(rgb: _ArrayLike) -> _NDArray:
  """Convert from RGB ([0, 1], [0, 1], [0, 1]) to HSL ([0, 360], [0, 1], [0, 1]).

  >>> rgb = (np.indices((10, 10, 10)).T.reshape(10, 100, 3) + 0.5) / 10
  >>> assert ((rgb_from_hsl(hsl_from_rgb(rgb)) - rgb) ** 2).sum() < 1e-20
  """
  return hsx_from_rgb(rgb, use_hsl=True)


def hsv_from_rgb(rgb: _ArrayLike) -> _NDArray:
  """Convert from RGB ([0, 1], [0, 1], [0, 1]) to HSL ([0, 360], [0, 1], [0, 1]).

  >>> rgb = (np.indices((10, 10, 10)).T.reshape(10, 100, 3) + 0.5) / 10
  >>> assert ((rgb_from_hsv(hsv_from_rgb(rgb)) - rgb) ** 2).sum() < 1e-20
  """
  return hsx_from_rgb(rgb, use_hsl=False)


def generate_random_colors(
    n_colors: int, min_intensity: int = 80, max_intensity: int = 160
) -> _NDArray:
  """Generate a deterministic array of distinct RGB np.uint8 colors with controlled gray intensity.

  >>> colors = generate_random_colors(4)
  >>> assert colors.shape == (4, 3) and colors.dtype == np.uint8
  >>> colors2 = generate_random_colors(20)
  >>> assert all(200 < ptp <= 255 for ptp in np.ptp(colors2, axis=0))
  >>> assert np.all(generate_random_colors(20) == colors2)
  """
  hue = np.linspace(0, 1, n_colors, endpoint=False) * 360
  saturation = np.full(n_colors, 1.0)
  n_intensities = 4
  intensity = (np.arange(n_colors) % n_intensities) / (n_intensities - 1)
  intensity = (intensity * (max_intensity - min_intensity) + min_intensity) / 255
  hsl = np.stack((hue, saturation, intensity), axis=-1)
  np.random.default_rng(1).shuffle(hsl)
  return (rgb_from_hsl(hsl) * 255).astype(np.uint8)


def image_from_yx_map(
    map_yx_value: Mapping[tuple[int, int], Any],
    /,
    background: Any,
    *,
    cmap: Mapping[Any, tuple[int, int, int]],
    pad: int | Sequence[int] = 0,
) -> _NDArray:
  """Return image from mapping `{yx: value}` and `cmap = {value: rgb}`.

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
  image = np.array([cmap[e] for e in array.flat], np.uint8).reshape(*array.shape, 3)
  return image


def _fit_shape(shape: Sequence[int], num: int, /) -> tuple[int, ...]:
  """Given `shape` (with optional -1 dimensions), make it fit `num` elements.

  Args:
    shape: Input dimensions.  Each dimension must either be positive or the special value -1 to
      indicate that it should be computed for tightest fit.  If all dimensions are positive,
      these must satisfy `math.prod(shape) >= num`.
    num: Number of elements to fit into the output shape.

  Returns:
    The original `shape` if all its dimensions are positive.  Otherwise, a new_shape where the
    dimensions with value -1 are replaced by the same smallest number such that
    `math.prod(new_shape) >= num`.

  >>> assert _fit_shape((3,), 2) == (3,)
  >>> assert _fit_shape((-1,), 2) == (2,)
  >>> assert _fit_shape((3, 4), 10) == (3, 4)
  >>> assert _fit_shape((3, -1), 10) == (3, 4)
  >>> assert _fit_shape((-1, 4), 10) == (3, 4)
  >>> assert _fit_shape((-1, 10), 51) == (6, 10)
  >>> assert _fit_shape((-1, -1), 51) == (8, 8)
  >>> assert _fit_shape((-1, -1), 25) == (5, 5)
  >>> assert _fit_shape((-1, 3, -1), 51) == (5, 3, 5)
  >>> _fit_shape((5, 2), 11)
  Traceback (most recent call last):
  ...
  ValueError: (5, 2) is insufficiently large for 11 elements.
  """
  shape = tuple(shape)
  if not all(dim > 0 for dim in shape if dim != -1):
    raise ValueError(f'Shape {shape} has non-positive dimensions.')
  if -1 in shape:
    positive_dims = [dim for dim in shape if dim != -1]
    slice_size = math.prod(positive_dims)  # Note that math.prod([]) == 1.
    n_neg = len(shape) - len(positive_dims)
    new_dim = math.ceil((num / slice_size) ** (1 / n_neg) * (1 - 1e-12))
    shape = tuple(new_dim if dim == -1 else dim for dim in shape)
  elif math.prod(shape) < num:
    raise ValueError(f'{shape} is insufficiently large for {num} elements.')
  return shape


def _offset(cell_length: int, size: int, align: str) -> int:
  """Return an offset to align element of given `size` within `cell_length`."""
  remainder = cell_length - size
  if align not in ('start', 'stop', 'center'):
    raise ValueError(f'Alignment {align} is not recognized.')
  return 0 if align == 'start' else remainder if align == 'stop' else remainder // 2


def assemble_arrays(
    arrays: Sequence[_NDArray] | _NDArray,
    shape: Sequence[int],
    *,
    background: _ArrayLike = 0,
    align: _ArrayLike = 'center',
    spacing: _ArrayLike = 0,
    round_to_even: _ArrayLike = False,
    from_end: bool = False,
) -> _NDArray:
  """Return an output array formed as a packed grid of input arrays.

  Args:
    arrays: Sequence of input arrays with the same data type and rank.  All arrays must share the
      same trailing dimensions, i.e., identical `arrays[:].shape[len(shape):]`.  The leading
      dimensions `arrays[:].shape[:len(shape)]` may be different and these are packed together as a
      grid to form `output.shape[:len(shape)]`.
    shape: Dimensions of the grid used to unravel the `arrays` before packing. The dimensions must
      be positive, with `prod(shape) >= len(arrays)`.  Each dimension must either be positive or
      the special value -1 to indicate that it should be computed for tightest fit.
    background: Broadcastable value used for the unassigned elements of the output array.
    align: Relative position (`'center'`, `'start'`, or `'stop'`) for each input array and for
      each axis within its output grid cell.  The value must be broadcastable onto the shape
      `[len(arrays), len(shape)]`.
    spacing: Extra space between grid elements.  The value may be specified per-axis, i.e.,
      it must be broadcastable onto the shape `[len(shape)]`.
    round_to_even: If True, ensure that the final output dimension of each axis is even.  The
      value must be broadcastable onto the shape `[len(shape)]`.
    from_end: If True, start assigning arrays in reverse order from the end of `shape`.

  Returns:
    A numpy output array of the same type as the input `arrays`, with
    `output.shape = packed_shape + arrays[0].shape[len(shape):]`, where `packed_shape` is
    obtained by packing `arrays[:].shape[:len(shape)]` into a grid of the specified `shape`.

  >>> assemble_arrays(
  ...    [np.array([[1, 2, 3]]), np.array([[4], [5]]), np.array([[6]]),
  ...     np.array([[7, 8]]), np.array([[9, 1, 2]])],
  ...    shape=(2, 3))
  array([[1, 2, 3, 0, 4, 0, 6],
         [0, 0, 0, 0, 5, 0, 0],
         [7, 8, 0, 9, 1, 2, 0]])
  """
  num = len(arrays)
  if num == 0:
    raise ValueError('There must be at least one input array.')
  shape = _fit_shape(shape, num)
  if any(array.dtype != arrays[0].dtype for array in arrays):
    raise ValueError(f'Arrays {arrays} have different types.')
  tail_dims = arrays[0].shape[len(shape) :]
  if any(array.shape[len(shape) :] != tail_dims for array in arrays):
    raise ValueError(f'Shapes of {arrays} do not all end in {tail_dims}')
  align2 = np.broadcast_to(np.asarray(align), (num, len(shape)))
  spacing2 = np.broadcast_to(np.asarray(spacing), len(shape))
  round_to_even2 = np.broadcast_to(np.asarray(round_to_even), len(shape))
  del align, spacing, round_to_even

  head_dims1 = [array.shape[: len(shape)] for array in arrays]
  extra_dims = [(0,) * len(shape)] * (math.prod(shape) - num)
  head_dims1 = extra_dims + head_dims1 if from_end else head_dims1 + extra_dims
  # [*shape] -> leading dimensions [:len(shape)] of each input array.
  head_dims = np.array(head_dims1).reshape(*shape, len(shape))

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

  # [*(shape + 1)] -> start coordinates of cell in output array.
  origins = np.moveaxis(np.array(np.meshgrid(*axis_origins, indexing='ij')), 0, -1)

  # Initialize the output array.
  output_shape = tuple(origins[(-1,) * len(shape)]) + tail_dims
  output_array = np.full(output_shape, background, arrays[0].dtype)

  # Copy each input array to its packed, aligned location in the output array.
  for i, array in enumerate(arrays):
    grid_index = (math.prod(shape) - num) + i if from_end else i
    coords = np.unravel_index(grid_index, shape)
    slices = []
    for axis in range(len(shape)):
      cell_start = origins[coords][axis]
      cell_length = axis_lengths[axis][coords[axis]]
      size = array.shape[axis]
      aligned_start = cell_start + _offset(cell_length, size, align2[i, axis])
      slices.append(slice(aligned_start, aligned_start + size))
    output_array[tuple(slices)] = array

  return output_array


def stack_arrays(
    arrays: Sequence[_ArrayLike],
    *,
    background: _ArrayLike = 0,
    align: _ArrayLike = 'center',
) -> _NDArray:
  """Return an output array formed by stacking uneven input arrays (generalizing `np.stack`).

  Args:
    arrays: Sequence of input arrays with the same data type and rank.
    background: Broadcastable value used for the unassigned elements of the output array.
    align: Relative position (`'center'`, `'start'`, or `'stop'`) for each input array and for
      each axis within its output grid cell.  The value must be broadcastable onto the shape
      `[len(arrays), arrays[0].ndim]`.

  Returns:
    A numpy output array of the same type as the input `arrays`, with an extra first dimension,
    and whose later dimensions are the max of the corresponding dimensions in `arrays`.

  >>> stack_arrays([np.array([1, 2]), np.array([4, 5, 6]), np.array([7])])
  array([[1, 2, 0],
         [4, 5, 6],
         [0, 7, 0]])
  """
  arrays2 = [np.asarray(array) for array in arrays]
  del arrays
  num = len(arrays2)
  if num == 0:
    raise ValueError('There must be at least one input array.')
  if any(array.dtype != arrays2[0].dtype for array in arrays2):
    raise ValueError(f'Arrays {arrays2} have different types.')
  if any(array.ndim != arrays2[0].ndim for array in arrays2):
    raise ValueError(f'Arrays {arrays2} have different ranks.')
  align2 = np.broadcast_to(np.asarray(align), (num, arrays2[0].ndim))
  del align

  # Initialize the output array.
  dims = tuple(max(ds) for ds in zip(*(array.shape for array in arrays2)))
  output_array = np.full((num, *dims), background, arrays2[0].dtype)

  # Copy each input array to its aligned location in the output array.
  for i, array in enumerate(arrays2):
    slices = []
    for axis, size in enumerate(array.shape):
      aligned_start = _offset(dims[axis], size, align2[i, axis])
      slices.append(slice(aligned_start, aligned_start + size))
    output_array[(i, *slices)] = array

  return output_array


def shift(array: _ArrayLike, offset: _ArrayLike, /, constant_values: _ArrayLike = 0) -> _NDArray:
  """Return a copy of the `array` shifted by `offset`, with fill using `constant_values`.

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

  new_array[tuple(slice_axis(o) for o in offset)] = array[tuple(slice_axis(-o) for o in offset)]

  for axis, o in enumerate(offset):
    slices = (slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)
    new_array[slices] = constant_values

  return new_array


@numba.njit(cache=True)  # type: ignore[misc]
def array_index(array: _NDArray, item: Any) -> int:
  """Return the index in `array` of the first element equal to `item`, or -1 if absent.

  See https://stackoverflow.com/a/41578614/1190077

  >>> assert array_index(np.array([], int), 3) == -1
  >>> assert array_index(np.array([1, 2]), 3) == -1
  >>> assert array_index(np.array([1, 2, 1]), 1) == 0
  >>> assert array_index(np.array([1, 2, 1]), 2) == 1
  >>> assert array_index(np.array([[1, 2], [1, 1], [1, 3]]), np.array([1, 4])) == -1
  >>> assert array_index(np.array([[1, 2], [1, 1], [1, 3]]), np.array([1, 1])) == 1
  >>> assert array_index(np.array(list('abcdef')), 'g') == -1
  >>> assert array_index(np.array(list('abcdef')), 'd') == 3
  """
  if array.ndim == 1:
    for i, value in enumerate(array):
      if value == item:
        return i
  else:
    for i, value in enumerate(array):
      if np.all(value == item):
        return i
  return -1


@contextlib.contextmanager
def pil_draw(image: _NDArray) -> Iterator[PIL.ImageDraw.ImageDraw]:
  """Create a PIL.ImageDraw.Draw whose content is copied back to `image` upon exit.

  >>> image = np.full((10, 10, 3), 255, np.uint8)
  >>> (y1, x1), (y2, x2) = (2, 3), (4, 6)
  >>> with pil_draw(image) as draw:
  ...   draw.rectangle(((x1, y1), (x2, y2)), fill=(50, 100, 150), width=0)
  >>> assert np.all(image[2:5, 3:7] == (50, 100, 150))
  >>> image[2:5, 3:7] = 255
  >>> assert np.all(image[2:5, 3:7] == 255)
  """
  import PIL.Image
  import PIL.ImageDraw

  pil_image = PIL.Image.fromarray(image)
  draw = PIL.ImageDraw.Draw(pil_image)
  yield draw
  image[:] = np.asarray(pil_image)


def _get_pil_font(font_size: int, font_name: str) -> Any:
  import matplotlib
  import PIL.ImageFont

  font_file = f'{list(matplotlib.__path__)[0]}/mpl-data/fonts/ttf/{font_name}.ttf'
  return PIL.ImageFont.truetype(font_file, font_size)  # Slow ~1.3 s but gets cached.


def rasterized_text(
    text: str,
    *,
    background: _ArrayLike = 255,
    foreground: _ArrayLike = 0,
    fontname: str = 'cmtt10',
    fontsize: int = 14,
    spacing: int | None = None,
    textalign: str = 'left',
    margin: _ArrayLike = ((4, 1), (1, 1)),  # [[t, b], [l, r]].
    min_width: int = 0,
) -> _NDArray:
  """Returns a uint8 RGB image with the text rasterized into it.

  This function tackles the challenge of letting both the text image size and the text position
  within it be independent of the text content, to avoid jittering in video animations.
  The extents of glyph characters often exceed the ascent and descent (e.g. '['); some characters
  extend left (e.g. 'm' has l=-1) and some extend right (e.g. 'amqRU').  In all cases we rely
  on `margin` to allocate sufficient room.

  Args:
    text: String to rasterize.  Embedded newlines indicate multiline text.
    background: RGB background color of created image.  Scalar indicates gray value.
    foreground: RGB color rasterized text.  Scalar indicates gray value.
    fontname: Name of font compatible with `PIL.ImageFont.truetype()`, such as `'cmtt10'`
      or `'cmr10'`.
    fontsize: Size of rasterized font, in pixels.
    spacing: Number of pixels between lines for multiline text.  If None, selected automatically
      based on `fontsize`.
    textalign: Inter-line horizontal alignment for multiline text: 'left', 'center', or 'right'.
    margin: Number of additional background pixels padded around text.  Must be broadcastable
      onto `[[top, bottom], [left, right]`; see `pad_array()`.
    min_width: Minimum width of returned text image.  This is particularly useful for proportional
      fonts.  Padding is performed using `background` color.

  >>> image = rasterized_text('Hello')
  >>> image[0][0]
  array([255, 255, 255], dtype=uint8)
  >>> 16 <= image.shape[0] <= 18, 38 <= image.shape[1] <= 40, image.shape[2]
  (True, True, 3)

  >>> image = rasterized_text('Hello', background=250, margin=3)
  >>> image[0][0]
  array([250, 250, 250], dtype=uint8)
  >>> 17 <= image.shape[0] <= 19, 42 <= image.shape[1] <= 44, image.shape[2]
  (True, True, 3)
  """
  text = text.rstrip('\n')
  num_lines = text.count('\n') + 1
  background = np.broadcast_to(background, 3)
  foreground = tuple(np.broadcast_to(foreground, 3))
  margin = np.broadcast_to(margin, (2, 2))
  font = _get_pil_font(fontsize, fontname)
  if spacing is None:
    spacing = (fontsize + 6) // 4  # Estimated for 'y['; can be smaller if text lacks brackets.
  assert textalign in ('left', 'center', 'right'), textalign
  draw_args = dict(font=font, anchor='la', spacing=spacing, align=typing.cast(Any, textalign))

  def get_height_width_y() -> tuple[float, float, float]:
    dummy_array = np.full((1, 1, 3), 0, np.uint8)
    with pil_draw(dummy_array) as draw:
      # We could instead use "ascent, descent = font.getmetrics()" but it seems less accurate.
      canonical_text = '\n'.join(['by'] * num_lines)  # Representative ascender and descender.
      unused_l, t, unused_r, b = draw.multiline_textbbox((0, 0), canonical_text, **draw_args)
      width = draw.multiline_textbbox((0, 0), text, **draw_args)[2]
      text1 = re.sub(r'[^\n]', 'm', text)
      width1 = draw.multiline_textbbox((0, 0), text1, **draw_args)[2]
      # Often, the last character on a line is wider by 1 pixel, so we stabilize that case.
      if width + 1 == width1:
        width = width1
      return b - t, width, -t

  height, width, y = get_height_width_y()
  shape = math.ceil(height + margin[0].sum()), math.ceil(max(width + margin[1].sum(), min_width))
  image = np.full((*shape, 3), background, dtype=np.uint8)
  with pil_draw(image) as draw:
    xy = margin[1, 0], margin[0, 0] + y
    draw.text(xy, text, **draw_args, fill=foreground)
  return image


def overlay_text(
    image: _NDArray,
    yx: _ArrayLike,
    text: str,
    align: str = 'tl',  # '[tmb][lcr]'.
    **kwargs: Any,
) -> None:
  """Modifies `image` in-place by overlaying of a box of rasterized `text` at a specified location.

  Args:
    image: uint8 RGB image whose contents are overlaid with a rasterized text box.
    yx: Pixel coordinates `y, x` for placement of the text box, according `align`.
    text: String to rasterize.  Embedded newlines indicate multiline text.
    align: Two-character alignment code [tmb][lcr].  The first character specifies vertical
      alignment about `yx[0]` as `'t'` for top, `'m'` for middle, or `'b'` for bottom.  The second
      character specifies horizontal alignment about `yx[1]` as `'l'` for left, `'c'` for center,
      or `'r'` for right.
    **kwargs: Additional parameters passed to `rasterized_text()`.

  >>> image = np.full((20, 20, 3), 250, np.uint8)
  >>> overlay_text(image, (1, 1), 'H', foreground=30, background=240)
  >>> image[6, :7, 0]
  array([250, 240, 240,  69, 194, 240, 240], dtype=uint8)
  """
  import PIL

  assert image.ndim == 3 and image.dtype == np.uint8
  yx = np.asarray(yx)
  assert yx.shape == (2,)
  assert len(align) == 2 and align[0] in 'tmb' and align[1] in 'lcr'
  if tuple(map(int, PIL.__version__.split('.'))) < (8, 0):
    warnings.warn('Pillow<8.0 lacks ImageDraw.Draw.multiline_textbbox; skipping overlay_text().')
    return
  text_image = rasterized_text(text, **kwargs)
  text_shape, image_shape = text_image.shape[:2], image.shape[:2]
  mid = np.array(text_shape) // 2
  top_left = (
      yx[0] + {'t': 0, 'm': -mid[0], 'b': -text_shape[0]}[align[0]],
      yx[1] + {'l': 0, 'c': -mid[1], 'r': -text_shape[1]}[align[1]],
  )
  slices = tuple(slice(top_left[c], top_left[c] + text_shape[c]) for c in range(2))
  if not all(0 <= s.start <= s.stop <= stop for s, stop in zip(slices, image_shape)):
    raise ValueError(f'Cannot place {text_shape=} at {top_left=} in {image_shape=}; {yx=}.')
  image[slices] = text_image


# ** Graph algorithms


class UnionFind(Generic[_T]):
  """An efficient representation for tracking equivalence classes as elements are unified.

  See https://en.wikipedia.org/wiki/Disjoint-set_data_structure .
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

  def union(self, a: _T, b: _T, /) -> None:
    """Merge the equivalence class of `b` into that of `a`.

    >>> union_find = UnionFind[int]()
    >>> union_find.union(1, 2)
    >>> assert union_find.same(1, 2) and not union_find.same(2, 3)
    """
    rep_a, rep_b = self.find(a), self.find(b)
    self._rep[rep_b] = rep_a

  def same(self, a: _T, b: _T, /) -> bool:
    """Return whether `a` and `b` are in the same equivalence class.

    >>> union_find = UnionFind[int]()
    >>> assert not union_find.same((1, 2), (2, 3))

    >>> union_find.union((1, 2), (2, 3))
    >>> assert union_find.same((1, 2), (2, 3))
    """
    result: bool = self.find(a) == self.find(b)
    return result

  def find(self, a: _T, /) -> _T:
    """Return a representative for the class of `a`; valid until the next `union()` operation.

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
    # while (parent := self._rep.setdefault(a, a)) != a:  # Python 3.10
    while True:
      parent = self._rep.setdefault(a, a)
      if parent == a:
        break
      parents.append(a)
      a = parent
    for p in parents:
      self._rep[p] = a
    return a


# ** Plotting


def graph_layout(graph: Any, *, prog: str) -> dict[Any, tuple[float, float]]:
  """Return dictionary of 2D coordinates for layout of graph nodes."""
  import networkx

  try:
    if sys.platform == 'win32':
      path = pathlib.Path(r'C:\Program Files\Graphviz\bin')
      if path.is_dir() and str(path) not in os.environ['PATH']:
        os.environ['PATH'] += f';{path}'
    args = '-Gstart=1'  # Deterministically seed the graphviz random number generator.
    return networkx.nx_agraph.graphviz_layout(graph, prog=prog, args=args)  # Requires pygraphviz.
  except ImportError:
    pass
  if 0:  # pydot is deprecated; https://github.com/networkx/networkx/issues/5723
    try:
      return networkx.nx_pydot.pydot_layout(graph, prog=prog)  # Requires package pydot.
    except ImportError:
      pass
  print('Cannot reach graphviz; resorting to simpler layout.')
  return networkx.kamada_kawai_layout(graph)


def rotate_layout_so_node_is_on_left(
    pos: dict[_T, tuple[float, float]], special_node: _T, angle: float = 0.0
) -> dict[_T, tuple[float, float]]:
  """Rotate `pos` dict of `x, y` coords (right, up) so special_node is `angle` clw from -X."""
  special_index = list(pos).index(special_node)
  points = np.asarray(list(pos.values()))
  mean_point = points.mean(0)
  translated_points = points - mean_point
  special_point = translated_points[special_index]
  angle = math.tau * 0.5 - np.arctan2(special_point[1], special_point[0]) - angle
  rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  rotated_points = translated_points @ rotation_matrix.T + mean_point
  return {node: tuple(rotated_point) for node, rotated_point in zip(pos, rotated_points)}


def rotate_layout_so_principal_component_is_on_x_axis(
    pos: dict[_T, tuple[float, float]]
) -> dict[_T, tuple[float, float]]:
  """Rotate `pos` dict of `x, y` coords so that its principal axis aligns with the X axis."""
  points = np.asarray(list(pos.values()))
  centered_points = points - points.mean(0)
  cov_matrix = np.cov(centered_points, rowvar=False)
  eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
  order = np.argsort(eigenvalues)[::-1]
  eigenvectors = eigenvectors[:, order]
  rotated_points = centered_points @ eigenvectors
  return {node: tuple(rotated_point) for node, rotated_point in zip(pos, rotated_points)}


def _composite_over_background(image: _NDArray, background: _ArrayLike) -> _NDArray:
  """Return an RGB image by compositing the RGBA `image` over the RGB `background`."""
  assert image.ndim == 3 and image.shape[2] == 4, image.shape
  assert image.dtype == np.uint8, image.dtype
  background_image = np.broadcast_to(np.asarray(background), (*image.shape[:2], 3))
  if np.all(image[..., 3] == 255):
    return image[..., :3]
  alpha = image[..., 3:4] / 255
  premultiplied_alpha = False  # As observed.
  if premultiplied_alpha:
    image = (image[..., :3] + background_image * (1.0 - alpha) + 0.5).astype(np.uint8)
  else:
    image = (image[..., :3] * alpha + background_image * (1.0 - alpha) + 0.5).astype(np.uint8)
  return image


def image_from_plt(fig: Any, background: _ArrayLike = 255) -> _NDArray:
  """Return an RGB image by rasterizing a matplotlib figure `fig` over an RGB `background`."""
  # assert isinstance(fig, matplotlib.figure.Figure)
  # One challenge is that matplotlib.get_backend() == module://matplotlib_inline.backend_inline
  # when it runs in a notebook, but matplotlib.get_backend() == 'agg' when it runs in bash,
  # and this gives rise to subtle differences.  Changing backend temporarily is hard.
  with io.BytesIO() as io_buf:
    # savefig(bbox_inches='tight', pad_inches=0.0) changes dims, so would require format='png'.
    # See https://github.com/matplotlib/matplotlib/issues/17118#issuecomment-612988008
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    shape = int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 4  # RGBA.
    image = np.frombuffer(io_buf.getvalue(), np.uint8).reshape(shape)
    image = _composite_over_background(image, background)
    return image


def images_from_animation(animation: Any, background: _ArrayLike = 255) -> list[_NDArray]:
  """Return a list of RGB images by rendering the matplotlib `animation` over an RGB background."""
  # assert isinstance(animation, matplotlib.animation.Animation)
  import matplotlib.animation

  class ImageListWriter(matplotlib.animation.AbstractMovieWriter):
    """Custom writer to capture image frames."""

    def __init__(self) -> None:
      super().__init__()
      self.images: list[_NDArray] = []
      self.fig: Any  # matplotlib.figure.Figure

    def setup(self, fig: Any, outfile: Any, dpi: float | None = None) -> None:
      del outfile
      self.fig = fig
      assert dpi is None or dpi == self.fig.dpi

    def grab_frame(self, **_: Any) -> None:
      if 0:  # Unnecessary.
        self.fig.canvas.draw()

      if in_notebook():  # Faster but works only if running within a notebook.
        image = np.array(self.fig.canvas.buffer_rgba())  # Not np.asarray() because we need a copy.
        image = _composite_over_background(image, background)

      else:
        # Using fig.savefig() is ~1.7x slower; it is used in *Writer.grab_frame() in
        # https://github.com/matplotlib/matplotlib/blob/v3.8.2/lib/matplotlib/animation.py
        image = image_from_plt(self.fig, background)

      self.images.append(image)

    def finish(self) -> None:
      pass

  writer = ImageListWriter()
  animation.save(None, writer=writer)  # dpi is taken automatically from `fig`.
  return writer.images


def image_from_plotly(fig: Any, **kwargs: Any) -> _NDArray:
  """Return an image obtained by rasterizing a plotly figure."""
  import mediapy as media

  return media.decompress_image(fig.to_image(format='png', **kwargs))[..., :3]


def _from_xyz(d: dict[str, float], /) -> _NDArray:
  """Return an [x, y, z] array from a dict(x=x, y=y, z=z)."""
  return np.array([d['x'], d['y'], d['z']], float)


def _to_xyz(a: _ArrayLike, /) -> dict[str, float]:
  """Return a dict(x=x, y=y, z=z) from an [x, y, z] array."""
  x, y, z = np.asarray(a)
  return dict(x=x, y=y, z=z)


def mesh3d_from_height(
    grid: _ArrayLike, /, *, facecolor: Any = None, color: Any = None, **kwargs: Any
) -> Any:
  """Return a plotly surface formed by extruding square columns from the `grid` height field.

  It is crucial to set lighting=dict(facenormalsepsilon=1e-15) to handle degenerate triangles.
  """
  import plotly.graph_objects as go

  grid = np.asarray(grid)
  assert grid.ndim == 2
  assert not (color is not None and facecolor is None)
  yy, xx = np.arange(grid.shape[0] + 1).repeat(2), np.arange(grid.shape[1] + 1).repeat(2)
  y, x = yy.repeat(len(xx)), np.tile(xx, len(yy))
  z = np.pad(grid.repeat(2, axis=0).repeat(2, axis=1), 1, constant_values=0.0).ravel()
  i, j, k = (
      [
          y2 * len(xx) + x2
          for y1, x1 in np.ndindex((len(yy) - 1, len(xx) - 1))
          for y2, x2 in ((y1 + c0, x1 + c1), (y1 + c2, x1 + c3))
      ]
      for c0, c1, c2, c3 in [(0, 0, 1, 1), (0, 1, 1, 0), (1, 1, 0, 0)]
  )

  if facecolor is not None:
    facecolor2 = np.full(((grid.shape[0] * 2 + 1) * (grid.shape[1] * 2 + 1) * 2, 3), color)
    for y0, x0 in np.ndindex(facecolor.shape[:2]):
      index = ((y0 * 2 + 1) * (grid.shape[1] * 2 + 1) + x0 * 2 + 1) * 2
      facecolor2[index : index + 2] = facecolor[y0, x0]
    facecolor = facecolor2

  intensity = z if facecolor is None else None
  return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=intensity, facecolor=facecolor, **kwargs)


def mesh3d_from_cubes(
    cubes: Iterable[tuple[float, float, float, float, float, float]], facecolors: Any
) -> Any:
  """Return a plotly surface formed by a union of colored cubes."""
  import plotly.graph_objects as go

  x, y, z, i, j, k, colors = [], [], [], [], [], [], []
  for index, (cube, facecolor) in enumerate(zip(cubes, facecolors)):
    x0, y0, z0, x1, y1, z1 = cube
    x.extend([x0, x0, x1, x1, x0, x0, x1, x1])
    y.extend([y0, y1, y1, y0, y0, y1, y1, y0])
    z.extend([z0, z0, z0, z0, z1, z1, z1, z1])
    i.extend([index * 8 + t for t in [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]])
    j.extend([index * 8 + t for t in [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]])
    k.extend([index * 8 + t for t in [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]])
    colors.extend([facecolor] * 12)
  return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=colors, flatshading=True)


def _vector_slerp(a: _ArrayLike, b: _ArrayLike, t: float) -> _NDArray:
  """Spherically interpolate two unit vectors, as in https://en.wikipedia.org/wiki/Slerp ."""
  a, b = np.asarray(a), np.asarray(b)
  angle = max(math.acos(np.dot(a, b)), 1e-10)
  return (math.sin((1.0 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)


def wobble_video(
    fig: Any,
    /,
    *,
    amplitude: float = 1.0,
    num_frames: int = 12,
    quantization: float = 1 / 3,
) -> list[_NDArray]:
  """Return a looping video from a 3D plotly figure by orbiting the eye position left/right.

  Args:
    fig: A `plotly` figure containing a 3D scene.
    amplitude: Magnitude of the angle displacement, in degrees, by which the eye is rotated.
    num_frames: Length of the returned array of frames.
    quantization: Granularity of the orbit angles, to allow frame reuse (e.g. for GIF).
  """
  import plotly

  # Default is [0, 2 / 3, 1, 1, 1, 2 / 3, 0, -2 / 3, -1, -1, -1, -2 / 3].
  rotation_fractions = (
      np.round(
          (np.sin(np.arange(num_frames) / num_frames * math.tau) * 1.05).clip(-1, 1) / quantization
      )
      * quantization
  )
  camera = fig['layout']['scene']['camera']
  if isinstance(camera, plotly.graph_objs.layout.scene.Camera):
    camera = camera.to_plotly_json()

  eye = _from_xyz(camera['eye'])
  up = normalize(_from_xyz(camera.get('up', dict(x=0, y=0, z=1))))
  center = _from_xyz(camera.get('center', dict(x=0, y=0, z=0)))
  from_center = eye - center
  planar_from_center = from_center - up * np.dot(up, from_center)
  unit_planar_from_center = normalize(planar_from_center)
  orthogonal: Any = np.cross(up, unit_planar_from_center)

  image_for_rotation = {}
  for rotation_fraction in set(rotation_fractions):
    angle = rotation_fraction * (amplitude / 360 * math.tau)
    rotation = np.array([math.cos(angle), math.sin(angle)])
    new_unit_planar_from_center = rotation @ [unit_planar_from_center, orthogonal]
    new_planar_from_center = new_unit_planar_from_center * np.linalg.norm(planar_from_center)
    new_eye = eye + (new_planar_from_center - planar_from_center)
    camera2 = camera.copy()
    camera2['eye'] = _to_xyz(new_eye)
    fig.layout.update(scene_camera=camera2)
    image_for_rotation[rotation_fraction] = image_from_plotly(fig)

  fig.layout.update(scene_camera=camera)
  return [image_for_rotation[rotation_fraction] for rotation_fraction in rotation_fractions]


def tilt_video(fig: Any) -> list[_NDArray]:
  """Return a looping video from a 3D plotly figure by displacing the eye towards `camera.up`."""
  import plotly

  rotation_fractions = [0, 0, 0, 1 / 6, 1 / 2, 5 / 6, 0.999, 0.999, 0.999, 5 / 6, 1 / 2, 1 / 6]
  camera = fig['layout']['scene']['camera']
  if isinstance(camera, plotly.graph_objs.layout.scene.Camera):
    camera = camera.to_plotly_json()

  eye = _from_xyz(camera['eye'])
  up = normalize(_from_xyz(camera.get('up', dict(x=0, y=0, z=1))))
  center = _from_xyz(camera.get('center', dict(x=0, y=0, z=0)))
  from_center = eye - center
  unit_from_center = normalize(from_center)

  image_for_rotation = {}
  for rotation_fraction in set(rotation_fractions):
    new_unit_from_center = _vector_slerp(unit_from_center, up, rotation_fraction)
    new_eye = center + np.linalg.norm(from_center) * new_unit_from_center
    camera2 = camera.copy()
    camera2['eye'] = _to_xyz(new_eye)
    fig.layout.update(scene_camera=camera2)
    image_for_rotation[rotation_fraction] = image_from_plotly(fig)

  fig.layout.update(scene_camera=camera)
  return [image_for_rotation[rotation_fraction] for rotation_fraction in rotation_fractions]


# ** Search algorithms


def discrete_binary_search(
    feval: Callable[[int], float], xl: int, xh: int, y_desired: float, /
) -> int:
  """Return `x` such that `feval(x) <= y_desired < feval(x + 1)`.

  Parameters must satisfy `xl < xh` and `feval(xl) <= y_desired < feval(xh)`.

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


@numba.njit  # type: ignore[misc]
def boyer_subsequence_find(seq: _NDArray, subseq: _NDArray, /) -> int:
  """Return the index of the first location of `subseq` in `seq`, or -1 if absent.

  See https://en.wikipedia.org/wiki/Boyer-Moore-Horspool_algorithm.

  Args:
    seq: Sequence to search; it must be an array of non-negative integers.
    subseq: Pattern to locate in the sequence; it must be an array of non-negative integers.

  >>> assert boyer_subsequence_find(np.array([], int), np.array([1])) == -1
  >>> assert boyer_subsequence_find(np.array([2]), np.array([1])) == -1
  >>> assert boyer_subsequence_find(np.array([1, 2]), np.array([1])) == 0
  >>> assert boyer_subsequence_find(np.array([2, 1]), np.array([1])) == 1
  >>> assert boyer_subsequence_find(np.array([1, 1, 2, 1]), np.array([1, 2])) == 1
  >>> assert boyer_subsequence_find(np.array([1, 1, 2, 1]), np.array([2, 2])) == -1
  """
  m, n = len(subseq), len(seq)
  skip_table = np.full(subseq.max() + 1, m)
  for i, value in enumerate(subseq[:-1]):
    skip_table[value] = m - 1 - i

  i = 0
  while i + m <= n:
    j = m - 1
    e = e_last = seq[i + j]
    while True:
      if e != subseq[j]:
        i += skip_table[e_last] if 0 <= e_last < len(skip_table) else m
        break
      if j == 0:
        return i
      j -= 1
      e = seq[i + j]

  return -1


# ** General I/O


def is_executable(path: _Path, /) -> bool:
  """Return True if the file `path` is executable.

  >>> import tempfile
  >>> with tempfile.TemporaryDirectory() as dir:
  ...   path = pathlib.Path(dir) / 'file'
  ...   _ = path.write_text('test', encoding='utf-8')
  ...   check_eq(is_executable(path), False)
  ...   if sys.platform not in ['cygwin', 'win32']:
  ...     # Copy R bits to X bits:
  ...     path.chmod(path.stat().st_mode | ((path.stat().st_mode & 0o444) >> 2))
  ...     check_eq(is_executable(path), True)
  """
  return bool(pathlib.Path(path).stat().st_mode & stat.S_IEXEC)


# ** OS commands


def get_env_bool(name: str, /, default: bool = False) -> bool:
  """Return boolean defined from environment variable `name` or else `default`.

  >>> get_env_bool('ABSENT_VAR')
  False
  >>> get_env_bool('ABSENT_VAR', True)
  True

  >>> os.environ['EXISTENT_VAR'] = '1'
  >>> get_env_bool('EXISTENT_VAR')
  True
  >>> os.environ.pop('EXISTENT_VAR')
  '1'
  >>> get_env_bool('EXISTENT_VAR')
  False
  """
  true_ = ['true', '1', 't']
  false_ = ['false', '0', 'f']
  value: str = os.environ.get(name, '01'[default]).lower()
  if value not in true_ + false_:
    raise ValueError(f'Invalid {value=} for environment variable {name=}.')
  return value in true_


def get_env_int(name: str, /, default: int = 0) -> int:
  """Return integer defined from environment variable `name` or else `default`.

  >>> get_env_int('ABSENT_VAR')
  0
  >>> get_env_int('ABSENT_VAR', 2)
  2

  >>> os.environ['EMPTY_VAR'] = ''
  >>> get_env_int('EMPTY_VAR', 2)
  1

  >>> os.environ['DEFINED_VAR'] = '3'
  >>> get_env_int('DEFINED_VAR', 2)
  3
  >>> os.environ.pop('DEFINED_VAR')
  '3'
  >>> get_env_int('DEFINED_VAR')
  0
  """
  x = os.environ.get(name)
  if x is None:
    return default
  if x == '':
    return 1
  return int(x)


def run(args: str | Sequence[str], /) -> None:
  """Execute the command `args`, printing to stdout its combined output from stdout and stderr.

  Args:
    args: Command to execute, which can be either a string or a sequence of word strings, as in
      `subprocess.run()`.  If `args` is a string, the shell is invoked to interpret it.

  Raises:
    RuntimeError: If the command's exit code is nonzero.

  >>> import tempfile
  >>> with tempfile.TemporaryDirectory() as dir:
  ...   path = pathlib.Path(dir) / 'file'
  ...   run(f'echo ab >{path}')
  ...   assert path.is_file() and 3 <= path.stat().st_size <= 5
  """
  proc = subprocess.run(
      args,
      shell=isinstance(args, str),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
      universal_newlines=True,
  )
  print(proc.stdout, end='', flush=True)
  if proc.returncode:
    raise RuntimeError(f"Command '{proc.args}' failed with code {proc.returncode}.")


if __name__ == '__main__':
  doctest.testmod()
