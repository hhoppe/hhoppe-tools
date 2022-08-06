#!/usr/bin/env python3
# -*- fill-column: 80; -*-
"""Tests for hhoppe_tools module."""

from __future__ import annotations

from typing import Any

import numpy as np
import hhoppe_tools as hh


def test_string_grid_string_roundtrip() -> None:
  s = '..A.\nC.#.\n.AA.\n'
  g = hh.grid_from_string(s, {'.': 0, '#': 1, 'A': 11, 'C': 12}, dtype=np.uint8)
  hh.check_eq(g.dtype, np.uint8)
  hh.check_eq(g.nbytes, 12)
  s2 = hh.string_from_grid(g, {0: '.', 1: '#', 11: 'A', 12: 'C'})
  hh.check_eq(s2, s.strip())

  g = hh.grid_from_string(s)
  hh.check_eq(g.dtype, '<U1')      # single unicode character
  hh.check_eq(g.nbytes, 48)
  hh.check_eq(hh.string_from_grid(g), s.strip())

  g = hh.grid_from_string(s).astype('S1')  # single ascii byte character
  hh.check_eq(g.dtype, '<S1')
  hh.check_eq(g.nbytes, 12)
  s2 = hh.string_from_grid(g)
  hh.check_eq(s2, s.strip())


def test_union_find() -> None:
  union_find = hh.UnionFind()
  hh.check_eq(union_find.same(12, 12), True)
  hh.check_eq(union_find.same(12, 23), False)
  hh.check_eq(union_find.same(12, 35), False)
  hh.check_eq(union_find.same(23, 35), False)
  union_find.union(12, 23)
  hh.check_eq(union_find.same(12, 12), True)
  hh.check_eq(union_find.same(12, 23), True)
  hh.check_eq(union_find.same(12, 35), False)
  hh.check_eq(union_find.same(23, 35), False)
  union_find.union(23, 35)
  hh.check_eq(union_find.same(12, 12), True)
  hh.check_eq(union_find.same(12, 23), True)
  hh.check_eq(union_find.same(12, 35), True)
  hh.check_eq(union_find.same(23, 35), True)


def test_noop_decorator() -> None:
  @hh.noop_decorator
  def func1(i: int) -> int:
    return i * 2

  @hh.noop_decorator()
  def func2(i: int) -> int:
    return i * 2

  @hh.noop_decorator('some_argument')
  def func3(i: int) -> int:
    return i * 2

  @hh.noop_decorator(some_kwarg='value')
  def func4(i: int) -> int:
    return i * 2

  hh.check_eq(func1(1), 2)
  hh.check_eq(func2(1), 2)
  hh.check_eq(func3(1), 2)
  hh.check_eq(func4(1), 2)


def test_selective_lru_cache() -> None:
  """Test `selective_lru_cache` where we ignore two keyword parameters."""
  called_args = []

  @hh.selective_lru_cache(maxsize=None, ignore_kwargs=('kw1', 'kw2'))
  def func1(arg1: int, *,
            kw0: bool, kw1: bool, kw2: bool, kw3: bool = False) -> int:
    """Dummy test function."""
    nonlocal called_args
    called_args = [arg1, kw0, kw1, kw2, kw3]
    return arg1 + int(kw0) + int(kw3)

  def f(*args: Any, expected: list[Any], **kwargs: Any) -> None:
    nonlocal called_args
    called_args = []
    func1(*args, **kwargs)
    hh.check_eq(called_args, expected)

  f(1, kw0=False, kw1=False, kw2=False, kw3=False,
    expected=[1, False, False, False, False])
  f(1, kw0=False, kw1=False, kw2=False, kw3=False, expected=[])
  f(2, kw0=False, kw1=False, kw2=False, kw3=False,
    expected=[2, False, False, False, False])
  f(2, kw0=False, kw1=True, kw2=True, kw3=False, expected=[])
  f(2, kw0=True, kw1=True, kw2=True, kw3=True,
    expected=[2, True, True, True, True])
  f(1, kw0=False, kw1=True, kw2=True, kw3=False, expected=[])


# Would require adding a "test_requires=['IPython']" in setup.py.
#
# def test_celltimer_is_noop_outside_notebook(capfd):
#   hh.start_timing_notebook_cells()
#   hh.show_notebook_cell_top_times()
#   captured = capfd.readouterr()
#   assert captured.out == ''
