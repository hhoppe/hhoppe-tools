#!/usr/bin/env python3
# -*- fill-column: 100; -*-
"""Tests for hhoppe_tools module."""

from __future__ import annotations

import math
import sys
import warnings
from typing import Any

import numpy as np
import pytest

import hhoppe_tools as hh

# pylint: disable=missing-function-docstring, protected-access


def test_string_grid_string_roundtrip() -> None:
  s = '..A.\nC.#.\n.AA.\n'
  g = hh.grid_from_string(s, {'.': 0, '#': 1, 'A': 11, 'C': 12}, dtype=np.uint8)
  hh.check_eq(g.dtype, np.uint8)
  hh.check_eq(g.nbytes, 12)
  s2 = hh.string_from_grid(g, {0: '.', 1: '#', 11: 'A', 12: 'C'})
  hh.check_eq(s2, s.strip())

  g = hh.grid_from_string(s)
  hh.check_eq(g.dtype, '<U1')  # single unicode character
  hh.check_eq(g.nbytes, 48)
  hh.check_eq(hh.string_from_grid(g), s.strip())

  g = hh.grid_from_string(s).astype('S1')  # single ascii byte character
  hh.check_eq(g.dtype, '<S1')
  hh.check_eq(g.nbytes, 12)
  s2 = hh.string_from_grid(g)
  hh.check_eq(s2, s.strip())


def test_union_find() -> None:
  union_find = hh.UnionFind[int]()
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
  called_args = []

  @hh.selective_lru_cache(maxsize=None, ignore_kwargs=('kw1', 'kw2'))
  def func1(arg1: int, *, kw0: bool, kw1: bool, kw2: bool, kw3: bool = False) -> int:
    nonlocal called_args
    called_args = [arg1, kw0, kw1, kw2, kw3]
    return arg1 + int(kw0) + int(kw3)

  def f(*args: Any, expected: list[Any], **kwargs: Any) -> None:
    nonlocal called_args
    called_args = []
    func1(*args, **kwargs)
    hh.check_eq(called_args, expected)

  f(1, kw0=False, kw1=False, kw2=False, kw3=False, expected=[1, False, False, False, False])
  f(1, kw0=False, kw1=False, kw2=False, kw3=False, expected=[])
  f(2, kw0=False, kw1=False, kw2=False, kw3=False, expected=[2, False, False, False, False])
  f(2, kw0=False, kw1=True, kw2=True, kw3=False, expected=[])
  f(2, kw0=True, kw1=True, kw2=True, kw3=True, expected=[2, True, True, True, True])
  f(1, kw0=False, kw1=True, kw2=True, kw3=False, expected=[])


def test_stack_arrays() -> None:
  arrays = [[1, 2], [4, 5, 6], [7]]
  np.testing.assert_array_equal(hh.stack_arrays(arrays), [[1, 2, 0], [4, 5, 6], [0, 7, 0]])
  expected_start = [[1, 2, 0], [4, 5, 6], [7, 0, 0]]
  np.testing.assert_array_equal(hh.stack_arrays(arrays, align='start'), expected_start)
  expected_stop = [[0, 1, 2], [4, 5, 6], [0, 0, 7]]
  np.testing.assert_array_equal(hh.stack_arrays(arrays, align='stop'), expected_stop)

  arrays2 = [np.array([[1, 2, 3]]), np.array([[4], [5], [6]]), np.array([[7, 8], [9, 7]])]
  expected = [
      [[0, 0, 0], [0, 0, 0], [1, 2, 3]],
      [[0, 0, 4], [0, 0, 5], [0, 0, 6]],
      [[0, 0, 0], [0, 7, 8], [0, 9, 7]],
  ]
  np.testing.assert_array_equal(hh.stack_arrays(arrays2, align='stop'), expected)
  expected = [
      [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
      [[0, 0, 4], [0, 0, 5], [0, 0, 6]],
      [[0, 7, 8], [0, 9, 7], [0, 0, 0]],
  ]
  np.testing.assert_array_equal(hh.stack_arrays(arrays2, align=['start', 'stop']), expected)
  expected = [
      [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
      [[0, 4, 0], [0, 5, 0], [0, 6, 0]],
      [[0, 0, 0], [0, 7, 8], [0, 9, 7]],
  ]
  result = hh.stack_arrays(arrays2, align=[['start'], ['center'], ['stop']])
  np.testing.assert_array_equal(result, expected)


def test_from_to_xyz() -> None:
  assert hh._to_xyz([0.5, 1, 2]) == dict(x=0.5, y=1, z=2)
  assert np.all(hh._from_xyz(hh._to_xyz([0.5, 1, 2])) == [0.5, 1, 2])


def test_vector_slerp() -> None:
  assert np.allclose(hh._vector_slerp([0, 1], [1, 0], 1 / 3), [0.5, np.cos(np.radians(30))])
  vector = np.array([1.0 + 2**-52, 0.0])  # As if rounding made a unit vector slightly longer.
  assert np.dot(vector, vector) > 1.0
  assert np.allclose(hh._vector_slerp(vector, vector, 0.5), vector)


def test_celltimer_is_noop_outside_notebook(capfd: Any) -> None:
  hh.start_timing_notebook_cells()
  hh.show_notebook_cell_top_times()
  captured = capfd.readouterr()
  assert captured.out == ''


def test_function_in_temporary_module() -> None:
  def function1(a: int, b: int) -> int:
    return a * 10 + b

  with hh.function_in_temporary_module(function1) as function:
    assert function.__name__ == 'function1'
    assert function.__module__.startswith('temp_module_')
    assert function(5, 4) == 54

  def function2(a: int, b: int) -> int:
    return math.prod([2, function1(a, b)])

  header = 'import math'
  with hh.function_in_temporary_module(function2, header=header, funcs=[function1]) as function:
    assert function.__name__ == 'function2'
    assert function.__module__.startswith('temp_module_')
    assert function(5, 4) == 108

  num_paths = len(sys.path)
  with hh.function_in_temporary_module(function1):
    pass
  assert len(sys.path) == num_paths  # The temporary directory is removed from sys.path.


def test_assemble_array() -> None:
  arrays = [
      np.array([[1, 2, 3]]),
      np.array([[4], [5]]),
      np.array([[6]]),
      np.array([[7, 8]]),
      np.array([[9, 1, 2]]),
  ]
  result = hh.assemble_arrays(arrays, shape=(2, 3), from_end=True)
  expected = np.array([[0, 1, 2, 3, 0, 4, 0], [0, 0, 0, 0, 0, 5, 0], [6, 7, 8, 0, 9, 1, 2]])
  np.testing.assert_array_equal(result, expected)


def test_rgb_hsv_hsl_known_values() -> None:
  rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0.25, 0.25]])
  hsv = np.array([[0, 1, 1], [120, 1, 1], [240, 1, 1], [60, 1, 1], [0, 0.5, 0.5]])
  hsl = np.array([[0, 1, 0.5], [120, 1, 0.5], [240, 1, 0.5], [60, 1, 0.5], [0, 1 / 3, 0.375]])
  np.testing.assert_allclose(hh.hsv_from_rgb(rgb), hsv)
  np.testing.assert_allclose(hh.hsl_from_rgb(rgb), hsl)
  np.testing.assert_allclose(hh.rgb_from_hsv(hsv), rgb)
  np.testing.assert_allclose(hh.rgb_from_hsl(hsl), rgb)


def test_rgb_hsv_hsl_edge_cases() -> None:
  np.testing.assert_allclose(hh.hsv_from_rgb([1.0, 0.5, 0.0]), [30, 1, 1])  # A single color.
  np.testing.assert_allclose(hh.hsl_from_rgb([1.0, 0.5, 0.0]), [30, 1, 0.5])
  np.testing.assert_allclose(hh.rgb_from_hsv([30, 1, 1]), [1, 0.5, 0])  # Integer input.
  with warnings.catch_warnings():
    warnings.simplefilter('error')  # No division warnings for black or white.
    np.testing.assert_allclose(hh.hsv_from_rgb([[0.0, 0.0, 0.0]]), [[0, 0, 0]])
    np.testing.assert_allclose(hh.hsl_from_rgb([[1.0, 1.0, 1.0]]), [[0, 0, 1]])


def test_rotate_layout_by_angle() -> None:
  pos = {'a': (0.0, 0.0), 'b': (2.0, 0.0), 'c': (1.0, 3.0)}  # The mean point is (1, 1).
  new_pos = hh.rotate_layout_by_angle(pos, math.tau / 4)  # Counterclockwise.
  np.testing.assert_allclose(new_pos['c'], (-1.0, 1.0), atol=1e-12)


def test_stats_of_large_integers() -> None:
  stats = hh.Stats([4_000_000_000, 0])
  np.testing.assert_allclose(stats.rms(), np.sqrt(np.mean(np.square([4e9, 0]))))
  np.testing.assert_allclose(stats.sdv(), np.std([4e9, 0], ddof=1))


def test_solve_modulo_congruences_without_solution() -> None:
  with pytest.raises(ValueError, match='No solution'):
    hh.solve_modulo_congruences([0, 1], [2, 4])


def test_as_float_of_bool() -> None:
  new = hh.as_float(np.array([True, False]))
  hh.check_eq(new.dtype, np.float32)
  np.testing.assert_array_equal(new, [1.0, 0.0])


def test_grid_from_string_with_ragged_lines() -> None:
  with pytest.raises(ValueError, match='different lengths'):
    hh.grid_from_string('abc\nde\n')


def test_grid_from_indices_with_float_foreground() -> None:
  grid = hh.grid_from_indices([(0, 0), (1, 1)], foreground=0.5)
  np.testing.assert_array_equal(grid, [[0.5, 0.0], [0.0, 0.5]])


def test_image_from_plt_is_writeable() -> None:
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(2, 1))
  image = hh.image_from_plt(fig)
  plt.close(fig)
  assert image.flags.writeable
  image[0, 0] = 0  # E.g., as by overlay_text().
