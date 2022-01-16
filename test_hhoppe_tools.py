#!/usr/bin/env python3
"""Tests for hhoppe_utils module."""
import numpy as np  # type: ignore
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

# Would require adding a "test_requires=['IPython']" in setup.py.
#
# def test_celltimer_is_noop_outside_notebook(capfd):
#   hh.start_timing_notebook_cells()
#   hh.show_notebook_cell_top_times()
#   captured = capfd.readouterr()
#   assert captured.out == ''
