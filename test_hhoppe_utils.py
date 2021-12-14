import hhoppe_utils as hh
import numpy as np  # type: ignore


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

