#!/usr/bin/env python3
"""Shared reporting helpers.

`report` and `check` compare against the manuscript and fail the run on a mismatch.
`inform` prints a quantity that the manuscript does not claim, so that exploratory
figures remain visible without being mistaken for published values.

Every analysis script prints one line per quantity it checks: the value it
computed and the value printed in the manuscript. A mismatch beyond tolerance is
recorded, listed again at the end, and makes the script exit non-zero, so a
silent disagreement between the code and the paper is not possible.
"""

import sys

import numpy as np

PROBLEMS = []


def report(label, got, want, fmt='{}', tol=0.011):
    """Print computed vs reported; record a mismatch beyond tol."""
    if isinstance(got, (int, float, np.floating)) and isinstance(want, (int, float, np.floating)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = str(got) == str(want)
    if not ok:
        PROBLEMS.append((label, got, want))
    print(f"  {'ok ' if ok else 'DIFF'} {label:<46} computed {fmt.format(got):<22} "
          f"reported {fmt.format(want)}")


def inform(label, got, fmt='{}', note=''):
    """Print a quantity that is computed for information and is not a claim of the
    manuscript. It is not compared against anything, so it can never produce a
    DIFF; the note says why it is printed."""
    line = f"  --  {label:<46} computed {fmt.format(got):<22}"
    print(line + (f"  ({note})" if note else ''))


def check(label, condition, detail=''):
    """Assert a qualitative property; recorded the same way as a numeric mismatch."""
    if not condition:
        PROBLEMS.append((label, detail, 'must hold'))
    print(f"  {'ok ' if condition else 'DIFF'} {label:<46} {detail}")


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def finish(scope='the manuscript'):
    section('Summary')
    if PROBLEMS:
        print(f"{len(PROBLEMS)} value(s) differ from {scope}:")
        for label, got, want in PROBLEMS:
            print(f"  {label}: computed {got}, reported {want}")
        sys.exit(1)
    print(f"every checked value matches {scope}.")
