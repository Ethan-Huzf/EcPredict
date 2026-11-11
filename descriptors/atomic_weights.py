"""Normalized atomic weights used by the pentacene OPT3D conversion.

Extracted without rounding from the six arrays in the research libob.py.
Order: U, M, V, P, E, IP, RC. Values are descriptor weights, not SI units.
Only C and H are supported because this release covers pentacene.
"""
WEIGHT_NAMES = ("U", "M", "V", "P", "E", "IP", "RC")
ATOMIC_WEIGHTS = {'H': [1.0, 0.0841, 0.2634, 0.3807, 0.9418, 1.2076, 0.4079], 'C': [1.0, 1, 1, 1, 1, 1.0, 1]}
