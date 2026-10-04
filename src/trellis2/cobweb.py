"""Cobweb for both TRELLIS v2 hierarchies: the compiled tree of cobweb-private.

``cobweb_cu`` (cobweb-private, branch ``karthik-experimental``) implements
Fisher's (1987) category utility over weighted instances whose attributes take
one nominal value or a bag of values (weights summing to one), the four
classic operators (best host, new child, merge, split), and stable leaves: a
fringe split puts a new parent above the old leaf, so a leaf keeps the
instances stored at it, which lets the grammar read-out aggregate records
through any cut of the tree. It reproduces the pure-Python reference in
``tests/trellis2/reference_cobweb.py`` exactly, random tie-breaking included.

Installing cobweb-private provides ``cobweb.cobweb_cu``; a local build
(``cmake -S . -B build && cmake --build build --target cobweb_cu`` in
cobweb-private) provides the module ``cobweb_cu`` once ``build/`` is on the
Python path.
"""
from typing import Dict, Hashable

try:
    from cobweb.cobweb_cu import CobwebCUNode as CobwebNode, CobwebCUTree as CobwebTree
except ImportError:
    try:
        from cobweb_cu import CobwebCUNode as CobwebNode, CobwebCUTree as CobwebTree
    except ImportError as error:
        raise ImportError(
            "TRELLIS v2 needs the compiled Cobweb `cobweb_cu` of cobweb-private: install "
            "cobweb-private, or build its `cobweb_cu` target and put the build directory "
            "on the Python path") from error

Instance = Dict[str, Hashable]

__all__ = ["CobwebNode", "CobwebTree", "Instance"]
