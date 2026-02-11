"""

Primary module for multi-hierarchy theory! We're programming the whole thing from scratch,
going to try to leverage some of the old code but the parse trees need a large rework
especially regarding their visualization (context-hierarchy and content-hierarchy both need
to be worked).

See implementation details in MULTIHIERARCHY.md!
"""
import uuid
import os
import json
import asyncio
from playwright.async_api import async_playwright
import re
from cobweb.cobweb_discrete import CobwebDiscreteTree, CobwebDiscreteNode
from viz import HTMLCobwebDrawer
from typing import List
from sortedcontainers import SortedList
import heapq
import time
import math
import random

class PrimitiveParseNode(object):
    pass

class CompositeParseNode(object):
    pass

class FiniteParseTree(object):
    pass

@DeprecationWarning
class RollingParseTree(object):
    pass


class LongTermMemory(object):
    pass

class WEBSTER(object):
    pass
