
graph = [
    {"id": "a", "type": "edible", "components": ["c1, c2"]},
    {"id": "c1", "diameter": 2.0},
    {"id": "c2", "length": 1.0}
]

class Graph:

    def __init__(self, _id, components, diameter, length):
        self.graph = [
            {"id": _id, "type": "edible", "components": components},
            {"id": components, "diameter": diameter},
            {"id": components, "length": length}
        ]

