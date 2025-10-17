import json
import os
import tempfile
from cobweb import cobweb_discrete as cd


def build_small_tree():
    tree = cd.CobwebTree(1.0, False, 0, True, False)
    inst1 = {0: {1: 1}}
    inst2 = {0: {2: 1}}
    tree.ifit(inst1, 1)
    tree.ifit(inst2, 1)
    return tree


def test_concept_hash_persistence_and_uniqueness():
    tree = build_small_tree()
    # collect hashes before dump
    pre_hashes = []

    def collect(node):
        pre_hashes.append(node.concept_hash())
        for c in node.children:
            collect(c)

    collect(tree.root)

    # ensure uniqueness among pre-dump nodes
    assert len(pre_hashes) == len(set(pre_hashes)), "Pre-dump concept_hash values are not unique"

    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        tree.write_json_stream(path)
        tree2 = cd.CobwebTree(1.0, False, 0, True, False)
        tree2.load_json_stream(path)

        post_hashes = []

        def collect2(node):
            post_hashes.append(node.concept_hash())
            for c in node.children:
                collect2(c)

        collect2(tree2.root)

        # hashes should match in number and values (order-insensitive)
        assert set(pre_hashes) == set(post_hashes)

        # uniqueness post-load too
        assert len(post_hashes) == len(set(post_hashes)), "Post-load concept_hash values are not unique"
    finally:
        os.remove(path)
