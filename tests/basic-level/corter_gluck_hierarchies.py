import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

# ============================================================
# Feature definitions
# ============================================================

# Murphy & Smith features
ms_feature_idx = {'handle': 0, 'shaft': 1, 'head': 2, 'size': 3}
ms_K = {'handle': 5, 'shaft': 5, 'head': 6, 'size': 2}

# Begriffshierarchie features
# shape:       1=jagged-top, 2=round-top
# interior:    1=square, 2=triangle, 3=star, 4=circle
# bottom_edge: 1=wavy, 2=lined
bg_feature_idx = {'shape': 0, 'interior': 1, 'bottom_edge': 2}
bg_K = {'shape': 2, 'interior': 4, 'bottom_edge': 4}

# Fruit data
fruit_feature_idx = {f'f{i}': i for i in range(24)}
fruit_K = {f'f{i}': 2 for i in range(24)}  # binary features, K=2

# Music data
music_feature_idx = {f'f{i}': i for i in range(21)}
music_K = {f'f{i}': 2 for i in range(21)}  # binary features, K=2

# Furniture data
furn_feature_idx = {f'f{i}': i for i in range(24)}
furn_K = {f'f{i}': 2 for i in range(24)}  # binary features, K=2

# ============================================================
# Murphy & Smith data
# ============================================================

ms_items = [
    (1,  'Pounder', 'Hammer',      'Hammer1', 1, 1, 1, 1),
    (2,  'Pounder', 'Hammer',      'Hammer1', 1, 1, 1, 2),
    (3,  'Pounder', 'Hammer',      'Hammer2', 1, 1, 2, 1),
    (4,  'Pounder', 'Hammer',      'Hammer2', 1, 1, 2, 2),
    (5,  'Pounder', 'Brick',       'Brick1',  2, 2, 3, 1),
    (6,  'Pounder', 'Brick',       'Brick1',  2, 2, 3, 2),
    (7,  'Pounder', 'Brick',       'Brick2',  3, 2, 3, 1),
    (8,  'Pounder', 'Brick',       'Brick2',  3, 2, 3, 2),
    (9,  'Cutter',  'Knife',       'Knife1',  4, 3, 4, 1),
    (10, 'Cutter',  'Knife',       'Knife1',  4, 3, 4, 2),
    (11, 'Cutter',  'Knife',       'Knife2',  4, 3, 5, 1),
    (12, 'Cutter',  'Knife',       'Knife2',  4, 3, 5, 2),
    (13, 'Cutter',  'PizzaCutter', 'Pizza1',  5, 4, 6, 1),
    (14, 'Cutter',  'PizzaCutter', 'Pizza1',  5, 4, 6, 2),
    (15, 'Cutter',  'PizzaCutter', 'Pizza2',  5, 5, 6, 1),
    (16, 'Cutter',  'PizzaCutter', 'Pizza2',  5, 5, 6, 2),
]

ms_X = np.array([[row[4], row[5], row[6], row[7]] for row in ms_items])

ms_node_members = {
    # Instances
    'MS_Hammer1-1': [0],  'MS_Hammer1-2': [1],
    'MS_Hammer2-1': [2],  'MS_Hammer2-2': [3],
    'MS_Brick1-1':  [4],  'MS_Brick1-2':  [5],
    'MS_Brick2-1':  [6],  'MS_Brick2-2':  [7],
    'MS_Knife1-1':  [8],  'MS_Knife1-2':  [9],
    'MS_Knife2-1':  [10], 'MS_Knife2-2':  [11],
    'MS_Pizza1-1':  [12], 'MS_Pizza1-2':  [13],
    'MS_Pizza2-1':  [14], 'MS_Pizza2-2':  [15],
    # Subordinate
    'MS_Hammer1': [0,1],   'MS_Hammer2': [2,3],
    'MS_Brick1':  [4,5],   'MS_Brick2':  [6,7],
    'MS_Knife1':  [8,9],   'MS_Knife2':  [10,11],
    'MS_Pizza1':  [12,13], 'MS_Pizza2':  [14,15],
    # Basic
    'MS_Hammer':      [0,1,2,3],
    'MS_Brick':       [4,5,6,7],
    'MS_Knife':       [8,9,10,11],
    'MS_PizzaCutter': [12,13,14,15],
    # Superordinate
    'MS_Pounder': [0,1,2,3,4,5,6,7],
    'MS_Cutter':  [8,9,10,11,12,13,14,15],
    # Root
    'MS_Root': list(range(16)),
}

ms_levels = {
    'Subordinate 2':      ['MS_Hammer1-1','MS_Hammer1-2','MS_Hammer2-1','MS_Hammer2-2',
                      'MS_Brick1-1', 'MS_Brick1-2', 'MS_Brick2-1', 'MS_Brick2-2',
                      'MS_Knife1-1', 'MS_Knife1-2', 'MS_Knife2-1', 'MS_Knife2-2',
                      'MS_Pizza1-1', 'MS_Pizza1-2', 'MS_Pizza2-1', 'MS_Pizza2-2'],
    'Subordinate 1':   ['MS_Hammer1','MS_Hammer2','MS_Brick1','MS_Brick2',
                      'MS_Knife1','MS_Knife2','MS_Pizza1','MS_Pizza2'],
    'Basic':         ['MS_Hammer','MS_Brick','MS_Knife','MS_PizzaCutter'],
    'Superordinate 1': ['MS_Pounder','MS_Cutter'],
    'Superordinate 2':          ['MS_Root'],
}

# ============================================================
# Begriffshierarchie instances
# Each instance is (name, shape, interior, bottom_edge)
# shape:       1=jagged, 2=round
# interior:    1=square, 2=triangle, 3=star, 4=circle
# bottom_edge: 1=spiky, 2=line, 3=square, 4=wavy
# ============================================================

bg_instances = {
    'lun': (1, 1, 1),  # jagged, square,    spiky 
    'fuk': (1, 1, 2),  # jagged, square,    lined
    'tuz': (1, 2, 3),  # jagged, triangle,  square
    'zut': (1, 2, 4),  # jagged, triangle,  wavy
    'nub': (2, 3, 1),  # round,  star,      spiky
    'duw': (2, 3, 2),  # round,  star,      lined
    'pux': (2, 4, 3),  # round,  circle,    square
    'bur': (2, 4, 4),  # round,  circle,    wavy
}

instance_names = ['lun', 'fuk', 'tuz', 'zut', 'nub', 'duw', 'pux', 'bur']
bg_X = np.array([bg_instances[n] for n in instance_names])
instance_idx = {name: i for i, name in enumerate(instance_names)}

def bg_members(names):
    return [instance_idx[n] for n in names]

# ============================================================
# Begriffshierarchie I node members
# ril: kas(lun,fuk) + jad(tuz,zut)   [jagged top]
# mip: gam(nub,duw) + saf(pux,bur)   [round top]
# ============================================================

bg1_node_members = {
    # Instances
    'B1_lun': bg_members(['lun']), 'B1_fuk': bg_members(['fuk']),
    'B1_tuz': bg_members(['tuz']), 'B1_zut': bg_members(['zut']),
    'B1_nub': bg_members(['nub']), 'B1_duw': bg_members(['duw']),
    'B1_pux': bg_members(['pux']), 'B1_bur': bg_members(['bur']),
    # Basic
    'B1_kas': bg_members(['lun','fuk']),
    'B1_jad': bg_members(['tuz','zut']),
    'B1_gam': bg_members(['nub','duw']),
    'B1_saf': bg_members(['pux','bur']),
    # Superordinate
    'B1_ril': bg_members(['lun','fuk','tuz','zut']),
    'B1_mip': bg_members(['nub','duw','pux','bur']),
    # Root
    'B1_Root': bg_members(['lun','fuk','tuz','zut','nub','duw','pux','bur']),
}

bg1_levels = {
    'Subordinate 2':      ['B1_lun','B1_fuk','B1_tuz','B1_zut',
                      'B1_nub','B1_duw','B1_pux','B1_bur'],
    'Subordinate 1':         ['B1_kas','B1_jad','B1_gam','B1_saf'],
    'Basic': ['B1_ril','B1_mip'],
    'Superordinate 1':          ['B1_Root'],
}

# ============================================================
# Begriffshierarchie II node members
# Note leaves use naming of Hierarchy 1.
# ============================================================

bg2_node_members = {
    # Instances (same X data, different grouping)
    'B2_lun': bg_members(['lun']), 'B2_fuk': bg_members(['fuk']),
    'B2_tuz': bg_members(['tuz']), 'B2_zut': bg_members(['zut']),
    'B2_nub': bg_members(['nub']), 'B2_duw': bg_members(['duw']),
    'B2_pux': bg_members(['pux']), 'B2_bur': bg_members(['bur']),
    # Subordinate 
    'B2_kas': bg_members(['lun','fuk']),   # jagged, wavy bottom
    'B2_jad': bg_members(['pux','bur']),   # jagged, lined bottom
    'B2_gam': bg_members(['nub','duw']),   # round, wavy bottom
    'B2_saf': bg_members(['tuz','zut']),   # round, lined bottom
    # Superordinate
    'B2_ril': bg_members(['lun','fuk','pux','bur']),
    'B2_mip': bg_members(['nub','duw','tuz','zut']),
    # Root
    'B2_Root': bg_members(['lun','fuk','tuz','zut','nub','duw','pux','bur']),
}

bg2_levels = {
    'Subordinate 1':      ['B2_lun','B2_fuk','B2_tuz','B2_zut',
                      'B2_nub','B2_duw','B2_pux','B2_bur'],
    'Basic':         ['B2_kas','B2_jad','B2_gam','B2_saf'],
    'Superordinate 1': ['B2_ril','B2_mip'],
    'Superordinate 2':          ['B2_Root'],
}

# ============================================================
# Begriffshierarchie III node members
# Note leaves use naming of Hierarchy 1.
# ============================================================

bg3_node_members = {
    # Instances
    'B3_lun': bg_members(['lun']), 'B3_fuk': bg_members(['fuk']),
    'B3_tuz': bg_members(['tuz']), 'B3_zut': bg_members(['zut']),
    'B3_nub': bg_members(['nub']), 'B3_duw': bg_members(['duw']),
    'B3_pux': bg_members(['pux']), 'B3_bur': bg_members(['bur']),
    # subordinate 
    'B3_kas': bg_members(['lun','pux']),   # H3: kas groups lun+fuk (as H1)
    'B3_jad': bg_members(['zut','duw']),   # H3: jad groups tuz+zut (as H1)
    'B3_gam': bg_members(['nub','tuz']),   # H3: gam groups nub+duw (as H1)
    'B3_saf': bg_members(['bur','fuk']),   # H3: saf groups pux+bur (as H1)
    # super?
    'B3_ril': bg_members(['lun','pux','zut','duw']),  # cross-cuts shape!
    'B3_mip': bg_members(['nub','tuz','bur','fuk']),  # cross-cuts shape!
    # Root
    'B3_Root': bg_members(['lun','fuk','tuz','zut','nub','duw','pux','bur']),
}

bg3_levels = {
    'Basic':      ['B3_lun','B3_fuk','B3_tuz','B3_zut',
                      'B3_nub','B3_duw','B3_pux','B3_bur'],
    'Superordinate 1':         ['B3_kas','B3_jad','B3_gam','B3_saf'],
    'Superordinate 2': ['B3_ril','B3_mip'],
    'Superordinate 3':          ['B3_Root'],
}

# ============================================================
# Fruit hierarchy data
# Features are binary (0/1), 24 features total
# ============================================================

# Revised clean layout:
# 0:  Seeds             (fruit superordinate)
# 1:  Sweet             (fruit superordinate)
# 2:  You eat it        (fruit superordinate)
# 3:  Stem              (apple basic)
# 4:  Core              (apple basic)
# 5:  Skin              (apple AND peach share this)
# 6:  Juicy             (apple basic, cling subordinate, grapes basic)
# 7:  Round             (apple basic)
# 8:  Grows on trees    (apple AND peach share this)
# 9:  Pit               (peach basic)
# 10: Yellow-Orange     (peach basic)
# 11: Fuzzy             (peach basic)
# 12: Soft              (peach basic)
# 13: Bunches           (grapes basic)
# 14: Makes wine        (grapes basic)
# 15: Grows on vine     (grapes basic)
# 16: Red               (delicious subordinate)
# 17: Crisp             (delicious subordinate)
# 18: Shiny             (delicious subordinate)
# 19: Tasty             (delicious subordinate)
# 20: Canned            (cling subordinate)
# 21: Purple            (concord subordinate)
# 22: Green             (green seedless subordinate)
# 23: Small             (green seedless subordinate)

# fruit_instances_raw = {
#     #               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#     'delicious': [  1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
#     'macintosh': [  1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'freestone': [  1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'cling':     [  1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     'concord':   [  1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
#     'green':     [  1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
# }

fruit_instances_raw = {
    #               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    'delicious': [  0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    'macintosh': [  0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'freestone': [  0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'cling':     [  0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'concord':   [  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    'green':     [  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
}

# these are not binary, so set them to 1 for alpha smoothing to work right.
# NOTE, I coded these negatively as 0 as there is only a single value and it should be the 0 one.
fruit_K['f0'] = 1
fruit_K['f1'] = 1
fruit_K['f2'] = 1

fruit_instance_names = ['delicious', 'macintosh', 'freestone', 'cling', 'concord', 'green']

# For binary features, encode as: 0 -> value 1, 1 -> value 2
# so that 1-indexing works consistently with the rest of the code
fruit_X = np.array([
    [v + 1 for v in fruit_instances_raw[n]]
    for n in fruit_instance_names
])  # shape (6, 24), values in {1, 2}

fruit_idx = {name: i for i, name in enumerate(fruit_instance_names)}

def fruit_members(names):
    return [fruit_idx[n] for n in names]

fruit_node_members = {
    # Instances (subordinate level)
    'FR_delicious': fruit_members(['delicious']),
    'FR_macintosh': fruit_members(['macintosh']),
    'FR_freestone': fruit_members(['freestone']),
    'FR_cling':     fruit_members(['cling']),
    'FR_concord':   fruit_members(['concord']),
    'FR_green':     fruit_members(['green']),
    # Basic level
    'FR_apple':  fruit_members(['delicious', 'macintosh']),
    'FR_peach':  fruit_members(['freestone', 'cling']),
    'FR_grapes': fruit_members(['concord', 'green']),
    # Superordinate
    'FR_fruit': fruit_members(['delicious', 'macintosh', 'freestone',
                                'cling', 'concord', 'green']),
}

fruit_levels = {
    'Subordinate 1':         ['FR_delicious', 'FR_macintosh', 'FR_freestone',
                      'FR_cling', 'FR_concord', 'FR_green'],
    'Basic': ['FR_apple', 'FR_peach', 'FR_grapes'],
    'Superordinate 1':            ['FR_fruit'],
}

# =============================================================================
# MUSICAL INSTRUMENT CATEGORY
# =============================================================================

# Revised clean layout:
# 0:  Makes sound
# 1:  Strings          (guitar AND piano share this)
# 2:  Tuning keys
# 3:  Neck
# 4:  Hole
# 5:  Wood
# 6:  Makes music
# 7:  You strum it
# 8:  Used by music groups
# 9:  Keys
# 10: Foot pedals
# 11: Legs
# 12: Lid
# 13: Black keys
# 14: White keys
# 15: Sticks
# 16: Skins
# 17: Round
# 18: Loud
# 19: Large            (grand piano)
# 20: Used in concert halls (grand piano)

# music_instances_raw = {
#     #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
#     'classical_guitar':      [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'folk_guitar':           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'upright_piano':         [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     'grand_piano':           [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
#     'kettle_drum':           [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
#     'bass_drum':             [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
# }

music_instances_raw = {
    #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    'classical_guitar':      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'folk_guitar':           [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'upright_piano':         [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'grand_piano':           [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    'kettle_drum':           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    'bass_drum':             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
}

# This feature is not binary, it is unary with only a single value so set to 1 for alpha smoothing.
# NOTE, that I coded this feature negatively as 0 as all the same at level 0.
music_K['f0'] = 1

music_instance_names = ['classical_guitar', 'folk_guitar', 'upright_piano',
                        'grand_piano', 'kettle_drum', 'bass_drum']

music_X = np.array([
    [v + 1 for v in music_instances_raw[n]]
    for n in music_instance_names
])  # shape (6, 24), values in {1, 2}

music_idx = {name: i for i, name in enumerate(music_instance_names)}

def music_members(names):
    return [music_idx[n] for n in names]

music_node_members = {
    # Instances (subordinate level)
    'MI_classical_guitar': music_members(['classical_guitar']),
    'MI_folk_guitar':      music_members(['folk_guitar']),
    'MI_upright_piano':    music_members(['upright_piano']),
    'MI_grand_piano':      music_members(['grand_piano']),
    'MI_kettle_drum':      music_members(['kettle_drum']),
    'MI_bass_drum':        music_members(['bass_drum']),
    # Basic level
    'MI_guitar': music_members(['classical_guitar', 'folk_guitar']),
    'MI_piano':  music_members(['upright_piano', 'grand_piano']),
    'MI_drum':   music_members(['kettle_drum', 'bass_drum']),
    # Superordinate
    'MI_instrument': music_members(['classical_guitar', 'folk_guitar',
                                    'upright_piano', 'grand_piano',
                                    'kettle_drum', 'bass_drum']),
    # # Root
    # 'MI_Root': music_members(['classical_guitar', 'folk_guitar',
    #                            'upright_piano', 'grand_piano',
    #                            'kettle_drum', 'bass_drum']),
}

music_levels = {
    'Subordinate 1': ['MI_classical_guitar', 'MI_folk_guitar',
                      'MI_upright_piano', 'MI_grand_piano',
                      'MI_kettle_drum', 'MI_bass_drum'],
    'Basic':         ['MI_guitar', 'MI_piano', 'MI_drum'],
    'Superordinate 1':          ['MI_instrument'],
}


# =============================================================================
# FURNITURE CATEGORY
# =============================================================================
# Revised clean layout:
# 0:  Legs              (table AND chair share this)
# 1:  Top
# 2:  Surface
# 3:  Wood              (table AND chair share this)
# 4:  You eat on it
# 5:  You put things on it
# 6:  Light bulb
# 7:  Shade
# 8:  Cord
# 9:  Switches
# 10: Base
# 11: Gives light
# 12: You read by it
# 13: Seat
# 14: Back
# 15: Arms
# 16: Comfortable
# 17: Four legs         (dining room table AND chair share this)
# 18: Holds people
# 19: You sit on it
# 20: Chairs            (kitchen table subordinate)
# 21: Cushion           (living room chair subordinate)
# 22: Large             (living room chair subordinate)
# 23: Soft              (living room chair subordinate)

furn_instances_raw = {
    #                        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    'kitchen_table':        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'dining_room_table':    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'floor_lamp':           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'desk_lamp':            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'kitchen_chair':        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'living_room_chair':    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
}

furn_instance_names = ['kitchen_table', 'dining_room_table', 'floor_lamp',
                       'desk_lamp', 'kitchen_chair', 'living_room_chair']

furn_X = np.array([
    [v + 1 for v in furn_instances_raw[n]]
    for n in furn_instance_names
])  # shape (6, 24), values in {1, 2}

furn_idx = {name: i for i, name in enumerate(furn_instance_names)}

def furn_members(names):
    return [furn_idx[n] for n in names]

furn_node_members = {
    # Instances (subordinate level)
    'FU_kitchen_table':      furn_members(['kitchen_table']),
    'FU_dining_room_table':  furn_members(['dining_room_table']),
    'FU_floor_lamp':         furn_members(['floor_lamp']),
    'FU_desk_lamp':          furn_members(['desk_lamp']),
    'FU_kitchen_chair':      furn_members(['kitchen_chair']),
    'FU_living_room_chair':  furn_members(['living_room_chair']),
    # Basic level
    'FU_table': furn_members(['kitchen_table', 'dining_room_table']),
    'FU_lamp':  furn_members(['floor_lamp', 'desk_lamp']),
    'FU_chair': furn_members(['kitchen_chair', 'living_room_chair']),
    # Superordinate
    'FU_furniture': furn_members(['kitchen_table', 'dining_room_table',
                                   'floor_lamp', 'desk_lamp',
                                   'kitchen_chair', 'living_room_chair']),
}

furn_levels = {
    'Subordinate 1': ['FU_kitchen_table', 'FU_dining_room_table',
                      'FU_floor_lamp', 'FU_desk_lamp',
                      'FU_kitchen_chair', 'FU_living_room_chair'],
    'Basic':         ['FU_table', 'FU_lamp', 'FU_chair'],
    'Superordinate 1':          ['FU_furniture'],
}

# ============================================================
# Dataset registry
# ============================================================

datasets = {
    'Murphy&Smith': {
        'X':            ms_X,
        'feature_idx':  ms_feature_idx,
        'K':            ms_K,
        'node_members': ms_node_members,
        'levels':       ms_levels,
        'leaf_level':   'Subordinate 2',
        # 'leaf_level':   'Root',
        'color':        '#1f77b4',
    },
    'Hierarchy I': {
        'X':            bg_X,
        'feature_idx':  bg_feature_idx,
        'K':            bg_K,
        'node_members': bg1_node_members,
        'levels':       bg1_levels,
        # 'leaf_level':   'Root',
        'leaf_level':   'Subordinate 2',
        'color':        '#2ca02c',
    },
    'Hierarchy II': {
        'X':            bg_X,
        'feature_idx':  bg_feature_idx,
        'K':            bg_K,
        'node_members': bg2_node_members,
        'levels':       bg2_levels,
        # 'leaf_level':   'Root',
        'leaf_level':   'Subordinate 1',
        'color':        '#ff7f0e',
    },
    'Hierarchy III': {
        'X':            bg_X,
        'feature_idx':  bg_feature_idx,
        'K':            bg_K,
        'node_members': bg3_node_members,
        'levels':       bg3_levels,
        'leaf_level':   'Basic',
        # 'leaf_level':   'Root',
        'color':        '#d62728',
    },
    'Musical Instrument': {
        'X':            music_X,
        'feature_idx':  music_feature_idx,
        'K':            music_K,
        'node_members': music_node_members,
        'levels':       music_levels,
        'leaf_level':   'Subordinate 1',
        'color':        '#9467bf',
    },
    'Furniture': {
        'X':            furn_X,
        'feature_idx':  furn_feature_idx,
        'K':            furn_K,
        'node_members': furn_node_members,
        'levels':       furn_levels,
        'leaf_level':   'Subordinate 1',
        'color':        '#8c564b',
    },
    'Fruit': {
        'X':            fruit_X,
        'feature_idx':  fruit_feature_idx,
        'K':            fruit_K,
        'node_members': fruit_node_members,
        'levels':       fruit_levels,
        'leaf_level':   'Subordinate 1',
        'color':        '#e377c2',
    },
}

# ============================================================
# Core computation functions
# ============================================================

def fit_node(member_indices, X, K, feature_idx, alpha):
    probs = {}
    node_X = X[member_indices]
    n = len(member_indices)
    for feat, fidx in feature_idx.items():
        k = K[feat]
        counts = np.zeros(k)
        for val in node_X[:, fidx]:
            counts[int(val) - 1] += 1
        probs[feat] = (counts + alpha) / (n + alpha * k)
    return probs


def compute_log_px(X, leaf_nodes, node_probs, feature_idx):
    N = len(X)
    L = len(leaf_nodes)
    log_weights = np.full(L, -np.log(L))
    log_p_x_given_leaf = np.zeros((L, N))
    for l_idx, leaf in enumerate(leaf_nodes):
        dist = node_probs[leaf]
        for i in range(N):
            log_p = sum(
                np.log(dist[feat][int(X[i, fidx]) - 1])
                for feat, fidx in feature_idx.items()
            )
            log_p_x_given_leaf[l_idx, i] = log_p
    # print()
    # print(log_p_x_given_leaf)
    # print(logsumexp(log_p_x_given_leaf + log_weights[:, None], axis=0))
    return logsumexp(log_p_x_given_leaf + log_weights[:, None], axis=0)


def epmi_own(node_dist, log_px, member_indices, X, feature_idx):
    total = sum(
        sum(np.log(node_dist[feat][int(X[i, fidx]) - 1])
            for feat, fidx in feature_idx.items()) - log_px[i]
        for i in member_indices
    )
    return total / len(member_indices)


def compute_level_scores(dataset, alpha):
    """
    Returns dict: level_name -> mean score across all nodes at that level
    """
    X           = dataset['X']
    feature_idx = dataset['feature_idx']
    K           = dataset['K']
    node_members= dataset['node_members']
    levels      = dataset['levels']
    leaf_level  = dataset['leaf_level']

    # Fit all nodes
    node_probs = {
        node: fit_node(members, X, K, feature_idx, alpha)
        for node, members in node_members.items()
    }

    # Marginal from leaf instances
    leaf_nodes = levels[leaf_level]
    log_px = compute_log_px(X, leaf_nodes, node_probs, feature_idx)

    # Mean score per level
    level_scores = {}
    for level_name, nodes in levels.items():
        scores = [
            epmi_own(node_probs[node], log_px, node_members[node], X, feature_idx)
            for node in nodes
        ]
        level_scores[level_name] = np.mean(scores)

    return level_scores


# ============================================================
# All unique level names across all datasets (for x-axis)
# ============================================================

all_level_names = ['Subordinate 2', 'Subordinate 1', 'Basic', 'Superordinate 1', 'Superordinate 2', 'Superordinate 3']

def level_x(level_name):
    # print(level_name)
    return all_level_names.index(level_name)

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(13, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[15, 1], hspace=0.35)
ax  = fig.add_subplot(gs[0])
ax_slider = fig.add_subplot(gs[1])

slider = Slider(
    ax=ax_slider,
    label='log₁₀(α)',
    valmin=-3,
    valmax=5,
    valinit=np.log10(1),
    valstep=0.01,
)


def draw(alpha):
    ax.clear()

    for ds_name, dataset in datasets.items():
        level_scores = compute_level_scores(dataset, alpha)
        color = dataset['color']

        xs = [level_x(ln) for ln in level_scores.keys()]
        ys = list(level_scores.values())

        ax.plot(xs, ys, color=color, linewidth=2, marker='o',
                markersize=8, label=ds_name, zorder=3)

        # Annotate mean score at each level
        for x, y, ln in zip(xs, ys, level_scores.keys()):
            ax.annotate(f'{y:.2f}', (x, y),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=7, color=color, ha='center')

    ax.set_xticks(range(len(all_level_names)))
    ax.set_xticklabels(all_level_names, fontsize=11)
    ax.set_ylabel('Mean E_{x|c} [ PMI(x,c) ]', fontsize=11)
    ax.set_title(f'Mean PMI Score by Hierarchy Level  (α = {alpha:.4f})', fontsize=13)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.3, len(all_level_names) - 0.7)

    fig.canvas.draw_idle()


def on_slider_change(val):
    draw(10 ** slider.val)

slider.on_changed(on_slider_change)
draw(10 ** slider.valinit)
plt.show()
