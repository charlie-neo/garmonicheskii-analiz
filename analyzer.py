# ============================================================
# analyzer.py
# Модуль анализа гармонии + загрузка обученной нейросети
# ============================================================

import os
import re
import json
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import music21
from music21 import converter, key, roman, chord, pitch

import tensorflow as tf
from tensorflow import keras

import joblib


# ============================================================
# 1. СЛОВАРЬ АККОРДОВ (немецкая система)
# ============================================================
# ============================================================
# 1.1 СУПЕР-СЛОВАРЬ АККОРДОВ 
# ============================================================

chord_patterns = {
    # ==================== C (До) ====================
    # Мажорные
    frozenset(['C', 'E', 'G']): 'C',
    frozenset(['C', 'E']): 'C',
    frozenset(['C', 'G']): 'C',
    # Минорные
    frozenset(['C', 'Es', 'G']): 'Cm',
    frozenset(['C', 'Es']): 'Cm',
    # Увеличенные
    frozenset(['C', 'E', 'G#']): 'Caug',
    frozenset(['C', 'E', 'As']): 'Caug',
    # Уменьшённые
    frozenset(['C', 'Es', 'Ges']): 'Cdim',
    frozenset(['C', 'Es', 'F#']): 'Cdim',
    # Доминантсептаккорды
    frozenset(['C', 'E', 'G', 'B']): 'C7',
    frozenset(['C', 'E', 'G', 'A#']): 'C7',
    # Мажорные септаккорды
    frozenset(['C', 'E', 'G', 'H']): 'Cmaj7',
    # Минорные септаккорды
    frozenset(['C', 'Es', 'G', 'B']): 'Cm7',
    frozenset(['C', 'Es', 'G', 'A#']): 'Cm7',
    # Полууменьшённые
    frozenset(['C', 'Es', 'Ges', 'B']): 'Cm7b5',
    frozenset(['C', 'Es', 'F#', 'B']): 'Cm7b5',
    # Уменьшённые септаккорды
    frozenset(['C', 'Es', 'Ges', 'Bbb']): 'Cdim7',
    frozenset(['C', 'Es', 'F#', 'A']): 'Cdim7',
    # Нонаккорды
    frozenset(['C', 'E', 'G', 'B', 'D']): 'C9',
    frozenset(['C', 'Es', 'G', 'B', 'D']): 'Cm9',
    frozenset(['C', 'E', 'G', 'H', 'D']): 'Cmaj9',

    # ==================== C# / Cis (До-диез) ====================
    frozenset(['C#', 'E#', 'G#']): 'C#',
    frozenset(['C#', 'E#']): 'C#',
    frozenset(['C#', 'G#']): 'C#',
    frozenset(['C#', 'E', 'G#']): 'C#m',
    frozenset(['C#', 'E']): 'C#m',
    frozenset(['C#', 'E#', 'G##']): 'C#aug',
    frozenset(['C#', 'E', 'G']): 'C#dim',
    frozenset(['C#', 'E#', 'G#', 'H']): 'C#7',
    frozenset(['C#', 'E#', 'G#', 'B#']): 'C#maj7',
    frozenset(['C#', 'E', 'G#', 'H']): 'C#m7',
    frozenset(['C#', 'E', 'G', 'H']): 'C#m7b5',
    frozenset(['C#', 'E', 'G', 'B#']): 'C#dim7',

    # ==================== D (Ре) ====================
    # Мажорные
    frozenset(['D', 'F#', 'A']): 'D',
    frozenset(['D', 'F#']): 'D',
    frozenset(['D', 'A']): 'D',
    # Минорные
    frozenset(['D', 'F', 'A']): 'Dm',
    frozenset(['D', 'F']): 'Dm',
    # Увеличенные
    frozenset(['D', 'F#', 'A#']): 'Daug',
    # Уменьшённые
    frozenset(['D', 'F', 'As']): 'Ddim',
    # Доминантсептаккорды
    frozenset(['D', 'F#', 'A', 'C']): 'D7',
    # Мажорные септаккорды
    frozenset(['D', 'F#', 'A', 'C#']): 'Dmaj7',
    # Минорные септаккорды
    frozenset(['D', 'F', 'A', 'C']): 'Dm7',
    # Полууменьшённые
    frozenset(['D', 'F', 'As', 'C']): 'Dm7b5',
    # Уменьшённые септаккорды
    frozenset(['D', 'F', 'As', 'Cb']): 'Ddim7',
    # Нонаккорды
    frozenset(['D', 'F#', 'A', 'C', 'E']): 'D9',
    frozenset(['D', 'F', 'A', 'C', 'E']): 'Dm9',
    frozenset(['D', 'F#', 'A', 'C#', 'E']): 'Dmaj9',

    # ==================== D# / Dis (Ре-диез) ====================
    frozenset(['D#', 'F##', 'A#']): 'D#',
    frozenset(['D#', 'F##']): 'D#',
    frozenset(['D#', 'A#']): 'D#',
    frozenset(['D#', 'F#', 'A#']): 'D#m',
    frozenset(['D#', 'F#']): 'D#m',
    frozenset(['D#', 'F##', 'A##']): 'D#aug',
    frozenset(['D#', 'F#', 'A']): 'D#dim',
    frozenset(['D#', 'F##', 'A#', 'C#']): 'D#7',
    frozenset(['D#', 'F##', 'A#', 'C##']): 'D#maj7',
    frozenset(['D#', 'F#', 'A#', 'C#']): 'D#m7',

    # ==================== Es / Eb (Ми-бемоль) ====================
    # Мажорные
    frozenset(['Es', 'G', 'B']): 'Es',
    frozenset(['Es', 'G']): 'Es',
    frozenset(['Es', 'B']): 'Es',
    # Минорные
    frozenset(['Es', 'Ges', 'B']): 'Esm',
    frozenset(['Es', 'Ges']): 'Esm',
    # Увеличенные
    frozenset(['Es', 'G', 'H']): 'Esaug',
    # Уменьшённые
    frozenset(['Es', 'Ges', 'Bbb']): 'Esdim',
    # Доминантсептаккорды
    frozenset(['Es', 'G', 'B', 'Des']): 'Es7',
    # Мажорные септаккорды
    frozenset(['Es', 'G', 'B', 'D']): 'Esmaj7',
    # Минорные септаккорды
    frozenset(['Es', 'Ges', 'B', 'Des']): 'Esm7',
    # Полууменьшённые
    frozenset(['Es', 'Ges', 'Bbb', 'Des']): 'Esm7b5',
    # Уменьшённые септаккорды
    frozenset(['Es', 'Ges', 'Bbb', 'Dbb']): 'Esdim7',
    # Нонаккорды
    frozenset(['Es', 'G', 'B', 'Des', 'F']): 'Es9',
    frozenset(['Es', 'Ges', 'B', 'Des', 'F']): 'Esm9',

    # ==================== E (Ми) ====================
    # Мажорные
    frozenset(['E', 'G#', 'H']): 'E',
    frozenset(['E', 'G#']): 'E',
    frozenset(['E', 'H']): 'E',
    # Минорные
    frozenset(['E', 'G', 'H']): 'Em',
    frozenset(['E', 'G']): 'Em',
    # Увеличенные
    frozenset(['E', 'G#', 'B#']): 'Eaug',
    # Уменьшённые
    frozenset(['E', 'G', 'B']): 'Edim',
    # Доминантсептаккорды
    frozenset(['E', 'G#', 'H', 'D']): 'E7',
    # Мажорные септаккорды
    frozenset(['E', 'G#', 'H', 'D#']): 'Emaj7',
    # Минорные септаккорды
    frozenset(['E', 'G', 'H', 'D']): 'Em7',
    # Полууменьшённые
    frozenset(['E', 'G', 'B', 'D']): 'Em7b5',
    # Уменьшённые септаккорды
    frozenset(['E', 'G', 'B', 'Des']): 'Edim7',
    # Нонаккорды
    frozenset(['E', 'G#', 'H', 'D', 'F#']): 'E9',
    frozenset(['E', 'G', 'H', 'D', 'F#']): 'Em9',
    frozenset(['E', 'G#', 'H', 'D#', 'F#']): 'Emaj9',

    # ==================== F (Фа) ====================
    # Мажорные
    frozenset(['F', 'A', 'C']): 'F',
    frozenset(['F', 'A']): 'F',
    frozenset(['F', 'C']): 'F',
    # Минорные
    frozenset(['F', 'As', 'C']): 'Fm',
    frozenset(['F', 'As']): 'Fm',
    # Увеличенные
    frozenset(['F', 'A', 'C#']): 'Faug',
    # Уменьшённые
    frozenset(['F', 'As', 'Cb']): 'Fdim',
    # Доминантсептаккорды
    frozenset(['F', 'A', 'C', 'Es']): 'F7',
    # Мажорные септаккорды
    frozenset(['F', 'A', 'C', 'E']): 'Fmaj7',
    # Минорные септаккорды
    frozenset(['F', 'As', 'C', 'Es']): 'Fm7',
    # Полууменьшённые
    frozenset(['F', 'As', 'Cb', 'Es']): 'Fm7b5',
    # Уменьшённые септаккорды
    frozenset(['F', 'As', 'Cb', 'Ebb']): 'Fdim7',
    # Нонаккорды
    frozenset(['F', 'A', 'C', 'Es', 'G']): 'F9',
    frozenset(['F', 'As', 'C', 'Es', 'G']): 'Fm9',
    frozenset(['F', 'A', 'C', 'E', 'G']): 'Fmaj9',

    # ==================== F# / Fis (Фа-диез) ====================
    frozenset(['F#', 'A#', 'C#']): 'F#',
    frozenset(['F#', 'A#']): 'F#',
    frozenset(['F#', 'C#']): 'F#',
    frozenset(['F#', 'A', 'C#']): 'F#m',
    frozenset(['F#', 'A']): 'F#m',
    frozenset(['F#', 'A#', 'C##']): 'F#aug',
    frozenset(['F#', 'A', 'C']): 'F#dim',
    frozenset(['F#', 'A#', 'C#', 'E']): 'F#7',
    frozenset(['F#', 'A#', 'C#', 'E#']): 'F#maj7',
    frozenset(['F#', 'A', 'C#', 'E']): 'F#m7',
    frozenset(['F#', 'A', 'C', 'E']): 'F#m7b5',
    frozenset(['F#', 'A', 'C', 'E#']): 'F#dim7',
    frozenset(['F#', 'A#', 'C#', 'E', 'G#']): 'F#9',
    frozenset(['F#', 'A', 'C#', 'E', 'G#']): 'F#m9',

    # ==================== G (Соль) ====================
    # Мажорные
    frozenset(['G', 'H', 'D']): 'G',
    frozenset(['G', 'H']): 'G',
    frozenset(['G', 'D']): 'G',
    # Минорные
    frozenset(['G', 'B', 'D']): 'Gm',
    frozenset(['G', 'B']): 'Gm',
    # Увеличенные
    frozenset(['G', 'H', 'D#']): 'Gaug',
    # Уменьшённые
    frozenset(['G', 'B', 'Des']): 'Gdim',
    # Доминантсептаккорды
    frozenset(['G', 'H', 'D', 'F']): 'G7',
    # Мажорные септаккорды
    frozenset(['G', 'H', 'D', 'F#']): 'Gmaj7',
    # Минорные септаккорды
    frozenset(['G', 'B', 'D', 'F']): 'Gm7',
    # Полууменьшённые
    frozenset(['G', 'B', 'Des', 'F']): 'Gm7b5',
    # Уменьшённые септаккорды
    frozenset(['G', 'B', 'Des', 'Fb']): 'Gdim7',
    # Нонаккорды
    frozenset(['G', 'H', 'D', 'F', 'A']): 'G9',
    frozenset(['G', 'B', 'D', 'F', 'A']): 'Gm9',
    frozenset(['G', 'H', 'D', 'F#', 'A']): 'Gmaj9',

    # ==================== G# / Gis (Соль-диез) ====================
    frozenset(['G#', 'B#', 'D#']): 'G#',
    frozenset(['G#', 'B#']): 'G#',
    frozenset(['G#', 'D#']): 'G#',
    frozenset(['G#', 'H', 'D#']): 'G#m',
    frozenset(['G#', 'H']): 'G#m',
    frozenset(['G#', 'B#', 'D##']): 'G#aug',
    frozenset(['G#', 'H', 'D']): 'G#dim',
    frozenset(['G#', 'B#', 'D#', 'F#']): 'G#7',
    frozenset(['G#', 'B#', 'D#', 'F##']): 'G#maj7',
    frozenset(['G#', 'H', 'D#', 'F#']): 'G#m7',
    frozenset(['G#', 'H', 'D', 'F#']): 'G#m7b5',
    frozenset(['G#', 'H', 'D', 'F##']): 'G#dim7',

    # ==================== As / Ab (Ля-бемоль) ====================
    # Мажорные
    frozenset(['As', 'C', 'Es']): 'As',
    frozenset(['As', 'C']): 'As',
    frozenset(['As', 'Es']): 'As',
    # Минорные
    frozenset(['As', 'Cb', 'Es']): 'Asm',
    frozenset(['As', 'Cb']): 'Asm',
    # Увеличенные
    frozenset(['As', 'C', 'E']): 'Asaug',
    # Уменьшённые
    frozenset(['As', 'Cb', 'Ebb']): 'Asdim',
    # Доминантсептаккорды
    frozenset(['As', 'C', 'Es', 'Ges']): 'As7',
    # Мажорные септаккорды
    frozenset(['As', 'C', 'Es', 'G']): 'Asmaj7',
    # Минорные септаккорды
    frozenset(['As', 'Cb', 'Es', 'Ges']): 'Asm7',
    # Полууменьшённые
    frozenset(['As', 'Cb', 'Ebb', 'Ges']): 'Asm7b5',
    # Уменьшённые септаккорды
    frozenset(['As', 'Cb', 'Ebb', 'Gbb']): 'Asdim7',
    # Нонаккорды
    frozenset(['As', 'C', 'Es', 'Ges', 'B']): 'As9',
    frozenset(['As', 'Cb', 'Es', 'Ges', 'B']): 'Asm9',

    # ==================== A (Ля) ====================
    # Мажорные
    frozenset(['A', 'C#', 'E']): 'A',
    frozenset(['A', 'C#']): 'A',
    frozenset(['A', 'E']): 'A',
    # Минорные
    frozenset(['A', 'C', 'E']): 'Am',
    frozenset(['A', 'C']): 'Am',
    # Увеличенные
    frozenset(['A', 'C#', 'E#']): 'Aaug',
    # Уменьшённые
    frozenset(['A', 'C', 'Es']): 'Adim',
    # Доминантсептаккорды
    frozenset(['A', 'C#', 'E', 'G']): 'A7',
    # Мажорные септаккорды
    frozenset(['A', 'C#', 'E', 'G#']): 'Amaj7',
    # Минорные септаккорды
    frozenset(['A', 'C', 'E', 'G']): 'Am7',
    # Полууменьшённые
    frozenset(['A', 'C', 'Es', 'G']): 'Am7b5',
    # Уменьшённые септаккорды
    frozenset(['A', 'C', 'Es', 'Ges']): 'Adim7',
    # Нонаккорды
    frozenset(['A', 'C#', 'E', 'G', 'H']): 'A9',
    frozenset(['A', 'C', 'E', 'G', 'H']): 'Am9',
    frozenset(['A', 'C#', 'E', 'G#', 'H']): 'Amaj9',

    # ==================== A# / Ais (Ля-диез) ====================
    frozenset(['A#', 'C##', 'E#']): 'A#',
    frozenset(['A#', 'C##']): 'A#',
    frozenset(['A#', 'E#']): 'A#',
    frozenset(['A#', 'C#', 'E#']): 'A#m',
    frozenset(['A#', 'C#']): 'A#m',
    frozenset(['A#', 'C##', 'E##']): 'A#aug',
    frozenset(['A#', 'C#', 'E']): 'A#dim',
    frozenset(['A#', 'C##', 'E#', 'G#']): 'A#7',
    frozenset(['A#', 'C##', 'E#', 'G##']): 'A#maj7',
    frozenset(['A#', 'C#', 'E#', 'G#']): 'A#m7',
    frozenset(['A#', 'C#', 'E', 'G#']): 'A#m7b5',
    frozenset(['A#', 'C#', 'E', 'G##']): 'A#dim7',

    # ==================== B (Си-бемоль) ====================
    # Мажорные
    frozenset(['B', 'D', 'F']): 'B',
    frozenset(['B', 'D']): 'B',
    frozenset(['B', 'F']): 'B',
    # Минорные
    frozenset(['B', 'Des', 'F']): 'Bm',
    frozenset(['B', 'Des']): 'Bm',
    # Увеличенные
    frozenset(['B', 'D', 'F#']): 'Baug',
    # Уменьшённые
    frozenset(['B', 'Des', 'Fb']): 'Bdim',
    # Доминантсептаккорды
    frozenset(['B', 'D', 'F', 'As']): 'B7',
    # Мажорные септаккорды
    frozenset(['B', 'D', 'F', 'A']): 'Bmaj7',
    # Минорные септаккорды
    frozenset(['B', 'Des', 'F', 'As']): 'Bm7',
    # Полууменьшённые
    frozenset(['B', 'Des', 'Fb', 'As']): 'Bm7b5',
    # Уменьшённые септаккорды
    frozenset(['B', 'Des', 'Fb', 'Abb']): 'Bdim7',
    # Нонаккорды
    frozenset(['B', 'D', 'F', 'As', 'C']): 'B9',
    frozenset(['B', 'Des', 'F', 'As', 'C']): 'Bm9',
    frozenset(['B', 'D', 'F', 'A', 'C']): 'Bmaj9',

    # ==================== H (Си-бекар) ====================
    # Мажорные
    frozenset(['H', 'D#', 'F#']): 'H',
    frozenset(['H', 'D#']): 'H',
    frozenset(['H', 'F#']): 'H',
    # Минорные
    frozenset(['H', 'D', 'F#']): 'Hm',
    frozenset(['H', 'D']): 'Hm',
    # Увеличенные
    frozenset(['H', 'D#', 'F##']): 'Haug',
    # Уменьшённые
    frozenset(['H', 'D', 'F']): 'Hdim',
    # Доминантсептаккорды
    frozenset(['H', 'D#', 'F#', 'A']): 'H7',
    # Мажорные септаккорды
    frozenset(['H', 'D#', 'F#', 'A#']): 'Hmaj7',
    # Минорные септаккорды
    frozenset(['H', 'D', 'F#', 'A']): 'Hm7',
    # Полууменьшённые
    frozenset(['H', 'D', 'F', 'A']): 'Hm7b5',
    # Уменьшённые септаккорды
    frozenset(['H', 'D', 'F', 'As']): 'Hdim7',
    # Нонаккорды
    frozenset(['H', 'D#', 'F#', 'A', 'C#']): 'H9',
    frozenset(['H', 'D', 'F#', 'A', 'C#']): 'Hm9',
    frozenset(['H', 'D#', 'F#', 'A#', 'C#']): 'Hmaj9',
}

# ============================================================
# 2. БАЗОВЫЕ ФУНКЦИИ
# ============================================================

def normalize_pitch(p):
    replacements = {'B-': 'Bb', 'E-': 'Eb', 'A-': 'Ab', 'D-': 'Db', 'G-': 'Gb'}
    for old, new in replacements.items():
        if old in p:
            return p.replace(old, new)
    return p


def get_note_name(note):
    name = normalize_pitch(re.sub(r'\d+$', '', note))
    if name == 'B':
        return 'H'
    if name == 'Bb':
        return 'B'
    if name == 'Eb':
        return 'Es'
    if name == 'Ab':
        return 'As'
    if name == 'Db':
        return 'Des'
    if name == 'Gb':
        return 'Ges'
    return name


def identify_chord(notes):
    if not notes or len(notes) < 2:
        return "?"

    unique = sorted(set([get_note_name(n) for n in notes]))

    for pattern, name in chord_patterns.items():
        if set(pattern) == set(unique):
            return name
    for pattern, name in chord_patterns.items():
        if set(pattern).issubset(set(unique)):
            return name

    try:
        from music21 import chord as chord21, pitch

        pitches = []
        for n in unique:
            if n == 'H':
                n = 'B'
            elif n == 'B':
                n = 'Bb'
            elif n == 'Es':
                n = 'Eb'
            elif n == 'As':
                n = 'Ab'
            elif n == 'Des':
                n = 'Db'
            elif n == 'Ges':
                n = 'Gb'
            pitches.append(pitch.Pitch(n))

        chord_obj = chord21.Chord(pitches)
        root = chord_obj.root().name

        if root == 'B':
            root = 'H'
        elif root == 'B-':
            root = 'B'
        elif root == 'Bb':
            root = 'B'
        elif root == 'E-':
            root = 'Es'
        elif root == 'Eb':
            root = 'Es'
        elif root == 'A-':
            root = 'As'
        elif root == 'Ab':
            root = 'As'
        elif root == 'D-':
            root = 'Des'
        elif root == 'Db':
            root = 'Des'
        elif root == 'G-':
            root = 'Ges'
        elif root == 'Gb':
            root = 'Ges'

        common = chord_obj.commonName.lower()
        if 'major triad' in common:
            return root
        elif 'minor triad' in common:
            return f"{root}m"
        elif 'dominant seventh' in common:
            return f"{root}7"
        elif 'major seventh' in common:
            return f"{root}maj7"
        elif 'minor seventh' in common:
            return f"{root}m7"
        elif 'diminished triad' in common:
            return f"{root}dim"
        elif 'augmented triad' in common:
            return f"{root}aug"
        elif 'diminished seventh' in common:
            return f"{root}dim7"
        elif 'half-diminished' in common:
            return f"{root}m7b5"
        else:
            return root
    except:
        return "?"


def convert_chord_to_russian_german(chord_name):
    if not chord_name or chord_name == '?':
        return '?'

    dash_to_german = {
        'A-': 'As', 'E-': 'Es', 'B-': 'B',
        'D-': 'Des', 'G-': 'Ges',
    }
    for dash, german in dash_to_german.items():
        if chord_name == dash:
            return german
        if chord_name.startswith(dash):
            suffix = chord_name[len(dash):]
            return f"{german}{suffix}"

    explicit_cases = [
        'H', 'Hm', 'H7', 'Hm7', 'Hmaj7', 'Hdim', 'Haug', 'Hdim7',
        'Hm7b5', 'H9', 'Hm9', 'Hmaj9',
        'Es', 'Esm', 'As', 'Asm', 'Des', 'Desm', 'Ges', 'Gesm',
    ]
    if chord_name in explicit_cases:
        return chord_name

    s = str(chord_name)
    s = s.replace('-major', '').replace(' major', '')
    s = s.replace('-minor', 'm').replace(' minor', 'm')
    s = s.replace(' triad', '').replace(' Triad', '')
    s = s.replace(' seventh', '').replace(' Seventh', '')
    s = s.replace(' dominant', '').replace(' Dominant', '')
    s = s.replace(' diminished', '').replace(' Diminished', '')
    s = s.replace(' augmented', '').replace(' Augmented', '')
    s = s.replace(' incomplete', '')
    s = s.replace(' chord', '')
    s = s.strip()

    if not s:
        return '?'

    replacements = {
        'Bb': 'B', 'Eb': 'Es', 'Ab': 'As', 'Db': 'Des', 'Gb': 'Ges',
        'Bbm': 'Bm', 'Ebm': 'Esm', 'Abm': 'Asm', 'Dbm': 'Desm', 'Gbm': 'Gesm',
        'Bb7': 'B7', 'Eb7': 'Es7', 'Ab7': 'As7', 'Db7': 'Des7', 'Gb7': 'Ges7',
        'Bbm7': 'Bm7', 'Ebm7': 'Esm7', 'Abm7': 'Asm7', 'Dbm7': 'Desm7', 'Gbm7': 'Gesm7',
        'B#': 'His', 'B#m': 'Hism', 'B#7': 'His7',
    }
    for eng, rus in replacements.items():
        if s == eng:
            s = rus
            break
        if s.endswith(eng):
            s = s.replace(eng, rus)
            break
        if s.startswith(eng):
            s = s.replace(eng, rus, 1)
            break

    s = re.sub(r'aug$', 'ув', s)
    s = re.sub(r'aug(?=[^a-z])', 'ув', s)
    s = re.sub(r'dim$', 'ум', s)
    s = re.sub(r'dim(?=[^a-z])', 'ум', s)
    s = re.sub(r'maj7$', 'маж7', s)
    s = re.sub(r'm7$', 'м7', s)
    s = re.sub(r'm7(?=[^a-z])', 'м7', s)
    s = re.sub(r'm7b5', 'м7♭5', s)

    return s


def parse_chord_from_string(chord_str):
    if not chord_str:
        return None
    s = str(chord_str)
    s_lower = s.lower()

    if 'diminished' in s_lower and 'seventh' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*diminished', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}dim"
    if 'diminished' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*diminished', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}dim"
    if 'dominant' in s_lower and 'seventh' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*dominant', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}7"
    if 'incomplete' in s_lower and 'dominant' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*incomplete', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}7"
    if 'augmented' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*augmented', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}aug"
    if 'major seventh' in s_lower or 'major 7th' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*major', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}maj7"
    if 'minor seventh' in s_lower or 'minor 7th' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*minor', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}m7"
    if 'dominant' in s_lower:
        m = re.search(r'([A-G][#b]?)\s*-?\s*dominant', s, re.IGNORECASE)
        if m:
            return f"{m.group(1)}7"

    m = re.search(r'([A-G][#b]?)', s)
    if m:
        return m.group(1)
    return None


def fix_complex_chord(chord_name):
    if not chord_name:
        return '?'
    s = str(chord_name)
    if len(s) <= 5 and not any(x in s for x in
                                ['dominant', 'diminished', 'augmented',
                                 'seventh', 'incomplete', 'triad']):
        return s
    parsed = parse_chord_from_string(s)
    return parsed if parsed else '?'


def convert_and_fix_chord(chord_name):
    if not chord_name or chord_name == '?':
        return '?'
    result = convert_chord_to_russian_german(chord_name)
    if result and len(result) > 5:
        result = fix_complex_chord(result)
        if result and result != '?':
            result = convert_chord_to_russian_german(result)
    return result if result and result != '?' else '?'


def remove_consecutive_duplicates(chords_list):
    if not chords_list:
        return []
    result = []
    last = None
    for ch in chords_list:
        if ch != last:
            result.append(ch)
            last = ch
    return result


def get_time_signature_and_measures(score):
    time_signature = None
    for ts in score.flatten().getElementsByClass('TimeSignature'):
        time_signature = ts
        break
    ts_str = (f"{time_signature.numerator}/{time_signature.denominator}"
              if time_signature else "не определён")
    total_measures = 0
    if score.parts:
        part = score.parts[0]
        measures = part.getElementsByClass('Measure')
        if measures:
            total_measures = measures[-1].measureNumber
    return ts_str, total_measures


def fix_last_measure(score, chords_with_measure, total_measures):
    if not chords_with_measure:
        return chords_with_measure

    last_measure_chords = [c for c in chords_with_measure
                           if c['measure'] == total_measures]

    if not last_measure_chords:
        for prev_measure in range(total_measures - 1, 0, -1):
            prev_chords = [c for c in chords_with_measure
                           if c['measure'] == prev_measure]
            if prev_chords:
                chords_with_measure.append({
                    'chord': prev_chords[-1]['chord'],
                    'measure': total_measures
                })
                break
    return chords_with_measure


# ============================================================
# 3. ТОНАЛЬНОСТЬ
# ============================================================
KEY_SIGNATURES = {
    'C': '0 знаков', 'G': '1 диез (Fis)', 'D': '2 диеза (Fis, Cis)',
    'A': '3 диеза (Fis, Cis, Gis)', 'E': '4 диеза (Fis, Cis, Gis, Dis)',
    'H': '5 диезов (Fis, Cis, Gis, Dis, Ais)',
    'F#': '6 диезов', 'C#': '7 диезов',
    'F': '1 бемоль (B)', 'B': '2 бемоля (B, Es)',
    'Es': '3 бемоля (B, Es, As)', 'As': '4 бемоля (B, Es, As, Des)',
    'Des': '5 бемолей', 'Ges': '6 бемолей',
    'Am': '0 знаков', 'Em': '1 диез (Fis)',
    'Hm': '2 диеза (Fis, Cis)',
    'Bm': '5 бемолей (B, Es, As, Des, Ges)',
    'F#m': '3 диеза (Fis, Cis, Gis)',
    'C#m': '4 диеза (Fis, Cis, Gis, Dis)',
    'Dm': '1 бемоль (B)', 'Gm': '2 бемоля (B, Es)',
    'Cm': '3 бемоля (B, Es, As)', 'Fm': '4 бемоля (B, Es, As, Des)',
    'Bbm': '5 бемолей', 'Ebm': '6 бемолей', 'Abm': '7 бемолей',
}

TONALITY_NAMES = {
    'C': 'До мажор', 'G': 'Соль мажор', 'D': 'Ре мажор',
    'A': 'Ля мажор', 'E': 'Ми мажор', 'H': 'Си мажор',
    'F#': 'Фа-диез мажор', 'C#': 'До-диез мажор',
    'F': 'Фа мажор', 'B': 'Си-бемоль мажор',
    'Es': 'Ми-бемоль мажор', 'As': 'Ля-бемоль мажор',
    'Des': 'Ре-бемоль мажор', 'Ges': 'Соль-бемоль мажор',
    'Am': 'Ля минор', 'Em': 'Ми минор',
    'Hm': 'Си минор', 'Bm': 'Си-бемоль минор',
    'F#m': 'Фа-диез минор', 'C#m': 'До-диез минор',
    'Dm': 'Ре минор', 'Gm': 'Соль минор',
    'Cm': 'До минор', 'Fm': 'Фа минор',
    'Bbm': 'Си-бемоль минор', 'Ebm': 'Ми-бемоль минор',
    'Abm': 'Ля-бемоль минор',
}


def determine_tonality(chord_sequence):
    if not chord_sequence:
        return "Тональность не определена"

    clean_chords = [c for c in chord_sequence if c != '?' and c != '—']
    if not clean_chords:
        return "Тональность не определена"

    first_chord = clean_chords[0]
    last_chord = clean_chords[-1]

    def get_root(ch):
        ch = ch.replace('m', '').replace('7', '').replace('6', '').replace('9', '')
        ch = ch.replace('dim', '').replace('aug', '').replace('ув', '').replace('ум', '')
        ch = ch.replace('маж', '').replace('м', '')
        return ch

    def is_minor(ch):
        return 'm' in ch or 'м' in ch

    first_root = get_root(first_chord)
    last_root = get_root(last_chord)
    first_is_minor = is_minor(first_chord)
    last_is_minor = is_minor(last_chord)

    first_key = f"{first_root}{'m' if first_is_minor else ''}"
    last_key = f"{last_root}{'m' if last_is_minor else ''}"

    if first_key != last_key:
        first_signs = KEY_SIGNATURES.get(first_key, "неизвестно")
        last_signs = KEY_SIGNATURES.get(last_key, "неизвестно")
        first_name = TONALITY_NAMES.get(first_key, first_key)
        last_name = TONALITY_NAMES.get(last_key, last_key)
        return f"{first_name} ({first_signs}) → {last_name} ({last_signs}) *модуляция*"
    else:
        signs = KEY_SIGNATURES.get(first_key, "неизвестно")
        russian_name = TONALITY_NAMES.get(first_key, first_key)
        return f"{russian_name} ({signs})"


# ============================================================
# 4. АНАЛИЗАТОРЫ
# ============================================================

def analyze_chorale(file_path):
    try:
        score = converter.parse(file_path)
        chordified = score.chordify()

        chords_with_measure = []
        for element in chordified.recurse().getElementsByClass('Chord'):
            if element.pitches:
                try:
                    root = element.root().name
                    if root == 'B':
                        root = 'H'
                    elif root == 'B-':
                        root = 'B'

                    chord_type = element.commonName
                    if chord_type == 'major triad':
                        chord_name = root
                    elif chord_type == 'minor triad':
                        chord_name = f"{root}m"
                    elif chord_type == 'dominant seventh chord':
                        chord_name = f"{root}7"
                    elif chord_type == 'major seventh chord':
                        chord_name = f"{root}maj7"
                    elif chord_type == 'minor seventh chord':
                        chord_name = f"{root}m7"
                    elif chord_type == 'diminished seventh chord':
                        chord_name = f"{root}dim"
                    elif chord_type == 'augmented triad':
                        chord_name = f"{root}aug"
                    elif chord_type == 'half-diminished seventh chord':
                        chord_name = f"{root}m7b5"
                    else:
                        chord_name = element.pitchedCommonName
                except:
                    chord_name = element.pitchedCommonName

                measure_num = element.measureNumber if element.measureNumber else 1
                chords_with_measure.append({
                    'chord': chord_name,
                    'measure': measure_num
                })

        ts_str, total_measures = get_time_signature_and_measures(score)
        chords_with_measure = fix_last_measure(score, chords_with_measure, total_measures)

        chords_by_measure = {}
        for data in chords_with_measure:
            measure_num = data['measure']
            chord_eng = data['chord']
            if measure_num not in chords_by_measure:
                chords_by_measure[measure_num] = []
            chords_by_measure[measure_num].append(chord_eng)

        measure_strings = []
        all_chords = []

        for measure_num in range(1, total_measures + 1):
            if measure_num in chords_by_measure:
                chords_eng = chords_by_measure[measure_num]
                chords_russian = []
                for ch in chords_eng:
                    if ch and ch != '?':
                        fixed = convert_and_fix_chord(ch)
                        if fixed and fixed != '?':
                            chords_russian.append(fixed)
                if chords_russian:
                    chords_unique = remove_consecutive_duplicates(chords_russian)
                    measure_str = " ".join(chords_unique)
                    all_chords.extend(chords_unique)
                else:
                    measure_str = "—"
            else:
                measure_str = "—"
            measure_strings.append(measure_str)

        chord_string = " | ".join(measure_strings)
        tonality = determine_tonality(all_chords) if all_chords else "Тональность не определена"

        return {
            'type': 'хорал',
            'time_signature': ts_str,
            'total_measures': total_measures,
            'total_chords': len(all_chords),
            'chords': chord_string,
            'tonality': tonality,
            'status': 'success'
        }
    except Exception as e:
        return {'type': 'хорал', 'error': str(e), 'status': 'error'}


def analyze_bass_chord(file_path):
    try:
        score = converter.parse(file_path)
        ts_str, total_measures = get_time_signature_and_measures(score)
        parts = score.parts

        chord_data = None
        part_index = -1

        for idx in range(len(parts) - 1, -1, -1):
            test_part = parts[idx]
            offset_map = {}

            for element in test_part.flatten().notesAndRests:
                if not element.isRest:
                    offset = round(element.offset, 3)
                    if offset not in offset_map:
                        offset_map[offset] = []
                    for p in element.pitches:
                        if p not in offset_map[offset]:
                            offset_map[offset].append(p)
                    offset_map[offset].sort(key=lambda p: p.ps)

            has_chords = False
            for notes in offset_map.values():
                if len(notes) >= 2:
                    has_chords = True
                    break

            if has_chords:
                chord_data = offset_map
                part_index = idx
                break

        if chord_data is None:
            return {
                'type': 'бас+аккорд',
                'time_signature': ts_str,
                'total_measures': total_measures,
                'total_chords': 0,
                'chords': '',
                'tonality': 'Тональность не определена',
                'status': 'success'
            }

        chord_by_offset = {}
        offsets = sorted(chord_data.keys())
        i = 0

        while i < len(offsets):
            current_notes = chord_data[offsets[i]]

            measure_num = None
            for part in score.parts:
                for el in part.flatten().notesAndRests:
                    if not el.isRest and round(el.offset, 3) == offsets[i]:
                        measure = el.getContextByClass('Measure')
                        if measure:
                            measure_num = measure.measureNumber
                        break
                if measure_num:
                    break

            if len(current_notes) == 1 and i + 1 < len(offsets):
                next_notes = chord_data[offsets[i + 1]]
                if len(next_notes) >= 2:
                    all_notes = list(current_notes) + list(next_notes)
                    chord_name = identify_chord([p.name for p in all_notes])
                    if chord_name and chord_name != '?':
                        chord_by_offset[offsets[i + 1]] = (chord_name, measure_num)
                    i += 2
                    continue
            elif len(current_notes) >= 2:
                chord_name = identify_chord([p.name for p in current_notes])
                if chord_name and chord_name != '?':
                    chord_by_offset[offsets[i]] = (chord_name, measure_num)
            i += 1

        bars = {}
        for offset, (chord_name, measure) in chord_by_offset.items():
            if measure not in bars:
                bars[measure] = []
            bars[measure].append(chord_name)

        for measure in bars:
            bars[measure] = remove_consecutive_duplicates(bars[measure])

        temp_chords = []
        for measure, chords in bars.items():
            for ch in chords:
                temp_chords.append({'chord': ch, 'measure': measure})

        temp_chords = fix_last_measure(score, temp_chords, total_measures)

        bars = {}
        for item in temp_chords:
            measure = item['measure']
            chord = item['chord']
            if measure not in bars:
                bars[measure] = []
            bars[measure].append(chord)

        for measure in bars:
            bars[measure] = remove_consecutive_duplicates(bars[measure])

        chord_sequence = []
        for measure in sorted(bars.keys()):
            for ch in bars[measure]:
                converted = convert_and_fix_chord(ch)
                if converted and converted != '?':
                    chord_sequence.append(converted)

        measure_strings = []
        for measure in sorted(bars.keys()):
            chords_in_measure = [convert_and_fix_chord(ch)
                                 for ch in bars[measure]
                                 if convert_and_fix_chord(ch) != '?']
            if chords_in_measure:
                chords_unique = remove_consecutive_duplicates(chords_in_measure)
                measure_strings.append(" ".join(chords_unique))
            else:
                measure_strings.append("—")

        formatted_output = " | ".join(measure_strings)
        tonality = determine_tonality(chord_sequence)

        return {
            'type': 'бас+аккорд',
            'time_signature': ts_str,
            'total_measures': total_measures,
            'total_chords': len(chord_sequence),
            'chords': formatted_output,
            'tonality': tonality,
            'status': 'success'
        }
    except Exception as e:
        return {'type': 'бас+аккорд', 'error': str(e), 'status': 'error'}


def analyze_arpeggio(file_path):
    try:
        score = converter.parse(file_path)
        ts_str, total_measures = get_time_signature_and_measures(score)

        notes_per_chord = 4
        if ts_str != "не определён":
            numerator = int(ts_str.split('/')[0])
            denominator = int(ts_str.split('/')[1])
            if denominator == 8 and numerator in [6, 9, 12]:
                notes_per_chord = 3
            elif numerator == 3 and denominator == 4:
                notes_per_chord = 6
            elif numerator in [2, 4] and denominator == 4:
                notes_per_chord = 4

        bottom_part = score.parts[-1]

        all_notes = []
        for element in bottom_part.flatten().notesAndRests:
            if not element.isRest:
                measure = element.getContextByClass('Measure')
                measure_num = measure.measureNumber if measure else 1
                offset = round(element.offset, 3)
                for p in element.pitches:
                    all_notes.append({
                        'note': p.name,
                        'measure': measure_num,
                        'offset': offset
                    })

        if not all_notes:
            return {'type': 'арпеджио', 'error': 'Нет нот', 'status': 'error'}

        notes_by_measure = {}
        for item in all_notes:
            measure_num = item['measure']
            if measure_num not in notes_by_measure:
                notes_by_measure[measure_num] = []
            notes_by_measure[measure_num].append(item)

        chords_by_measure = {}

        for measure_num in sorted(notes_by_measure.keys()):
            notes = notes_by_measure[measure_num]
            notes.sort(key=lambda x: x['offset'])

            for i in range(0, len(notes), notes_per_chord):
                chord_notes = notes[i:i + notes_per_chord]
                if len(chord_notes) == notes_per_chord:
                    note_names = [n['note'] for n in chord_notes]
                    chord_name = identify_chord(note_names)
                    if chord_name and chord_name != '?':
                        if measure_num not in chords_by_measure:
                            chords_by_measure[measure_num] = []
                        chords_by_measure[measure_num].append(chord_name)

        temp_chords = []
        for measure, chords in chords_by_measure.items():
            for ch in chords:
                temp_chords.append({'chord': ch, 'measure': measure})

        temp_chords = fix_last_measure(score, temp_chords, total_measures)

        chords_by_measure = {}
        for item in temp_chords:
            measure = item['measure']
            chord = item['chord']
            if measure not in chords_by_measure:
                chords_by_measure[measure] = []
            chords_by_measure[measure].append(chord)

        for measure in chords_by_measure:
            chords_by_measure[measure] = remove_consecutive_duplicates(
                chords_by_measure[measure]
            )

        chord_sequence = []
        measure_strings = []

        for measure_num in range(1, total_measures + 1):
            if measure_num in chords_by_measure and chords_by_measure[measure_num]:
                chords_eng = chords_by_measure[measure_num]
                chords_russian = [convert_chord_to_russian_german(ch)
                                  for ch in chords_eng if ch and ch != '?']
                if chords_russian:
                    chords_unique = remove_consecutive_duplicates(chords_russian)
                    measure_str = " ".join(chords_unique)
                    chord_sequence.extend(chords_unique)
                else:
                    measure_str = "—"
            else:
                measure_str = "—"
            measure_strings.append(measure_str)

        chord_string = " | ".join(measure_strings)
        tonality = (determine_tonality(chord_sequence)
                    if chord_sequence else "Тональность не определена")

        return {
            'type': 'арпеджио',
            'time_signature': ts_str,
            'total_measures': total_measures,
            'total_chords': len(chord_sequence),
            'chords': chord_string,
            'tonality': tonality,
            'status': 'success'
        }
    except Exception as e:
        return {'type': 'арпеджио', 'error': str(e), 'status': 'error'}


# ============================================================
# 5. ПРИЗНАКИ ДЛЯ НЕЙРОСЕТИ
# ============================================================

def extract_features_for_nn(file_path):
    try:
        score = converter.parse(file_path)
        features = {}

        total_notes = 0
        total_chords = 0
        chord_sizes = []
        durations = []
        pitches = []
        notes_per_measure = []

        has_melody = False
        if score.parts:
            top_part = score.parts[0]
            top_notes = []
            for element in top_part.flatten().notesAndRests:
                if not element.isRest:
                    for p in element.pitches:
                        top_notes.append(p.midi)

            if len(top_notes) > 5:
                unique_top = len(set(top_notes))
                if unique_top > 3:
                    changes = 0
                    for i in range(len(top_notes) - 1):
                        if top_notes[i + 1] != top_notes[i]:
                            changes += 1
                    if changes > len(top_notes) * 0.3:
                        has_melody = True

        features['has_melody'] = 1 if has_melody else 0

        for part in score.parts:
            for measure in part.getElementsByClass('Measure'):
                notes_in_measure = 0
                for element in measure.flatten().notesAndRests:
                    if not element.isRest:
                        if isinstance(element, music21.note.Note):
                            total_notes += 1
                            notes_in_measure += 1
                            durations.append(float(element.quarterLength))
                            pitches.append(element.pitch.midi)
                        elif isinstance(element, music21.chord.Chord):
                            total_chords += 1
                            chord_sizes.append(len(element.pitches))
                            notes_in_measure += len(element.pitches)
                            for p in element.pitches:
                                pitches.append(p.midi)
                                durations.append(float(element.quarterLength))
                if notes_in_measure > 0:
                    notes_per_measure.append(notes_in_measure)

        features['num_parts'] = len(score.parts)
        features['total_notes'] = total_notes
        features['total_chords'] = total_chords
        features['avg_chord_size'] = np.mean(chord_sizes) if chord_sizes else 0
        features['std_chord_size'] = np.std(chord_sizes) if chord_sizes else 0
        features['avg_duration'] = np.mean(durations) if durations else 0
        features['std_duration'] = np.std(durations) if durations else 0
        features['avg_pitch'] = np.mean(pitches) if pitches else 0
        features['pitch_range'] = max(pitches) - min(pitches) if pitches else 0
        features['avg_notes_per_measure'] = np.mean(notes_per_measure) if notes_per_measure else 0
        features['std_notes_per_measure'] = np.std(notes_per_measure) if notes_per_measure else 0
        features['chord_note_ratio'] = total_chords / total_notes if total_notes > 0 else 0

        features['time_sig_num'] = 4
        features['time_sig_den'] = 4
        try:
            for ts in score.flatten().getElementsByClass(music21.meter.TimeSignature):
                features['time_sig_num'] = ts.numerator
                features['time_sig_den'] = ts.denominator
                break
        except:
            pass

        parts_with_voices = 0
        parts_with_chords = 0

        for part in score.parts:
            max_notes = 0
            for element in part.flatten().notesAndRests:
                if not element.isRest:
                    if isinstance(element, music21.chord.Chord):
                        max_notes = max(max_notes, len(element.pitches))
                    elif isinstance(element, music21.note.Note):
                        max_notes = max(max_notes, 1)

            if max_notes <= 2:
                parts_with_voices += 1
            else:
                parts_with_chords += 1

        features['parts_with_voices'] = parts_with_voices
        features['parts_with_chords'] = parts_with_chords
        features['voice_part_ratio'] = (parts_with_voices / len(score.parts)
                                        if score.parts else 0)

        max_offset_notes = 0
        for part in score.parts:
            for element in part.flatten().notesAndRests:
                if not element.isRest:
                    if isinstance(element, music21.chord.Chord):
                        max_offset_notes = max(max_offset_notes, len(element.pitches))
                    elif isinstance(element, music21.note.Note):
                        max_offset_notes = max(max_offset_notes, 1)

        features['max_notes_in_offset'] = max_offset_notes

        has_alternating = False
        for part in score.parts:
            notes_per_offset = []
            for element in part.flatten().notesAndRests:
                if not element.isRest:
                    if isinstance(element, music21.chord.Chord):
                        notes_per_offset.append(len(element.pitches))
                    elif isinstance(element, music21.note.Note):
                        notes_per_offset.append(1)

            if len(notes_per_offset) > 2:
                for i in range(len(notes_per_offset) - 1):
                    if (notes_per_offset[i] == 1 and notes_per_offset[i + 1] >= 3) or \
                       (notes_per_offset[i] >= 3 and notes_per_offset[i + 1] == 1):
                        has_alternating = True
                        break
                if has_alternating:
                    break

        features['has_alternating'] = 1 if has_alternating else 0

        all_notes_per_offset = []
        for part in score.parts:
            for element in part.flatten().notesAndRests:
                if not element.isRest:
                    if isinstance(element, music21.chord.Chord):
                        all_notes_per_offset.append(len(element.pitches))
                    elif isinstance(element, music21.note.Note):
                        all_notes_per_offset.append(1)

        features['std_notes_per_offset'] = (np.std(all_notes_per_offset)
                                            if all_notes_per_offset else 0)

        all_offsets = set()
        for part in score.parts:
            for element in part.flatten().notesAndRests:
                if not element.isRest:
                    all_offsets.add(round(element.offset, 3))

        features['num_offsets'] = len(all_offsets)
        features['avg_notes_per_offset'] = (np.mean(all_notes_per_offset)
                                            if all_notes_per_offset else 0)

        return features
    except Exception:
        return None


# ============================================================
# 6. ГЛАВНЫЙ КЛАСС — ЗАГРУЗКА МОДЕЛИ + АНАЛИЗ
# ============================================================

class MusicAnalyzer:
    def __init__(self, models_dir="dataset/models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_map = {'хорал': 0, 'бас+аккорд': 1, 'арпеджио': 2}
        self._load()

    def _load(self):
        model_file = self.models_dir / "classifier.h5"
        scaler_file = self.models_dir / "scaler.pkl"
        names_file = self.models_dir / "feature_names.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Нет модели: {model_file}")

        self.model = keras.models.load_model(str(model_file))
        self.scaler = joblib.load(str(scaler_file))
        with open(names_file, 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)

        print(f"✅ Модель загружена: {len(self.feature_names)} признаков")

    def predict_type(self, file_path):
        features = extract_features_for_nn(file_path)
        if features is None:
            return None, 0.0

        X = np.array([[features.get(n, 0) for n in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled, verbose=0)
        idx = int(np.argmax(pred[0]))
        conf = float(pred[0][idx])
        rev = {v: k for k, v in self.label_map.items()}
        return rev[idx], conf

    def analyze(self, file_path):
        file_type, confidence = self.predict_type(file_path)
        if file_type is None:
            return {'status': 'error', 'error': 'Не удалось определить тип'}

        if file_type == 'хорал':
            analysis = analyze_chorale(file_path)
        elif file_type == 'бас+аккорд':
            analysis = analyze_bass_chord(file_path)
        elif file_type == 'арпеджио':
            analysis = analyze_arpeggio(file_path)
        else:
            analysis = {'status': 'error', 'error': f'Неизвестный тип: {file_type}'}

        analysis['predicted_type'] = file_type
        analysis['confidence'] = round(confidence * 100, 2)
        return analysis