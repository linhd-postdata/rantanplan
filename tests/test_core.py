import json
from pathlib import Path
from unittest import mock

import pytest
import spacy

import rantanplan.core
from rantanplan.core import _get_scansion
from rantanplan.core import apply_exception_rules
from rantanplan.core import apply_exception_rules_post
from rantanplan.core import clean_phonological_groups
from rantanplan.core import format_stress
from rantanplan.core import generate_liaison_positions
from rantanplan.core import generate_phonological_groups
from rantanplan.core import get_last_syllable
from rantanplan.core import get_orthographic_accent
from rantanplan.core import get_phonological_groups
from rantanplan.core import get_rhythmical_pattern
from rantanplan.core import get_scansion
from rantanplan.core import get_stresses
from rantanplan.core import get_syllables_word_end
from rantanplan.core import get_word_stress
from rantanplan.core import get_words
from rantanplan.core import has_single_liaisons
from rantanplan.core import have_prosodic_liaison
from rantanplan.core import is_paroxytone
from rantanplan.core import remove_exact_length_matches
from rantanplan.core import set_stress_exceptions
from rantanplan.core import spacy_tag_to_dict
from rantanplan.core import syllabify

nlp = spacy.load('es_core_news_md')


@pytest.fixture
def phonological_groups():
    return json.loads(
        Path("tests/fixtures/phonological_groups.json").read_text())


@pytest.fixture
def sonnet():
    return json.loads(
        Path("tests/fixtures/sonnet.json").read_text())


@pytest.fixture
def scansion_sonnet():
    return json.loads(
        Path("tests/fixtures/scansion_sonnet.json").read_text())


@pytest.fixture
def haiku():
    return json.loads(Path("tests/fixtures/haiku.json").read_text())


@pytest.fixture
def pos_output():
    return json.loads(
        Path("tests/fixtures/pos_output.json").read_text())


class TokenMock(mock.MagicMock):
    _ = property(lambda self: mock.Mock(has_tmesis=self.has_tmesis,
                                        line=self.line))

    def __isinstance__(self, token):  # noqa
        return True

    @staticmethod
    def is_ancestor(token):  # noqa
        return True

    @staticmethod
    def nbor():  # noqa
        return TokenMock()


def test_get_scansion_spacy_doc(monkeypatch):
    token = TokenMock(text="Agüita", i=0, is_punct=False, has_tmesis=False,
                      line=1, pos_="NOUN")

    def mockreturn(lang=None):
        return lambda _: [
            token
        ]

    monkeypatch.setattr(rantanplan.core, 'load_pipeline', mockreturn)
    enjambment = get_scansion(token)
    assert enjambment == [
        {'tokens': [
            {'word': [
                {'syllable': 'A', 'is_stressed': False},
                {'syllable': 'güi', 'is_stressed': True},
                {'syllable': 'ta', 'is_stressed': False,
                 'is_word_end': True}
            ],
                'stress_position': -2}
        ],
            'phonological_groups': [
                {'syllable': 'A', 'is_stressed': False},
                {'syllable': 'güi', 'is_stressed': True},
                {'syllable': 'ta', 'is_stressed': False,
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '-+-', 'type': 'pattern', 'length': 3}
        }
    ]


def test_have_prosodic_liaison():
    first_syllable = {'syllable': 'ca', 'is_stressed': True}
    second_syllable = {'syllable': 'en', 'is_stressed': False}
    assert have_prosodic_liaison(first_syllable, second_syllable) is True


def test_have_prosodic_liaison_second_syllable_y_with_vowel():
    first_syllable = {'syllable': 'ca', 'is_stressed': True}
    second_syllable = {'syllable': 'yen', 'is_stressed': False}
    assert have_prosodic_liaison(first_syllable, second_syllable) is False


def test_syllabify_exceptions_en():
    word = "entender"
    output = ['en', 'ten', 'der']
    assert syllabify(word)[0] == output


def test_syllabify_exceptions_en_2():
    word = "desentender"
    output = ['de', 'sen', 'ten', 'der']
    assert syllabify(word)[0] == output


def test_syllabify_exceptions_en_3():
    word = "desenmarañados"
    output = ['de', 'sen', 'ma', 'ra', 'ña', 'dos']
    assert syllabify(word)[0] == output


def test_syllabify_exceptions_prefix_des_consonant():
    word = "destapar"
    output = ['des', 'ta', 'par']
    assert syllabify(word)[0] == output


def test_syllabify_exceptions_prefix_sin_consonant():
    word = "sinhueso"
    output = ['sin', 'hue', 'so']
    assert syllabify(word)[0] == output


def test_syllabify_tl():
    word = "atlante"
    output = ['a', 'tlan', 'te']
    assert syllabify(word)[0] == output


def test_syllabify_group_1():
    word = "antihumano"
    output = ['an', 'tihu', 'ma', 'no']
    assert syllabify(word)[0] == output


def test_syllabify_group_2():
    word = "entrehierro"
    output = ['en', 'tre', 'hie', 'rro']
    assert syllabify(word)[0] == output


def test_syllabify_group_3():
    word = "yihad"
    output = ['yi', 'had']
    assert syllabify(word)[0] == output


def test_syllabify_group_4():
    word = "coche"
    output = ['co', 'che']
    assert syllabify(word)[0] == output


def test_syllabify_group_4_rl():
    word = "abarloar"
    output = ['a', 'bar', 'lo', 'ar']
    assert syllabify(word)[0] == output


def test_syllabify_group_5():
    word = "checo"
    output = ['che', 'co']
    assert syllabify(word)[0] == output


def test_syllabify_group_6():
    word = "año"
    output = ['a', 'ño']
    assert syllabify(word)[0] == output


def test_syllabify_group_7():
    word = "desvirtúe"
    output = ['des', 'vir', 'tú', 'e']
    assert syllabify(word)[0] == output


def test_syllabify_umlaut_u_e():
    word = "güegüecho"
    output = ['güe', 'güe', 'cho']
    assert syllabify(word)[0] == output


def test_syllabify_umlaut_hyatus_with_consonant_1():
    word = "insacïable"
    output = ['in', 'sa', 'cï', 'a', 'ble']
    assert syllabify(word)[0] == output


def test_syllabify_umlaut_hyatus_with_consonant_2():
    word = "ruïdo"
    output = ['ru', 'ï', 'do']
    assert syllabify(word)[0] == output


def test_syllabify_umlaut_hyatus_with_vowel():
    word = "ruëa"
    output = ['ru', 'ë', 'a']
    assert syllabify(word)[0] == output


def test_syllabify_umlaut_u_i():
    word = "güito"
    output = ['güi', 'to']
    assert syllabify(word)[0] == output


def test_syllabify_umlaut_u_i_tilde():
    word = "agüío"
    output = ['a', 'güí', 'o']
    assert syllabify(word)[0] == output


def test_syllabify_alternatives():
    word = "arcaizabas"
    output = (['ar', 'ca', 'i', 'za', 'bas'], (1, 2))
    assert syllabify(word, alternative_syllabification=True) == output


def test_syllabify_alternatives_2():
    word = "puntual"
    output = (['pun', 'tu', 'al'], (1, 2))
    assert syllabify(word, alternative_syllabification=True) == output


def test_get_orthographic_accent():
    syllable_list = ['plá', 'ta', 'no']
    output = 0
    assert get_orthographic_accent(syllable_list) == output


def test_apply_exception_rules_post_hiatus_first_vowel():
    syllabified_word = "hüe-co"
    output = "hü-e-co"
    assert apply_exception_rules_post(syllabified_word) == output


def test_apply_exception_rules_post_consonant_cluster():
    syllabified_word = "c-ne-o-rá-ce-a"
    output = "cne-o-rá-ce-a"
    assert apply_exception_rules_post(syllabified_word) == output


def test_apply_exception_rules_post_raising_diphthong():
    syllabified_word = "a-hi-ja-dor"
    output = "ahi-ja-dor"
    assert apply_exception_rules_post(syllabified_word) == output


def test_apply_exception_rules_post_lowering_diphthong():
    syllabified_word = "bu-hi-ti-ho"
    output = "buhi-tiho"
    assert apply_exception_rules_post(syllabified_word) == output


def test_get_orthographic_accent_with_no_tilde():
    syllable_list = ['pla', 'ta', 'ne', 'ro']
    assert get_orthographic_accent(syllable_list) is None


def test_is_paroxytone():
    syllable_list = ['pla', 'ta', 'ne', 'ro']
    assert is_paroxytone(syllable_list) is True


def test_is_paroxytone_with_tilde():
    syllable_list = ['cés', 'ped']
    assert is_paroxytone(syllable_list) is False


def test_is_paroxytone_with_proparoxytone():
    syllable_list = ['es', 'drú', 'ju', 'la']
    assert is_paroxytone(syllable_list) is False


def test_is_paroxytone_with_oxytone_with_tilde():
    syllable_list = ['a', 'com', 'pa', 'ñó']
    assert is_paroxytone(syllable_list) is False


def test_is_paroxytone_with_oxytone_no_tilde():
    syllable_list = ['tam', 'bor']
    assert is_paroxytone(syllable_list) is False


def test_get_word_stress():
    word = "plátano"
    pos = "NOUN"
    tag = {'Gender': 'Masc', 'Number': 'Sing'}
    output = {
        'word': [
            {'syllable': 'plá', 'is_stressed': True},
            {'syllable': 'ta', 'is_stressed': False},
            {'syllable': 'no', 'is_stressed': False}
        ], 'stress_position': -3}
    assert get_word_stress(word, pos, tag) == output


def test_get_word_stress_unstressed():
    word = "platano"
    pos = "DET"
    tag = {'Gender': 'Masc', 'Number': 'Sing'}
    output = {
        'word': [
            {'syllable': 'pla', 'is_stressed': False},
            {'syllable': 'ta', 'is_stressed': False},
            {'syllable': 'no', 'is_stressed': False}
        ], 'stress_position': 0}
    assert get_word_stress(word, pos, tag) == output


def test_get_word_stress_stressed_monosyllables_without_tilde():
    word = "yo"
    pos = "PRON"
    tag = {'Case': 'Nom', 'Number': 'Sing', 'Person': '1', 'PronType': 'Prs'}
    output = {
        'word': [
            {'syllable': 'yo', 'is_stressed': True}
        ],
        'stress_position': -1}
    assert get_word_stress(word, pos, tag) == output


def test_get_word_stress_unstressed_monosyllables_without_tilde():
    word = "mi"
    pos = "DET"
    tag = {'Number': 'Sing', 'Number[psor]': 'Sing', 'Person': '1',
           'Poss': 'Yes', 'PronType': 'Prs'}
    output = {
        'word': [
            {'syllable': 'mi', 'is_stressed': False}
        ],
        'stress_position': 0}
    assert get_word_stress(word, pos, tag) == output


def test_get_word_stress_no_tilde():
    word = "campo"
    pos = "NOUN"
    tag = {'Gender': 'Masc', 'Number': 'Sing'}
    output = {
        'word': [
            {'syllable': 'cam', 'is_stressed': True},
            {'syllable': 'po', 'is_stressed': False}
        ],
        'stress_position': -2}
    assert get_word_stress(word, pos, tag) == output


def test_get_word_stress_oxytone():
    word = "tambor"
    pos = "NOUN"
    tag = {'Gender': 'Fem', 'Number': 'Sing'}
    output = {
        'word': [
            {'syllable': 'tam', 'is_stressed': False},
            {'syllable': 'bor', 'is_stressed': True}
        ],
        'stress_position': -1}
    assert get_word_stress(word, pos, tag) == output


def test_get_words():
    word = nlp('físico-químico')
    output = [
        {
            'word': [
                {'syllable': 'fí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'co', 'is_stressed': False}
            ], 'stress_position': -3, 'pos': 'NOUN'}, {'symbol': '-'}, {
            'word': [
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'mi', 'is_stressed': False},
                {'syllable': 'co', 'is_stressed': False}
            ], 'stress_position': -3, 'pos': 'ADJ'}
    ]
    assert get_words(word) == output


def test_get_scansion_spacy_doc_text():
    text = nlp("patata")
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'pa', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': True},
                {'syllable': 'ta', 'is_stressed': False, 'is_word_end': True}
            ],
                'stress_position': -2}
        ],
            'phonological_groups': [
                {'syllable': 'pa', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': True},
                {'syllable': 'ta', 'is_stressed': False,
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '-+-', 'type': 'pattern', 'length': 3}
        }
    ]
    assert get_scansion(text) == output
    assert _get_scansion(text) == output


def test_get_scansion_rhyme_analysis_sonnet(sonnet):
    text = """Cruel amor, ¿tan fieras sinrazones
    tras tanta confusión, tras pena tanta?
    ¿De qué sirve la argolla a la garganta
    a quién jamás huyó de sus prisiones?
    ¿Hierro por premio das a mis pasiones?
    Dueño cruel, tu sinrazón espanta,
    el castigo a la pena se adelanta
    y cuando sirvo bien hierros me pones.
    ¡Gentil laurel, amor; buenos despojos!
    Y en un sujeto a tus mudanzas firme
    hierro, virote, lágrimas y enojos.
    Mas pienso que has querido persuadirme
    que trayendo los hierros a los ojos
    no pueda de la causa arrepentirme."""
    assert get_scansion(text, rhyme_analysis=True) == sonnet
    assert _get_scansion(text, rhyme_analysis=True) == sonnet


def test_get_scansion_stanzas():
    poem = """Que se caiga la torre
    de Valladolid
    como a mí no me coja,
    ¿qué se me da a mí?

    -*-

    ¡Cuán solitaria la nación que un día
    poblara inmensa gente,
    la nación cuyo imperio se extendía
    del Ocaso al Oriente!"""
    # Note that -*- is actually 4 spaces into the line
    split_on = r"[\s]+-\*-[\s]+"
    seguidilla, cuarteto_lira = get_scansion(
        poem, rhyme_analysis=True, split_stanzas_on=split_on
    )
    assert seguidilla[0]["structure"] == "seguidilla"
    assert cuarteto_lira[0]["structure"] == "cuarteto_lira"


def test_get_scansion_structures_length():
    text = "casa azul"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'ca', 'is_stressed': True},
                {'syllable': 'sa',
                 'is_stressed': False,
                 'has_synalepha': True,
                 'is_word_end': True}],
                'stress_position': -2},
            {'word': [
                {'syllable': 'a', 'is_stressed': False},
                {'syllable': 'zul', 'is_stressed': True,
                 'is_word_end': True}
            ],
                'stress_position': -1}
        ],
            'phonological_groups': [
                {'syllable': 'ca', 'is_stressed': True},
                {'syllable': 'sa',
                 'is_stressed': False,
                 'has_synalepha': False,
                 'is_word_end': True},
                {'syllable': 'a', 'is_stressed': False},
                {'syllable': 'zul', 'is_stressed': True,
                 'is_word_end': True}],
            'rhythm': {'stress': '+--+-', 'type': 'pattern', 'length': 5}
        }
    ]
    assert get_scansion(text, rhythmical_lengths=[5]) == output
    assert _get_scansion(text, rhythmical_lengths=[5]) == output


def test_get_scansion_rhyme_analysis_haiku_no_rhyme(haiku):
    text = """Noche sin luna.
    La tempestad estruja
    los viejos cedros."""
    assert get_scansion(text, rhyme_analysis=True) == haiku
    assert _get_scansion(text, rhyme_analysis=True) == haiku


def test_get_scansion(scansion_sonnet):
    text = """Siempre en octubre comenzaba el año.
    ¡Y cuántas veces esa luz de otoño
    me recordó a Fray Luis:
    «Ya el tiempo nos convida
    A los estudios nobles...»!"""
    assert get_scansion(
        text, rhythm_format="pattern", rhyme_analysis=True,
        always_return_rhyme=True) == scansion_sonnet
    assert _get_scansion(
        text, rhythm_format="pattern", rhyme_analysis=True,
        always_return_rhyme=True) == scansion_sonnet


def test_get_scansion_stressed_last_syllable():
    text = "altavoz"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'al', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': False},
                {'syllable': 'voz', 'is_stressed': True,
                 'is_word_end': True}
            ],
                'stress_position': -1}
        ],
            'phonological_groups': [
                {'syllable': 'al', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': False},
                {'syllable': 'voz', 'is_stressed': True,
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '--+-', 'type': 'pattern', 'length': 4}}
    ]
    assert get_scansion(text, rhythm_format="pattern") == output
    assert _get_scansion(text, rhythm_format="pattern") == output


def test_get_scansion_stressed_last_syllable_index_metrical_pattern():
    text = "altavoz"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'al', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': False},
                {'syllable': 'voz', 'is_stressed': True,
                 'is_word_end': True}
            ],
                'stress_position': -1}
        ],
            'phonological_groups': [
                {'syllable': 'al', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': False},
                {'syllable': 'voz', 'is_stressed': True,
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '3', 'type': 'indexed', 'length': 4}}
    ]
    assert get_scansion(text, rhythm_format="indexed") == output
    assert _get_scansion(text, rhythm_format="indexed") == output


def test_get_scansion_sinaeresis():
    text = "héroe"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'hé', 'is_stressed': True},
                {'syllable': 'ro', 'is_stressed': False,
                 'has_sinaeresis': True},
                {'syllable': 'e', 'is_stressed': False,
                 'is_word_end': True}
            ],
                'stress_position': -3}
        ],
            'phonological_groups': [
                {'syllable': 'hé', 'is_stressed': True},
                {'syllable': 'roe',
                 'is_stressed': False,
                 'sinaeresis_index': [1],
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '+-', 'type': 'pattern', 'length': 2}}
    ]
    assert get_scansion(text, rhythm_format="pattern") == output
    assert _get_scansion(text, rhythm_format="pattern") == output


def test_get_scansion_affixes():
    text = "antiquísimo"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mo', 'is_stressed': False,
                 'is_word_end': True}
            ],
                'stress_position': -3}
        ],
            'phonological_groups': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mo', 'is_stressed': False,
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '--+-', 'type': 'pattern', 'length': 4}}
    ]
    assert get_scansion(text, rhythm_format="pattern") == output
    assert _get_scansion(text, rhythm_format="pattern") == output


def test_get_scansion_sinaeresis_synalepha_affixes():
    text = "antiquísimo héroe"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mo',
                 'is_stressed': False,
                 'has_synalepha': True,
                 'is_word_end': True}
            ],
                'stress_position': -3},
            {'word': [
                {'syllable': 'hé', 'is_stressed': True},
                {'syllable': 'ro', 'is_stressed': False,
                 'has_sinaeresis': True},
                {'syllable': 'e', 'is_stressed': False,
                 'is_word_end': True}
            ],
                'stress_position': -3}
        ],
            'phonological_groups': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mohé', 'is_stressed': True,
                 'synalepha_index': [1]},
                {'syllable': 'roe',
                 'is_stressed': False,
                 'sinaeresis_index': [1],
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '--+-+-', 'type': 'pattern', 'length': 6}}
    ]
    assert get_scansion(text, rhythm_format="pattern") == output
    assert _get_scansion(text, rhythm_format="pattern") == output


def test_get_scansion_sinaeresis_synalepha_affixes_index_metrical_pattern():
    text = "antiquísimo héroe"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mo',
                 'is_stressed': False,
                 'has_synalepha': True,
                 'is_word_end': True}
            ],
                'stress_position': -3},
            {'word': [
                {'syllable': 'hé', 'is_stressed': True},
                {'syllable': 'ro', 'is_stressed': False,
                 'has_sinaeresis': True},
                {'syllable': 'e', 'is_stressed': False,
                 'is_word_end': True}
            ],
                'stress_position': -3}
        ],
            'phonological_groups': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mohé', 'is_stressed': True,
                 'synalepha_index': [1]},
                {'syllable': 'roe',
                 'is_stressed': False,
                 'sinaeresis_index': [1],
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '3-5', 'type': 'indexed', 'length': 6}}
    ]
    assert get_scansion(text, rhythm_format="indexed") == output
    assert _get_scansion(text, rhythm_format="indexed") == output


def test_get_scansion_sinaeresis_synalepha_affixes_binary_metrical_pattern():
    text = "antiquísimo héroe"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mo',
                 'is_stressed': False,
                 'has_synalepha': True,
                 'is_word_end': True}
            ],
                'stress_position': -3},
            {'word': [
                {'syllable': 'hé', 'is_stressed': True},
                {'syllable': 'ro', 'is_stressed': False,
                 'has_sinaeresis': True},
                {'syllable': 'e', 'is_stressed': False,
                 'is_word_end': True}
            ],
                'stress_position': -3}
        ],
            'phonological_groups': [
                {'syllable': 'an', 'is_stressed': False},
                {'syllable': 'ti', 'is_stressed': False},
                {'syllable': 'quí', 'is_stressed': True},
                {'syllable': 'si', 'is_stressed': False},
                {'syllable': 'mohé', 'is_stressed': True,
                 'synalepha_index': [1]},
                {'syllable': 'roe',
                 'is_stressed': False,
                 'sinaeresis_index': [1],
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '001010', 'type': 'binary', 'length': 6}}
    ]
    assert get_scansion(text, rhythm_format="binary") == output
    assert _get_scansion(text, rhythm_format="binary") == output


def test_spacy_tag_to_dict():
    tag = "DET__Number=Sing|Number[psor]=Sing|Person=1|Poss=Yes|PronType=Prs"
    output = {'DET__Number': 'Sing', 'Number[psor]': 'Sing', 'Person': '1',
              'Poss': 'Yes', 'PronType': 'Prs'}
    assert spacy_tag_to_dict(tag) == output


def test_spacy_tag_to_dict_no_tags():
    tag = "DET___"
    assert spacy_tag_to_dict(tag) == {}


def test_get_syllables_word_end():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': True,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'has_synalepha': True,
         'is_word_end': True},
        {'syllable': 'un', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    words = [
        {'word': [
            {'syllable': 'tu', 'is_stressed': False},
            {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': True}
        ],
            'stress_position': -1},
        {'word': [
            {'syllable': 'a', 'is_stressed': False, 'has_synalepha': True}
        ],
            'stress_position': 0},
        {'word': [
            {'syllable': 'un', 'is_stressed': True}
        ],
            'stress_position': -1},
        {'word': [
            {'syllable': 'Du', 'is_stressed': True},
            {'syllable': 'que', 'is_stressed': False}
        ],
            'stress_position': -2}
    ]
    assert get_syllables_word_end(words) == output


def test_get_phonological_groups_synalephas():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'llóaun', 'is_stressed': True,
         'synalepha_index': [2, 3], 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': True,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'has_synalepha': True,
         'is_word_end': True},
        {'syllable': 'un', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(words) == output


def test_get_phonological_groups_synalepha():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'llóaun', 'is_stressed': True,
         'synalepha_index': [2]},
        {'syllable': 'que', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'yo', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': True,
         'is_word_end': True},
        {'syllable': 'aun', 'is_stressed': False},
        {'syllable': 'que', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'yo', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(words) == output


def test_get_phonological_groups_no_synalepha():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(words) == output


def test_get_phonological_groups_sinaeresis():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': False,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Dua', 'is_stressed': True,
         'sinaeresis_index': [1]},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': False,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True, 'has_sinaeresis': True},
        {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(words, liaison_type="sinaeresis") == output


def test_get_phonological_groups_no_sinaeresis():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'is_word_end': True,
         'has_synalepha': False},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True, 'has_sinaeresis': False},
        {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': False,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True, 'has_sinaeresis': False},
        {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(words, liaison_type="sinaeresis") == output


def test_get_phonological_groups_no_synalepha_no_sinaeresis():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'is_word_end': True,
         'has_synalepha': False},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True, 'has_sinaeresis': False},
        {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': False,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True, 'has_sinaeresis': False},
        {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(
        get_phonological_groups(words, liaison_type="sinaeresis")
    ) == output


def test_get_phonological_groups_synalepha_sinaeresis():
    output = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'llóaun', 'is_stressed': True, 'is_word_end': True,
         'synalepha_index': [2, 3]},
        {'syllable': 'Dua', 'is_stressed': True, 'sinaeresis_index': [1]},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True}
    ]
    words = [
        {'syllable': 'tu', 'is_stressed': False},
        {'syllable': 'lló', 'is_stressed': True, 'has_synalepha': True,
         'is_word_end': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True,
         'has_synalepha': True},
        {'syllable': 'un', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'Du', 'is_stressed': True, 'has_sinaeresis': True},
        {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'que', 'is_stressed': False, 'is_word_end': True},
    ]
    assert get_phonological_groups(
        get_phonological_groups(words, liaison_type="sinaeresis")
    ) == output


def test_generate_phonological_groups(phonological_groups):
    tokens = nlp("el perro hace aguas")
    assert list(generate_phonological_groups(tokens)) == phonological_groups


def test_generate_liaison_positions_synalepha():
    syllables = [
        {'syllable': 'el', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'pe', 'is_stressed': True},
        {'syllable': 'rro', 'is_stressed': False, 'has_synalepha': True,
         'is_word_end': True}, {'syllable': 'ha', 'is_stressed': True},
        {'syllable': 'ce', 'is_stressed': False, 'has_synalepha': True,
         'is_word_end': True}, {'syllable': 'a', 'is_stressed': True},
        {'syllable': 'guas', 'is_stressed': False, 'is_word_end': True}
    ]
    output = [
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    assert list(
        generate_liaison_positions(syllables, liaison="synalepha")) == output


def test_generate_liaison_positions_sinaeresis():
    syllables = [
        {'syllable': 'ha', 'is_stressed': False},
        {'syllable': 'cí', 'is_stressed': True, 'has_sinaeresis': True},
        {'syllable': 'a', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'mu', 'is_stressed': True},
        {'syllable': 'chas', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'ca', 'is_stressed': False, 'has_sinaeresis': True},
        {'syllable': 'í', 'is_stressed': True},
        {'syllable': 'das', 'is_stressed': False, 'is_word_end': True}
    ]
    output = [
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    assert list(
        generate_liaison_positions(syllables, liaison="sinaeresis")) == output


def test_clean_phonological_groups():
    phonological_groups = [
        {'syllable': 'es', 'is_stressed': True},
        {'syllable': 'to', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'tie', 'is_stressed': True},
        {'syllable': 'ne', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'mu', 'is_stressed': True},
        {'syllable': 'cho', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'rit', 'is_stressed': True},
        {'syllable': 'mo', 'is_stressed': False,
         'is_word_end': True}
    ]
    liaison_positions = [0, 0, 0, 0, 0, 0, 0, 0]
    liaison_property = "has_sinaeresis"
    output = [
        {'syllable': 'es', 'is_stressed': True},
        {'syllable': 'to', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'tie', 'is_stressed': True},
        {'syllable': 'ne', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'mu', 'is_stressed': True},
        {'syllable': 'cho', 'is_stressed': False, 'is_word_end': True},
        {'syllable': 'rit', 'is_stressed': True},
        {'syllable': 'mo', 'is_stressed': False, 'is_word_end': True}
    ]
    assert clean_phonological_groups(
        phonological_groups, liaison_positions, liaison_property) == output


def test_get_stresses():
    phonological_groups = [
        {'syllable': 'es', 'is_stressed': True},
        {'syllable': 'to', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'tie', 'is_stressed': True},
        {'syllable': 'ne', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'mu', 'is_stressed': True},
        {'syllable': 'cho', 'is_stressed': False,
         'is_word_end': True},
        {'syllable': 'rit', 'is_stressed': True},
        {'syllable': 'mo', 'is_stressed': False,
         'is_word_end': True}
    ]
    output = [True, False, True, False, True, False, True, False]
    assert (get_stresses(phonological_groups) == output)


def test_get_rhythmical_pattern_proparoxytone():
    phonological_groups = [
        {'syllable': 'es', 'is_stressed': True},
        {'syllable': 'toes', 'is_stressed': True, 'synalepha_index': [1],
         'is_word_end': True}, {'syllable': 'mú', 'is_stressed': True},
        {'syllable': 'si', 'is_stressed': False},
        {'syllable': 'ca', 'is_stressed': False, 'is_word_end': True}
    ]
    output = {'stress': '+++-', 'type': 'pattern', 'length': 4}
    assert get_rhythmical_pattern(phonological_groups) == output


def test_get_rhythmical_pattern_paroxytone():
    phonological_groups = [
        {'syllable': 'es', 'is_stressed': True},
        {'syllable': 'taes', 'is_stressed': True, 'synalepha_index': [1],
         'is_word_end': True},
        {'syllable': 'mi', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'ca', 'is_stressed': True},
        {'syllable': 'sa', 'is_stressed': False, 'is_word_end': True}
    ]
    output = {'stress': '++++-', 'type': 'pattern', 'length': 5}
    assert get_rhythmical_pattern(phonological_groups) == output


def test_get_rhythmical_pattern_oxytone():
    phonological_groups = [
        {'syllable': 'es', 'is_stressed': True},
        {'syllable': 'tees', 'is_stressed': True, 'synalepha_index': [1],
         'is_word_end': True},
        {'syllable': 'mi', 'is_stressed': True, 'is_word_end': True},
        {'syllable': 'cha', 'is_stressed': False},
        {'syllable': 'lé', 'is_stressed': True, 'is_word_end': True}
    ]
    output = {'stress': '+++-+-', 'type': 'pattern', 'length': 6}
    assert get_rhythmical_pattern(phonological_groups) == output


def test_format_stress_pattern():
    stresses = (True, True, False, True)
    output = '++-+'
    assert (format_stress(stresses) == output)


def test_format_stress_indexed():
    stresses = (True, True, False, True)
    output = '1-2-4'
    assert (format_stress(stresses, rhythm_format="indexed") == output)


def test_format_stress_indexed_custom_separator():
    stresses = (True, True, False, True)
    output = '1,2,4'
    assert (format_stress(stresses, rhythm_format="indexed",
                          indexed_separator=",") == output)


def test_format_stress_binary():
    stresses = (True, True, False, True)
    output = '1101'
    assert (format_stress(stresses, rhythm_format="binary") == output)


def test_get_last_syllable():
    word = [
        {'word': [
            {'syllable': 'úl', 'is_stressed': True},
            {'syllable': 'ti', 'is_stressed': False},
            {'syllable': 'ma', 'is_stressed': False}
        ], 'stress_position': -3}
    ]
    output = {'syllable': 'ma', 'is_stressed': False}
    assert get_last_syllable(word) == output


def test_apply_exception_rules_des():
    word = "destituir"
    output = "des-tituir"
    assert apply_exception_rules(word) == output


def test_apply_exception_rules_sin():
    word = "sinhueso"
    output = "sin-hueso"
    assert apply_exception_rules(word) == output


def test_apply_exception_rules_consonan_group_dl():
    word = "adlativo"
    output = "ad-lativo"
    assert apply_exception_rules(word) == output


def test_apply_exception_rules_consonan_group_ll():
    word = "alhábega"
    output = "al-hábega"
    assert apply_exception_rules(word) == output


def test_apply_exception_rules_consonan_group():
    word = "aislable"
    output = "ais-lable"
    assert apply_exception_rules(word) == output


def test_apply_exception_rules_consonan_w_vowel():
    word = "kiwi"
    output = "ki-wi"
    assert apply_exception_rules(word) == output


def test_has_single_liaisons_false():
    liaisons = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    output = has_single_liaisons(liaisons)
    assert not output


def test_has_single_liaisons_true():
    liaisons = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    output = has_single_liaisons(liaisons)
    assert output


def test_remove_exact_length_matches():
    lines = [
        {'tokens': [
            {'word': [
                {'syllable': 'me',
                 'is_stressed': False,
                 'has_synalepha': True,
                 'is_word_end': True}],
                'stress_position': 0},
            {'word': [
                {'syllable': 'he', 'is_stressed': True, 'is_word_end': True}],
                'stress_position': -1}],
            'phonological_groups': [
                {'syllable': 'mehe',
                 'is_stressed': True,
                 'synalepha_index': [1],
                 'is_word_end': True}],
            'rhythm': {'stress': '+-',
                       'type': 'pattern',
                       'length': 2,
                       'length_range': {'min_length': 2, 'max_length': 3}}},
        {'tokens': [{'word': [
            {'syllable': 'i', 'is_stressed': True},
            {'syllable': 'do', 'is_stressed': False, 'is_word_end': True}],
            'stress_position': -2}],
            'phonological_groups': [
                {'syllable': 'i', 'is_stressed': True},
                {'syllable': 'do', 'is_stressed': False, 'is_word_end': True}],
            'rhythm': {'stress': '+-',
                       'type': 'pattern',
                       'length': 2,
                       'length_range': {'min_length': 2, 'max_length': 2}}}]
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'me',
                 'is_stressed': False,
                 'has_synalepha': True,
                 'is_word_end': True}],
                'stress_position': 0},
            {'word': [
                {'syllable': 'he', 'is_stressed': True, 'is_word_end': True}],
                'stress_position': -1}],
            'phonological_groups': [
                {'syllable': 'mehe',
                 'is_stressed': True,
                 'synalepha_index': [1],
                 'is_word_end': True}],
            'rhythm': {'stress': '+-',
                       'type': 'pattern',
                       'length': 2,
                       'length_range': {'min_length': 2, 'max_length': 3}}},
        {'tokens': [
            {'word': [
                {'syllable': 'i', 'is_stressed': True},
                {'syllable': 'do', 'is_stressed': False, 'is_word_end': True}],
                'stress_position': -2}],
            'phonological_groups': [
                {'syllable': 'i', 'is_stressed': True},
                {'syllable': 'do', 'is_stressed': False, 'is_word_end': True}],
            'rhythm': {'stress': '+-', 'type': 'pattern', 'length': 2}}]
    assert remove_exact_length_matches(lines) == output


def test_get_scansion_stress_last_word():
    text = "¡oh!"
    output = [
        {'tokens': [
            {'symbol': '¡'},
            {'word': [
                {'syllable': 'oh', 'is_stressed': True, 'is_word_end': True}],
                'stress_position': -1},
            {'symbol': '!'}],
            'phonological_groups': [{'syllable': 'oh',
                                     'is_stressed': True,
                                     'is_word_end': True}],
            'rhythm': {'stress': '+-', 'type': 'pattern', 'length': 2}}]
    assert get_scansion(text) == output


def test_get_word_stress_last_word_false():
    text = nlp("¡oh!")
    output = [
        {'symbol': '¡'},
        {'word': [
            {'syllable': 'oh', 'is_stressed': False}],
            'stress_position': 0,
            'pos': 'INTJ'},
        {'symbol': '!'}]
    assert get_words(text) == output


def test_get_scansion_pos_output():
    text = "patata"
    output = [
        {'tokens': [
            {'word': [
                {'syllable': 'pa', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': True},
                {'syllable': 'ta', 'is_stressed': False, 'is_word_end': True}
            ],
                'stress_position': -2,
                'pos': 'NOUN'}
        ],
            'phonological_groups': [
                {'syllable': 'pa', 'is_stressed': False},
                {'syllable': 'ta', 'is_stressed': True},
                {'syllable': 'ta', 'is_stressed': False,
                 'is_word_end': True}
            ],
            'rhythm': {'stress': '-+-', 'type': 'pattern', 'length': 3}
        }
    ]
    assert get_scansion(text, pos_output=True) == output
    assert _get_scansion(text,  pos_output=True) == output


def test_get_scansion_pos_output_affixes():
    text = "dímelo"
    output = [{
        'tokens': [
            {'word': [{'syllable': 'dí', 'is_stressed': False},
                      {'syllable': 'me', 'is_stressed': False},
                      {'syllable': 'lo',
                       'is_stressed': True,
                       'is_word_end': True}],
             'stress_position': -1,
             'pos': 'VERB+PRON+PRON'}],
        'phonological_groups': [{'syllable': 'dí', 'is_stressed': False},
                                {'syllable': 'me', 'is_stressed': False},
                                {'syllable': 'lo',
                                 'is_stressed': True,
                                 'is_word_end': True}],
        'rhythm': {'stress': '--+-', 'type': 'pattern', 'length': 4}
    }]
    assert get_scansion(text, pos_output=True) == output
    assert _get_scansion(text, pos_output=True) == output


def test_get_scansion_pos_output_affixes_mente():
    text = "dímelo mismamente"
    output = [
        {'tokens': [{'word': [{'syllable': 'dí', 'is_stressed': True},
                              {'syllable': 'me', 'is_stressed': False},
                              {'syllable': 'lo',
                               'is_stressed': False,
                               'is_word_end': True}],
                     'stress_position': -3,
                     'pos': 'VERB+PRON+DET'},
                    {'word': [{'syllable': 'mis', 'is_stressed': True},
                              {'syllable': 'ma', 'is_stressed': False},
                              {'syllable': 'men', 'is_stressed': True},
                              {'syllable': 'te',
                               'is_stressed': False,
                               'is_word_end': True}],
                     'stress_position': -4,
                     'secondary_stress_positions': [-2],
                     'pos': 'ADV'}],
         'phonological_groups': [{'syllable': 'dí', 'is_stressed': True},
                                 {'syllable': 'me', 'is_stressed': False},
                                 {'syllable': 'lo',
                                  'is_stressed': False,
                                  'is_word_end': True},
                                 {'syllable': 'mis', 'is_stressed': True},
                                 {'syllable': 'ma', 'is_stressed': False},
                                 {'syllable': 'men', 'is_stressed': True},
                                 {'syllable': 'te',
                                  'is_stressed': False,
                                  'is_word_end': True}],
         'rhythm': {'stress': '+--+-+-', 'type': 'pattern', 'length': 7}}
    ]
    assert get_scansion(text, pos_output=True) == output
    assert _get_scansion(text, pos_output=True) == output


def test_get_scansion_sinaeresis_synalepha_affixes_pos_output():
    text = "antiquísimo héroe"
    output = [
        {'tokens': [{'word': [{'syllable': 'an', 'is_stressed': False},
                              {'syllable': 'ti', 'is_stressed': False},
                              {'syllable': 'quí', 'is_stressed': True},
                              {'syllable': 'si', 'is_stressed': False},
                              {'syllable': 'mo',
                               'is_stressed': False,
                               'has_synalepha': True,
                               'is_word_end': True}],
                     'stress_position': -3,
                     'pos': 'DET'},
                    {'word': [{'syllable': 'hé', 'is_stressed': True},
                              {'syllable': 'ro',
                               'is_stressed': False,
                               'has_sinaeresis': True},
                              {'syllable': 'e',
                               'is_stressed': False,
                               'is_word_end': True}],
                     'stress_position': -3,
                     'pos': 'NOUN'}],
         'phonological_groups': [{'syllable': 'an', 'is_stressed': False},
                                 {'syllable': 'ti', 'is_stressed': False},
                                 {'syllable': 'quí', 'is_stressed': True},
                                 {'syllable': 'si', 'is_stressed': False},
                                 {'syllable': 'mohé',
                                  'is_stressed': True,
                                  'synalepha_index': [1]},
                                 {'syllable': 'roe',
                                  'is_stressed': False,
                                  'sinaeresis_index': [1],
                                  'is_word_end': True}],
         'rhythm': {'stress': '--+-+-', 'type': 'pattern', 'length': 6}}
    ]
    assert get_scansion(text, pos_output=True) == output
    assert _get_scansion(text, pos_output=True) == output


def test_get_scansion_pos_output_mente(pos_output):
    text = """Díselo al héroe
    que rápidamente vivió"""
    assert get_scansion(text, pos_output=True) == pos_output
    assert _get_scansion(text,  pos_output=True) == pos_output


def test_set_stress_exceptions_paroxytone_with_clitic():
    word = {'word': [
        {'syllable': 'dí', 'is_stressed': True},
        {'syllable': 'me', 'is_stressed': False},
        {'syllable': 'lo', 'is_stressed': False, 'is_word_end': True}],
        'stress_position': -3, 'pos': 'VERB+PRON+PRON'}
    output = {'word': [
        {'syllable': 'dí', 'is_stressed': False},
        {'syllable': 'me', 'is_stressed': False},
        {'syllable': 'lo', 'is_stressed': True, 'is_word_end': True}],
        'stress_position': -3, 'pos': 'VERB+PRON+PRON'}
    assert set_stress_exceptions(word) == output


def test_set_stress_exceptions_proparoxytone():
    word = {'word': [
        {'syllable': 'có', 'is_stressed': True},
        {'syllable': 'ge', 'is_stressed': False},
        {'syllable': 'me', 'is_stressed': False},
        {'syllable': 'lo', 'is_stressed': False,
         'is_word_end': True}], 'stress_position': -4,
        'pos': 'VERB+PRON+PRON'}
    output = {'word': [
        {'syllable': 'có', 'is_stressed': False},
        {'syllable': 'ge', 'is_stressed': False},
        {'syllable': 'me', 'is_stressed': False},
        {'syllable': 'lo', 'is_stressed': True, 'is_word_end': True}],
        'stress_position': -4, 'pos': 'VERB+PRON+PRON'}
    assert set_stress_exceptions(word) == output


def test_set_stress_exceptions_proparoxytone_adverb():
    word = {'word': [
        {'syllable': 'rá', 'is_stressed': True},
        {'syllable': 'pi', 'is_stressed': False},
        {'syllable': 'da', 'is_stressed': False},
        {'syllable': 'men', 'is_stressed': True},
        {'syllable': 'te', 'is_stressed': False,
         'is_word_end': True}],
        'stress_position': -5,
        'secondary_stress_positions': [-2],
        'pos': 'ADV'}
    output = word
    assert set_stress_exceptions(word) == output
