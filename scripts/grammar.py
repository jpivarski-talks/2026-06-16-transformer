"""
Shared Lark grammar and validator for the toy English sentence dataset.

Vocabulary:
  - Articles:          THE, A, AN
  - Prepositions:      IN, UNDER, ABOVE
  - Entity nouns:      CAT, MOUSE, ROBOT, ALLIGATOR, ANT
  - Place nouns:       HOUSE, YARD, STREET, AIRPORT, OCEAN
  - Intransitive verbs: SMILES, BURPS, SLEEPS, EXPLODES
  - Transitive verbs:  CHASES, EATS, LOVES, HATES
  - Padding:           <BLANK>

Article rules:
  THE precedes any noun.
  A precedes consonant-starting nouns (CAT MOUSE ROBOT HOUSE YARD STREET).
  AN precedes vowel-starting nouns (ALLIGATOR ANT AIRPORT OCEAN).

Valid sentence forms (with optional prepositional phrase):
  subj intrans
  subj trans obj
  pp subj intrans
  subj intrans pp
  subj pp intrans
  pp subj trans obj
  subj trans obj pp
  subj pp trans obj
"""

from lark import Lark

ENTITY_CONS  = ["CAT", "MOUSE", "ROBOT"]
ENTITY_VOWEL = ["ALLIGATOR", "ANT"]
PLACE_CONS   = ["HOUSE", "YARD", "STREET"]
PLACE_VOWEL  = ["AIRPORT", "OCEAN"]
INTRANS      = ["SMILES", "BURPS", "SLEEPS", "EXPLODES"]
TRANS        = ["CHASES", "EATS", "LOVES", "HATES"]
PREPS        = ["IN", "UNDER", "ABOVE"]

BLANK = "<BLANK>"

ALL_TOKENS = (
    ["THE", "A", "AN"]
    + PREPS
    + ENTITY_CONS + ENTITY_VOWEL
    + PLACE_CONS  + PLACE_VOWEL
    + INTRANS + TRANS
    + [BLANK]
)
TOKEN_TO_IDX = {t: i for i, t in enumerate(ALL_TOKENS)}
VOCAB_SIZE   = len(ALL_TOKENS)  # 25
SEQ_LEN      = 8


def _alt(*words):
    return " | ".join(f'"{w}"' for w in words)


GRAMMAR = f"""
    sentence: subj intrans
            | subj trans obj
            | pp subj intrans
            | subj intrans pp
            | subj pp intrans
            | pp subj trans obj
            | subj trans obj pp
            | subj pp trans obj

    subj: "THE" entity | "A" entity_cons | "AN" entity_vowel
    obj:  "THE" entity | "A" entity_cons | "AN" entity_vowel
    pp:   prep "THE" place | prep "A" place_cons | prep "AN" place_vowel

    entity:       entity_cons | entity_vowel
    entity_cons:  {_alt(*ENTITY_CONS)}
    entity_vowel: {_alt(*ENTITY_VOWEL)}
    place:        place_cons | place_vowel
    place_cons:   {_alt(*PLACE_CONS)}
    place_vowel:  {_alt(*PLACE_VOWEL)}

    prep:   {_alt(*PREPS)}
    intrans:{_alt(*INTRANS)}
    trans:  {_alt(*TRANS)}

    %ignore " "
"""

_parser = Lark(GRAMMAR, start="sentence", parser="lalr")


def is_valid(tokens_8):
    """Return True if an 8-token sequence is a valid padded sentence."""
    # BLANKs must be contiguous at the end
    hit_blank = False
    for t in tokens_8:
        if t == BLANK:
            hit_blank = True
        elif hit_blank:
            return False
    core = [t for t in tokens_8 if t != BLANK]
    if not core:
        return False
    try:
        _parser.parse(" ".join(core))
        return True
    except Exception:
        return False


def all_valid_sentences():
    """Return a list of all valid 8-token padded sentences."""
    sentences = []

    def make_subj(art, noun):
        return [art, noun]

    def make_pp(prep, art, noun):
        return [prep, art, noun]

    # All valid NPs (article, noun) obeying a/an/the rules
    def entity_nps():
        nps = []
        for n in ENTITY_CONS + ENTITY_VOWEL:
            nps.append(["THE", n])
        for n in ENTITY_CONS:
            nps.append(["A", n])
        for n in ENTITY_VOWEL:
            nps.append(["AN", n])
        return nps

    def place_nps():
        nps = []
        for n in PLACE_CONS + PLACE_VOWEL:
            nps.append(["THE", n])
        for n in PLACE_CONS:
            nps.append(["A", n])
        for n in PLACE_VOWEL:
            nps.append(["AN", n])
        return nps

    def pps():
        phrases = []
        for prep in PREPS:
            for np in place_nps():
                phrases.append([prep] + np)
        return phrases

    enps = entity_nps()
    ppl  = pps()
    pad  = lambda seq: seq + [BLANK] * (SEQ_LEN - len(seq))

    for subj in enps:
        for v in INTRANS:
            core = subj + [v]                   # length 3
            sentences.append(pad(core))
            for pp in ppl:
                sentences.append(pad(pp + core))           # pp subj intrans, len 6
                sentences.append(pad(core + pp))           # subj intrans pp, len 6
                sentences.append(pad(subj + pp + [v]))     # subj pp intrans, len 6

        for v in TRANS:
            for obj in enps:
                core = subj + [v] + obj             # length 5
                sentences.append(pad(core))
                for pp in ppl:
                    sentences.append(pad(pp + core))            # pp subj trans obj, len 8
                    sentences.append(pad(core + pp))            # subj trans obj pp, len 8
                    sentences.append(pad(subj + pp + [v] + obj))# subj pp trans obj, len 8

    return sentences
