"""
data/prompts.py — Structure vocabulary and prompt banks.

Three prompt sets:
  CANONICAL  : one fixed prompt per structure (the paper's default setting).
  TRAIN_BANK : paraphrases sampled during training when prompt_mode="paraphrase".
  HELDOUT    : paraphrases NEVER seen in training. Used only at evaluation to
               test whether the text pathway generalizes beyond memorizing three
               strings. If a model trained with fixed prompts collapses on
               HELDOUT while a paraphrase-trained model does not, that is direct
               evidence that language (not a class index) is doing work.

Structure ids follow the CAMUS label convention:
  1 = LV endocardium (cavity), 2 = LV myocardium, 3 = left atrium.
"""

STRUCTURES = {
    1: "lv_endo",
    2: "myocardium",
    3: "left_atrium",
}
STRUCTURE_IDS = sorted(STRUCTURES.keys())

CANONICAL = {
    1: "the left ventricular endocardium",
    2: "the myocardium",
    3: "the left atrium",
}

TRAIN_BANK = {
    1: [
        "the left ventricular endocardium",
        "the left ventricular cavity",
        "the left ventricle cavity",
        "the LV blood pool",
        "the endocardial border of the left ventricle",
        "the inner boundary of the left ventricle",
        "the LV cavity",
    ],
    2: [
        "the myocardium",
        "the left ventricular myocardium",
        "the LV wall",
        "the myocardial wall",
        "the heart muscle of the left ventricle",
        "the muscle surrounding the left ventricle",
        "the LV myocardium",
    ],
    3: [
        "the left atrium",
        "the LA",
        "the left atrial cavity",
        "the left atrial chamber",
        "the LA blood pool",
        "the atrium on the left side of the heart",
    ],
}

HELDOUT = {
    1: [
        "the lumen of the left ventricle",
        "the blood-filled chamber inside the left ventricular wall",
        "LV endocardial region",
    ],
    2: [
        "the muscular wall between the LV endocardium and epicardium",
        "left ventricular muscle tissue",
        "the LV myocardial tissue",
    ],
    3: [
        "the chamber receiving blood from the pulmonary veins",
        "left atrial lumen",
        "the LA chamber",
    ],
}

# Guard against accidental leakage between banks.
for _s in STRUCTURE_IDS:
    _overlap = set(TRAIN_BANK[_s]) & set(HELDOUT[_s])
    assert not _overlap, "HELDOUT prompts leaked into TRAIN_BANK: {}".format(_overlap)
    assert CANONICAL[_s] in TRAIN_BANK[_s]
