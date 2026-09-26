STRUCTURES = {1: "lv_endo", 2: "myocardium", 3: "left_atrium"}
CANONICAL = {
    1: "the left ventricular endocardium",
    2: "the myocardium",
    3: "the left atrium",
}
TRAIN_BANK = {
    1: [
        "the left ventricular endocardium",
        "LV endocardial border",
        "the left ventricle endocardium",
        "endocardial surface of the left ventricle",
        "the LV cavity blood pool border",
        "the endocardial wall of the left ventricle",
        "left ventricular endocardial contour",
    ],
    2: [
        "the myocardium",
        "the LV myocardium",
        "myocardial wall",
        "the left ventricular myocardium",
        "the myocardium of the left ventricle",
        "LV myocardial tissue",
        "left ventricle myocardial wall",
    ],
    3: [
        "the left atrium",
        "LA chamber",
        "left atrial cavity",
        "the left atrial blood pool",
        "the left atrium appendage region",
        "atrium of the left ventricle side",
        "the left atrial cavity boundary",
    ],
}
HELDOUT = {
    1: [
        "ventricular endocardial surface",
        "the left ventricle endocardial trace",
        "the LV endocardial line",
    ],
    2: [
        "ventricular myocardial shell",
        "the cardiac muscle of the left ventricle",
        "wall of the LV myocardium",
    ],
    3: [
        "left atrial chamber boundary",
        "the LA cavity contour",
        "the left atrial region",
    ],
}
TRAIN_BANK_PROMPTS = {prompt for prompts in TRAIN_BANK.values() for prompt in prompts}
HELDOUT_PROMPTS = {prompt for prompts in HELDOUT.values() for prompt in prompts}
for key in STRUCTURES:
    assert CANONICAL[key] in TRAIN_BANK[key], f"Canonical prompt missing for {key}"
    assert not set(TRAIN_BANK[key]) & set(HELDOUT[key]), f"Prompt overlap for structure {key}"
assert not TRAIN_BANK_PROMPTS & HELDOUT_PROMPTS, "Prompt overlap between training and held-out banks"
