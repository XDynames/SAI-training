import copy

BOUNDING_BOX_PADDING = 10
DINOLIGHT_NAMES_TO_CATEGORY_ID = {"Closed Stomata": 0, "Open Stomata": 1}
ARABIDOPSIS_NAMES_TO_CATEGORY_ID = copy.deepcopy(DINOLIGHT_NAMES_TO_CATEGORY_ID)
ARABIDOPSIS_NAMES_TO_CATEGORY_ID["Stomatal Pore"] = 2
BARLEY_NAMES_TO_CATEGORY_ID = copy.deepcopy(ARABIDOPSIS_NAMES_TO_CATEGORY_ID)
BARLEY_NAMES_TO_CATEGORY_ID["Subsidiary cells"] = 3
BARLEY_HUMAN_TEST_SAMPLES = {
    "5Mar 3",
    "5Mar 6",
    "5Mar 10",
    "10Dec 19",
    "10Dec 24",
    "10Dec 29",
    "10Dec 30",
    "18Mar 28",
    "18Mar 31",
    "18Mar 34",
    "18Mar 37",
    "25Feb 27",
    "25Feb 29",
    "25Feb 32",
    "25Feb 37",
}

ARABIDOPSIS_HUMAN_TEST_SAMPLES = {
    "B-t-1-1",
    "B-t-1-3",
    "B-t-1-5",
    "B-t-2-1",
    "B-t-2-3",
    "C-g-1-1",
    "C-g-1-3",
    "C-g-1-5",
    "C-g-1-7",
    "C-g-2-2",
    "C-g-2-4",
    "C-t-1-2",
    "C-t-1-4",
    "C-t-2-1",
    "C-t-2-4",
    "C-W-1-1",
    "C-W-1-3",
    "C-W-1-6",
    "C-W-2-2",
    "C-W-2-4",
}

HUMAN_TEST_SAMPLES = BARLEY_HUMAN_TEST_SAMPLES.union(ARABIDOPSIS_HUMAN_TEST_SAMPLES)
