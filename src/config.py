"""Shared project configuration for EduLens."""

RANDOM_SEED = 42
DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95
SMALL_CONDITIONAL_SAMPLE_SIZE = 5

LEAKY_COLUMNS = ["G1", "G2", "G3"]

# Column roles are based only on the variable table in DATASET.md §3.
NUMERIC_COLUMNS = ["age", "absences", "G1", "G2", "G3"]
ORDINAL_COLUMNS = [
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
]
NOMINAL_COLUMNS = ["school", "Mjob", "Fjob", "reason", "guardian"]
BINARY_COLUMNS = [
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

# Fixed bands from the Project Decisions in PHASES.md; medium includes both endpoints.
PERFORMANCE_BANDS = {
    "LOW_UPPER_EXCLUSIVE": 10,
    "MEDIUM_LOWER_INCLUSIVE": 10,
    "MEDIUM_UPPER_INCLUSIVE": 13,
    "HIGH_LOWER_INCLUSIVE": 14,
}

# Feature sets are assembled exclusively from documented column roles; G3 is the target only.
MODEL_A_FEATURES = tuple(
    column
    for column in NUMERIC_COLUMNS + ORDINAL_COLUMNS + NOMINAL_COLUMNS + BINARY_COLUMNS
    if column not in LEAKY_COLUMNS
)
MODEL_B_FEATURES = MODEL_A_FEATURES + ("G1", "G2")
