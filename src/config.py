from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_data.csv"

# Financial Assumptions
LOAN_TERM_MONTHS = 36  # You mentioned forcing this. We define it once here.
RECOVERY_RATE = 0.30    # Worst-case scenario (0% recovery on default)

# Modeling Constants
TARGET = 'not.fully.paid'
TEST_SIZE = 0.2
RANDOM_STATE = 42