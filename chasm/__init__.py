from pathlib import Path

from dotenv import load_dotenv

# Read CHASM_* settings (CHASM_DATA_DIR, CHASM_SIM_DATA_DIR, ...) from a .env
# file in the repository root, so they apply to every chasm module, not only
# to the one that happens to be imported first. Variables already set in the
# environment win over .env.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
