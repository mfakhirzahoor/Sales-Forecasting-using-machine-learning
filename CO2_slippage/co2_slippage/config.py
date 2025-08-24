from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"

FEATURE_COLUMNS = [
    "pressure_bar",
    "temperature_c",
    "gas_flow_rate_nm3h",
    "solvent_concentration_pct",
    "contact_time_s",
    "packing_height_m",
]
TARGET_COLUMN = "co2_slippage_pct"

for _dir in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)