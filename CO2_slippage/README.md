# CO2_slippage

Reduce CO₂ slippage using machine learning.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --use-synthetic --n-samples 5000
python scripts/evaluate.py --dataset test
python scripts/predict.py --features '{"pressure_bar": 12, "temperature_c": 40, "gas_flow_rate_nm3h": 800, "solvent_concentration_pct": 25, "contact_time_s": 3.2, "packing_height_m": 8.0}'
```

## Serve API

```bash
python scripts/serve.py
# or
uvicorn co2_slippage.api:app --reload --port 8000
```

## Project layout

- `co2_slippage/`: package
- `scripts/`: CLIs for train/eval/predict/serve
- `data/`: datasets
- `models/`: trained artifacts