# Milk Adulteration Detection — Backend (Flask)

Serves the `analyzer_12` Random Forest. Output is a **preliminary ML classification,
not a definitive food-safety determination**.

## Run
```bash
pip install -r requirements.txt
# put your trained model at models/analyzer_12_random_forest.joblib
# (or for a dev stub: python scripts/make_dummy_model.py)
python wsgi.py            # dev
# production: gunicorn -w 2 -b 0.0.0.0:8000 wsgi:app
```
API at http://localhost:8000

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status |
| GET | `/features` | the 12 features + labels |
| POST | `/predict` | one sample → label + probabilities |
| POST | `/predict/batch` | up to 1000 samples |
| GET | `/history` | stored predictions (if enabled) |

Example body: `{"Cells":..,"QValue":..,"Fat":..,"Protein":..,"Lactose":..,"Solids":..,"FFA":..,"Citrate":..,"FrzPoint":..,"SNF":..,"MUN":..,"Casein":..}`

## Open decisions (from the project notes)
- **Valid ranges/units:** not enforced until confirmed. Once known, create a JSON
  like `{"Fat":[0,10],"FrzPoint":[-0.6,-0.4]}` and set `MILK_RANGES_PATH`.
- **History storage:** off by default. `MILK_ENABLE_HISTORY=true` uses SQLite
  (`app/history.py`), which is easy to swap for the team's chosen DB.

## Config (env vars, prefix `MILK_`)
`MODEL_PATH`, `MODEL_NAME`, `ENABLE_HISTORY`, `DB_PATH`, `RANGES_PATH`, `CORS_ORIGINS`, `MAX_BATCH`

## Tests
`pytest` (needs a model file in `models/`)
