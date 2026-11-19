# Metadata-Driven Carbon-Efficient Model Selection

This project benchmarks classification models on tabular datasets, stores their results in a knowledge base, and uses dataset metadata plus user constraints to select a candidate model for a new dataset. The optional AI runner generates a Gemini explanation; if Gemini is unavailable or no API key is configured, it prints a local fallback explanation.

## Project Flow

1. **Phase 1:** `src/Phase1.py` preprocesses configured/discovered datasets, trains candidate classifiers, and writes performance results to `knowledge_base.csv`.
2. **Meta-model:** `train_meta_model.py` trains and saves `meta_model.pkl` using the knowledge base.
3. **Phase 2:** `src/phase2.py` selects a model using metadata and constraints, trains it on the new dataset, and measures training resources.
4. **AI demo (optional):** `run_phase2_with_ai.py` runs the lightweight Phase 2 path and prints a Gemini or fallback explanation. It accepts a dataset path, target column, and constraints.

The AI runner uses `src/phase2_lite.py`, which does not measure energy or CO2 during its new training run. Resource constraints and the displayed energy/CO2 values come from the historical knowledge base. Use `src/phase2.py` when you need resource measurement for the new training run.

## Setup

Requires Python 3.10 or later. From the project root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

The repository includes `knowledge_base.csv` and `meta_model.pkl` for a quick run. To benchmark datasets again and rebuild those artifacts, run:

```powershell
python src/Phase1.py --output knowledge_base.csv --root .
python train_meta_model.py
```

This benchmarking step can take a while because it trains multiple classifiers across datasets.

### Phase 2 with Gemini/fallback explanation and constraints

From the project root:

```powershell
python run_phase2_with_ai.py "Data/Bank Marketing Dataset/bank.csv" --target deposit --knowledge-base knowledge_base.csv --meta-model meta_model.pkl --min-accuracy 0.85 --max-energy 0.01 --max-co2 0.001
```

To try a different dataset, replace the path and target. Example:

```powershell
python run_phase2_with_ai.py "Data/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv" --target Churn --knowledge-base knowledge_base.csv --meta-model meta_model.pkl --min-accuracy 0.85 --max-energy 0.01 --max-co2 0.001
```

The accuracy constraint is a fraction from `0` to `1`; energy is in kWh and CO2 is in kg. Constraints are optional. If no model in the knowledge base satisfies all supplied constraints, the selector warns and falls back to the best overall candidate.

### Gemini API key (optional)

The AI runner works without a key by using its local fallback explanation. To request a Gemini-generated explanation, set your key in PowerShell before running the command:

```powershell
$env:GOOGLE_API_KEY="your_google_ai_api_key"
python run_phase2_with_ai.py "Data/Bank Marketing Dataset/bank.csv" --target deposit --knowledge-base knowledge_base.csv --meta-model meta_model.pkl --min-accuracy 0.85 --max-energy 0.01 --max-co2 0.001
```

Never put the real key in source code or commit it to GitHub. If a key has previously been exposed in source code, chat, logs, or a public repository, revoke it and create a replacement.

### Full Phase 2 with resource measurement

```powershell
python src/phase2.py "Data/Bank Marketing Dataset/bank.csv" --target deposit --knowledge-base knowledge_base.csv --meta-model meta_model.pkl --min-accuracy 0.85 --max-energy 0.01 --max-co2 0.001
```

### Tests

```powershell
pytest tests/ -v
```

## Main Files

```text
src/Phase1.py            Benchmark models and generate the knowledge base
src/phase2.py            Full Phase 2 with energy/carbon tracking
src/phase2_lite.py       Lightweight Phase 2 without live energy tracking
src/model_selector.py    Meta-model prediction and constraint-based selection
src/preprocessing.py     Data cleaning, encoding, and train/test split
src/metadata_extractor.py Extract dataset metadata
src/config.py            Dataset paths and target configuration
train_meta_model.py      Train and save the meta-model
run_phase2_with_ai.py    Optional Gemini explanation runner
ai_explainer.py          Gemini integration and fallback explanation
knowledge_base.csv       Historical benchmark results
meta_model.pkl           Saved selector model and label encoder
```

## Datasets and GitHub

Raw datasets are intentionally excluded from this repository by `.gitignore`. Download datasets from their original sources and place them under the matching `Data/` paths used by `src/config.py` and the run commands. For example, the bank dataset should be at `Data/Bank Marketing Dataset/bank.csv`.

Check each dataset's license and terms before downloading or using it. The knowledge base and meta-model are included so the project can use its existing benchmark results, but Phase 2 still needs the selected input dataset file locally.
