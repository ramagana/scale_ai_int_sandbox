
# ML Debug Drills — JSON → Tensor

Quick, realistic debugging exercises for a 60‑minute ML coding interview.
Use `debug_drill.py` to trigger specific failures, then fix them step-by-step.

## Files
- `data.json` — valid sample data with keys: `values`, `labels`
- `bad_data.json` — intentionally mismatched keys/types
- `debug_drill.py` — buggy script with multiple modes to trigger common errors
- `debug_drill_solution.py` — a robust, fixed version for reference

## Setup
```bash
pip install torch numpy
```

## Exercises
- KeyError:
  ```bash
  python debug_drill.py --mode keyerror --json data.json
  ```
- TypeError:
  ```bash
  python debug_drill.py --mode typeerror --json data.json
  ```
- IndexError:
  ```bash
  python debug_drill.py --mode indexerror --json data.json
  ```
- FileNotFoundError:
  ```bash
  python debug_drill.py --mode filenotfound --json data.json
  ```
- Device mismatch (RuntimeError):
  ```bash
  python debug_drill.py --mode devicemismatch --json data.json
  ```
- All good:
  ```bash
  python debug_drill.py --mode ok --json data.json
  ```

## Debugger
Add:
```python
breakpoint()
```
Or run with pdb:
```bash
python -m pdb debug_drill.py --mode keyerror --json data.json
```

## Solution
```bash
python debug_drill_solution.py --json bad_data.json --device cpu
```
