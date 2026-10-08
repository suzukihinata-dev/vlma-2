#!/usr/bin/env bash
set -euo pipefail

cd /opt/genesisgeo
result_dir=/artifacts/genesisgeo
mkdir -p "$result_dir"
printf 'R1 GenesisGeo verification running; previous success does not apply.\n' > "$result_dir/r1-result.txt"

git rev-parse HEAD > "$result_dir/genesisgeo.commit"
test "$(cat "$result_dir/genesisgeo.commit")" = '22fe601034ddc32b095c38d07a29ba051bdc8142'
python -m pip freeze > "$result_dir/pip-freeze.txt"
python -m pip check > "$result_dir/pip-check.log"
sha256sum /tmp/genesisgeo-small-batch.patch > "$result_dir/generator-patch-sha256.txt"
python - <<'PY' > "$result_dir/native-imports.log"
import newclid.DDAR.build
import newclid.matchinC
import newclid.generation.auxiliary
import newclid.dependencies.geometry
print('All four native modules imported.')
PY

pytest -q \
  tests/test_api.py \
  tests/test_individual_rules.py \
  tests/test_generation_pipeline_cli.py \
  tests/test_generation_writer.py \
  > "$result_dir/upstream-tests.log" 2>&1

python - <<'PY' > "$result_dir/solver.log" 2>&1
from pathlib import Path
from newclid.api import GeometricSolverBuilder

problem = (
    'a b c = triangle a b c; '
    'h = on_tline h b a c, on_tline h c a b '
    '? perp a h b c'
)
solver = GeometricSolverBuilder(seed=123).load_problem_from_txt(problem).build()
assert solver.run(timeout=120), solver.run_infos
out = Path('/artifacts/genesisgeo/solver')
out.mkdir(parents=True, exist_ok=True)
solver.write_proof_steps(out / 'proof_steps.txt')
solver.draw_figure(out_file=out / 'construction.svg')
assert (out / 'proof_steps.txt').stat().st_size > 0
assert (out / 'construction.svg').stat().st_size > 0
print(solver.run_infos)
PY

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
generated_dir="$result_dir/generated/$run_id"
timeout 600 python src/newclid/generation/pipeline.py \
  --n_clauses 8 \
  --n_samples 2 \
  --n_threads 2 \
  --max_level 100 \
  --aux_only 0 \
  --img 1 \
  --dir "$generated_dir" \
  > "$result_dir/generation.log" 2>&1

R1_GENERATED_DIR="$generated_dir" python - <<'PY' > "$result_dir/generated-validation.log"
import glob
import json
import os
from pathlib import Path

paths = glob.glob(os.environ['R1_GENERATED_DIR'] + '/*/*.jsonl')
assert len(paths) == 1, paths
records = [json.loads(line) for line in Path(paths[0]).read_text().splitlines() if line]
assert len(records) >= 2, len(records)
for record in records:
    assert '<problem>' in record['llm_input_renamed']
    assert '<proof>' in record['llm_output_renamed']
    assert record['n_proof_steps'] > 0
    image = Path(record['image_path'])
    assert image.is_file() and image.stat().st_size > 0, image
print(f'Validated {len(records)} new problems, proof traces, and PNG diagrams: {paths[0]}')
PY

printf 'R1 GenesisGeo symbolic engine and generator passed; model inference is a separate check.\n' \
  | tee "$result_dir/r1-result.txt"
