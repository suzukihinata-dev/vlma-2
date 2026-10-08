#!/usr/bin/env bash
set -euo pipefail

cd /opt/alphageometry
mkdir -p /artifacts
printf 'R1 verification running; previous success does not apply.\n' > /artifacts/r1-result.txt

test -s ag_ckpt_vocab/geometry.757.model
test -s ag_ckpt_vocab/checkpoint_10999999
test -s ag_ckpt_vocab/geometry.757.vocab

printf '%s\n' \
  '02d6728be6269e768a485620834a627aaa11d2852f16a8a478be38aee04123cb  ag_ckpt_vocab/checkpoint_10999999' \
  'a219a8cf71d57c2e71e345bf1777fedec81452086ec39446c804fb7fd07fbedb  ag_ckpt_vocab/geometry.757.model' \
  '65845380b9c4341ac12460312f82e1b911f02e38559e51d4d9a6c4749dded9d2  ag_ckpt_vocab/geometry.757.vocab' \
  | sha256sum --check

git rev-parse HEAD > /artifacts/alphageometry.commit
git -C meliad_lib/meliad rev-parse HEAD > /artifacts/meliad.commit
find ag_ckpt_vocab -type f -print0 | sort -z | xargs -0 sha256sum > /artifacts/asset-sha256.txt

if bash run_tests.sh > /artifacts/official-tests.log 2>&1; then
  printf 'Official test suite passed.\n' > /artifacts/official-tests-result.txt
else
  failed_tests="$(grep -F '[  FAILED  ]' /artifacts/official-tests.log || true)"
  tracebacks="$(grep -c '^Traceback' /artifacts/official-tests.log || true)"
  if [[ "$failed_tests" == '[  FAILED  ] LmInferenceTest.test_lm_score_may_fail_numerically_for_external_meliad' ]] \
      && [[ "$tracebacks" == 1 ]] \
      && grep -Fqx 'FAILED (failures=1)' /artifacts/official-tests.log \
      && ! grep -Fq 'ERROR:' /artifacts/official-tests.log; then
    printf 'Official test suite: one documented external-Meliad numeric score mismatch.\n' \
      > /artifacts/official-tests-result.txt
  else
    printf 'Official test suite failed outside the documented numeric mismatch.\n' \
      > /artifacts/official-tests-result.txt
    exit 1
  fi
fi
cat /artifacts/official-tests-result.txt

python -m alphageometry \
  --alsologtostderr \
  --problems_file="$(pwd)/imo_ag_30.txt" \
  --problem_name=translated_imo_2000_p1 \
  --mode=ddar \
  --defs_file="$(pwd)/defs.txt" \
  --rules_file="$(pwd)/rules.txt" \
  > /artifacts/ddar-imo-2000-p1.log 2>&1

grep -Fq 'Proof steps:' /artifacts/ddar-imo-2000-p1.log
grep -Eq '(EP|PE) = (EQ|QE)|(EQ|QE) = (EP|PE)' /artifacts/ddar-imo-2000-p1.log
printf 'DDAR solved translated_imo_2000_p1.\n'

python -m alphageometry \
  --alsologtostderr \
  --problems_file="$(pwd)/examples.txt" \
  --problem_name=orthocenter \
  --mode=alphageometry \
  --defs_file="$(pwd)/defs.txt" \
  --rules_file="$(pwd)/rules.txt" \
  --beam_size=2 \
  --search_depth=2 \
  --ckpt_path=ag_ckpt_vocab \
  --vocab_path=ag_ckpt_vocab/geometry.757.model \
  --gin_search_paths="$(pwd)/meliad_lib/meliad/transformer/configs,$(pwd)" \
  --gin_file=base_htrans.gin \
  --gin_file=size/medium_150M.gin \
  --gin_file=options/positions_t5.gin \
  --gin_file=options/lr_cosine_decay.gin \
  --gin_file=options/seq_1024_nocache.gin \
  --gin_file=geometry_150M_generate.gin \
  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True \
  --gin_param=TransformerTaskConfig.batch_size=2 \
  --gin_param=TransformerTaskConfig.sequence_length=128 \
  --gin_param=Trainer.restore_state_variables=False \
  > /artifacts/alphageometry-orthocenter.log 2>&1

grep -Fq 'Translation:' /artifacts/alphageometry-orthocenter.log
grep -Fq 'Solved.' /artifacts/alphageometry-orthocenter.log
printf 'AlphaGeometry solved orthocenter.\n'
printf 'R1 functional verification passed; see official-tests-result.txt for numeric score exception.\n' \
  | tee /artifacts/r1-result.txt
