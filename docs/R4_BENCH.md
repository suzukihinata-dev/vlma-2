# R4: ベンチマーク（モデルに問題を解かせ、検証器で測る）

学習した方式（A' / B）やモデルを、**同じ問題・同じ検証器**で比べるための仕組み。損失や正解率（教師強制）は、方式が違うと
比べられないので、実際に解かせて、目標に届いたかを見る。

## 何を測るか
1問 = 1エピソード。環境が、学習データと同じ形の状態（問題・情報欄・過去の手・図）を返し、モデルが1手を出し、環境（`vlma.verify.StepVerifier`、
DDAR）が1手ずつ検証する。受理なら状態が進み（作図なら、その作図を含む図を描き直し、情報欄に数値を足す）、却下なら状態は変わらない。
目標に届く（受理した手の結論が目標と一致）か、試行の上限で終わる。上限は、参照証明の手数 n から `min(400, max(2n, n+20))`。

| 指標 | 意味 |
|---|---|
| `success_rate` | 目標に届いた問題の割合（主指標） |
| `kind_rate` | 届いた問題の導出の種類。shortest（却下・不要な手なし、手数が参照以下）/ wasteful / incomplete（届かない） |
| `move_accept_rate` | 試した手のうち、受理された割合 |
| `progress` | 受理した手数 ÷ 参照の手数（上限1）の平均。届かなくても、どこまで進めたか |
| `reject_reasons` | 却下の理由（not_hold, build_failed, not_derivable, invalid_citation, rule_mismatch, ...） |
| `reference_coverage` | 参照証明が使う事実（適用の手の結論）のうち、予算の中で見つけた割合の平均。届かなくても、必要な事実をどこまで見つけたか |
| `success_within` | 目標に届いた割合を、参照の手数の 1 / 1.5 / 2 倍までの試行（却下を含む）で数えたもの |
| `needed_share_of_accepted_apply` | 受理した適用のうち、参照証明に必要な手の割合 |
| `reward_mean` | `vlma.reward` の手ごとの報酬の合計の平均（値は仮） |
| `success_by_reference_length` | 参照の手数ごとの成功率 |

**打ち切りは、ターン予算で行う（既定）。** 却下は、間違えた回数としては数えず、1ターンを使うだけで、予算が尽きるまで続ける
（「決められたターンの間に、必要な事実を見つけられるか」を測る）。`--retries R`（1問で却下してよい回数）で、回数でも打ち切れる。
R = 0 は、最初の却下で失敗（厳密）。却下された状態では、サンプリング（`--temperature`）で出し直す。受理された手は、常に貪欲（temperature 0）。
乱数は `--seed` で固定する。**結果を比べるときは、R・seed・問題の一覧を同じにする。**

## 構成
- `vlma/env.py` `Episode`: 1問の環境（状態、検証、図の描き直し、結果）。placement_seed は学習データの作成時と同じ。検証器が例外を出しても、却下として返す。
- `vlma/envserver.py`: Episode を HTTP で出す（標準ライブラリ）。DDAR は CPU なので、`scripts/bench_env.sh <名前> [K]` で K 個のプロセス（ポート 8100〜）を起動する。
  **genesisgeo のコンテナ**（newclid が要る）で動かす。
- `vlma/bench.py`: 状態から学習データと同じプロンプトを作る（`TurnPrompt` = A'、`SeqPrompt` = B）と、結果の集計（`summarize`）。
- `scripts/bench_run.py`: **train のコンテナ**でモデルを読み、複数の問題をまとめて生成し、環境に送る。結果は `/artifacts/bench/<名前>/results.jsonl`（途中から再開できる）と `summary.json`。進捗は Slack（スレッドの中だけ）。
- `scripts/bench_make_set.py`: テスト分割から、決まった乱数で N 問の一覧（`bench_200.json`）を作る。モデルによらず固定。

## 使い方（WNPC）
```
# 1. 環境サーバー（genesisgeo のコンテナ。一度起動すれば、複数のモデルで使い回せる）
docker compose run -d --name vlma-env genesisgeo bash /opt/vlma2-scripts/bench_env.sh <名前> 6
# 2. モデルを測る（train のコンテナ。--format は、そのモデルを学習した方式: turn = A'、seq = B）
docker compose run --rm train python scripts/bench_run.py --model /artifacts/r3/<run>/model-final --format <turn|seq> --name <結果の名前> [--retries 5] [--limit N]
```
問題の一覧の作り直し: `python scripts/bench_make_set.py --dir /artifacts/genesisgeo/r2/build-100k --n 200`。

## 確認したこと
- 参照証明を環境に流すと、全手が受理され、最後の手で目標に届く（kind = shortest）。
- 環境の状態から作るプロンプトは、学習データ（A' の各ターン、B の系列）と、文章・図を見せる位置が完全に一致する（tests/test_env.py）。
- 検証器が例外を出す図（モデルが、点を別の点に重ねる作図をしたとき）で、環境サーバーが 500 を返して、問題が落ちた。検証器の例外は、そのエンジンを使わず（全部だめなら not_hold で却下）、環境でも却下として返すようにした。

## 既知の限界
- 退化した作図（新しい点が既存の点と重なる、例: `coll a c l; coll b d l` で l = a）を、検証器は受理する。以後の手が、その退化した図で成り立ってしまう可能性がある。
  参照証明には出ないので、受理しない規則を足せるか（参照が全部通るか）を確かめてから、検証器に足す（未実施）。
- `--retries 0`（厳密）は、1回の却下で失敗なので、成功率は低く出る。run1・runB の最初の測定（200問）は、この設定で、成功は 0 だった。ターン予算による打ち切りが、既定。
- 生成は、毎ターン、プロンプトの全体を計算し直す（キャッシュを使い回さない）。B は系列が長いので、遅い。
