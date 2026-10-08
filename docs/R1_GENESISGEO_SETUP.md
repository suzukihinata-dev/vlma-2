# R1: GenesisGeo のセットアップと検証

## 対象と固定版

- 対象ホスト: `<user>@<gpu-host>`、Windows Docker Desktop（Linux コンテナ）。作業ディレクトリは `<WORKDIR>\vlma-2`。
- 上流コード: [ZJUVAI/GenesisGeo](https://github.com/ZJUVAI/GenesisGeo) の `22fe601034ddc32b095c38d07a29ba051bdc8142`。
- 公開モデル: [ZJUVAI/GenesisGeo-2B](https://huggingface.co/ZJUVAI/GenesisGeo-2B) の `3295483676aa256cbc90b013adae97a1bb45e043`。
- Python 3.10.19 のイメージをダイジェストで固定。`Dockerfile.genesisgeo` はDDARと生成機、`Dockerfile.genesisgeo-model` はPyTorch CUDA 12.8と画像言語モデルを追加する。
- 上流の `full` 追加依存は学習・文書作成まで含み、基本依存の `adjustText` が無関係のHTTPプロキシを経由する。そのためR1では必要な依存のみ公式PyPI・PyTorch配布先から導入する。実際に解決された版は `artifacts/genesisgeo/pip-freeze.txt` に記録する。完全な依存ロックではないため、再ビルド時は同ファイルとの比較が必要。
- 上流のC++ DDARは通常のwheelインストールでは実行位置から見えない。編集可能な形でインストールし、4つのC++モジュールの読み込みをビルド時・検証時に確認する。
- 上流生成機の少量実行では、書き込み件数がしきい値に達せず終了しない。`patches/genesisgeo-small-batch.patch` は目標件数に達した時点で保留中のデータを出力する。この修正は上流固定版にだけ適用する。

## セットアップ・検証

PowerShellでリポジトリのディレクトリに移動して実行する。

```powershell
docker compose build genesisgeo
docker compose run --rm genesisgeo bash /usr/local/bin/r1_genesisgeo_verify
docker compose build genesisgeo-model
docker compose run --rm genesisgeo-model python /usr/local/bin/r1_genesisgeo_model_verify.py
```

最初の検証は上流のAPI・規則・生成関連テスト、実際の定理適用を伴う直交心の証明、少量の図付き問題生成を行う。生成JSONLの各行に問題・証明・PNG画像があることを確認する。出力は `artifacts/genesisgeo/` に記録する。

モデル検証は公開重みを `data/genesisgeo-2b/` に取得し、GPUに読み込んで図と問題文から `<aux>` の補助作図を生成する。GenesisGeo自身の変換関数で作図命令へ解析可能なことを確認する。重みのSHA-256と応答を `artifacts/genesisgeo/model-smoke.json` に記録する。これは単発推論の確認であり、IMOベンチマークの再現や、モデルの提案をDDARに戻す完全な探索系の性能検証ではない。

`data/` と `artifacts/` はリポジトリへ追加しない。旧 AlphaGeometry の環境は消去せず、別のComposeサービスとして残す。

## 実行結果

2026-10-06、`<gpu-host>` 上で上記の両検証コマンドが終了コード0で成功した。

- 上流の証明・生成関連テスト: 60 passed、5 xfailed。`artifacts/genesisgeo/upstream-tests.log` に記録。
- DDAR: 直交心の問題を証明し、`r43 Orthocenter theorem` の適用を `solver/proof_steps.txt` に記録。SVG図も作成。
- 生成機: 独立した新規出力ディレクトリで56件を生成し、各件の問題文、証明の手順、PNG図を確認。`--n_samples 2` を指定したが、上流のバッチ処理によって実件数は56件となった。厳密な件数制御はR2で検討する。
- 図: PNGを1枚目視確認。図の全件に対する幾何学的整合性・視認性の検証はR2の対象。
- 公開モデル: RTX PRO 4500 Blackwell上で`torch 2.8.0+cu128`を使用。図と問題文から `<aux> x00 f : ... </aux>` を生成し、GenesisGeoの変換関数で `f = on_dia f c b, on_pline f a c d` として解析できた。`model-smoke.json` に入力・応答を記録。
- モデル重み `model.safetensors` は4,877,470,336バイトで、SHA-256は `21eea002e8e565d4ebdddeb8a574ee55928b7c588deaf38d0baaab7940797ad0`。モデル環境でも `pip check` は成功した。

残る境界: モデルが提案した補助作図をDDARに投入して目標を証明する閉ループ、ベンチマーク解答率、大規模生成速度はまだ未検証。Transformersはトークナイザーの正規表現に関する警告を表示したが、今回の単発推論は成功した。再現実験前にこの警告の影響を調べる。

## 公式探索コードのスモーク実行（vLLM 経由）

2026-10-06、`genesisgeo-vllm` サービス（`Dockerfile.genesisgeo-vllm`、起動は `scripts/r1_genesisgeo_vllm.sh`）上で、GenesisGeo 公式の `scripts/evaluation.py` を最小設定で実行した。

```powershell
docker compose exec -T genesisgeo-vllm python scripts/evaluation.py --agent qwen3_vl --problems_path /artifacts/genesisgeo/eval-smoke-problems.txt --vllm_base_url http://127.0.0.1:8000 --decoding_size 2 --beam_size 2 --search_depth 1 --ray_num_cpus 2 --timeout 120 --log_dir /artifacts/genesisgeo/eval-smoke --enable_trace
```

- 問題: `worlds_hardest_easy_geometry_problem1` の1問のみ。
- 結果: **未解決**（`Tried but failed.`）。モデル呼び出し2回、DDAR 呼び出し5回、所要48.17秒。成功例には数えない。
- 実行記録（トレース JSONL）で確認できた範囲: モデルは各探索で2候補を返し、4候補すべてが `<aux>` として解析（parsed=True）され、作図に変換（built=True）され、DDAR に投入され、結果は4件とも `unsolved`、エラーなし。つまり「モデル提案 → 解析 → 作図 → DDAR 検証」の閉ループは動作した。
- 提案例: `g = on_circle g a c, on_circle g b c` など。いずれも目標を証明するに至らなかった。
- 1回目のモデル呼び出しは44.3秒、2回目は0.3秒。初回はウォームアップが支配的とみられる（未検証の推測）。
- 設定を `d2 / b2 / depth 1` に絞ったため、この結果からモデルの性能は判断できない。解答率は未測定。
- トークナイザーの `fix_mistral_regex` 警告は vLLM 経由でも出た。影響調査の結果は下の「R1 完了判定」を参照。
- 出力はWNPCの `artifacts/genesisgeo/eval-smoke/` に保存（CSV、`run_meta.json`、問題ごとの JSONL、描画PNG）。

## トークナイザー警告の調査

transformers 4.57.6 で `AutoTokenizer.from_pretrained` を `fix_mistral_regex=True` の有無で比較した（`artifacts/genesisgeo/tok.py`）。補助作図 DSL、作図命令、述語列を含む4文のトークンIDは、フラグの有無で完全に一致した（4/4、各13〜29トークン）。この範囲では警告はこのモデルの入出力に影響しない。ただし4文だけの確認で、自然文や多言語の問題文は見ていない。

## R1 完了判定

R1「GenesisGeo の証明エンジン、生成機、公開モデルが仕様通りに動く」は、**各部品が動作する**という基準で完了とする（この基準は要件に明記がなく、2026-10-06 に採用した）。

- 証明エンジン: 上流テスト60 passed・5 xfailed、直交心の証明。
- 生成機: 図付き問題を生成。件数制御（`--n_samples 2` で56件）と全件の図の整合性は R2 で扱う。
- 公開モデル: 重み読み込み、補助作図の生成、DDAR による検証の閉ループが動作。

未達成のまま残るもの（R1 の完了には含めない）: モデルの解答率と論文値との比較、大規模生成速度。1問・`d2/b2/depth1` の探索は未解決だった。
