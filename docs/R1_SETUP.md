# R1: AlphaGeometry v1 のセットアップと検証

> これは旧実装の検証記録です。現在のR1は [GenesisGeo版](R1_GENESISGEO_SETUP.md) を使用します。

## 構成

- 対象ホスト: `<user>@<gpu-host>` の Windows Docker Desktop（Linux コンテナ）。作業ディレクトリは `<WORKDIR>\vlma-2`。
- 上流コード: `google-deepmind/alphageometry` の `6777cb586cbb46beed28db12dc72c69770b68337`。
- 言語モデル実装: `google-research/meliad` の `e8af0543441222c1c4c60d58803511f7cf92908b`。以後の版では AlphaGeometry が参照する `transformer.decoder_stack` が削除されている。
- Docker の画面なし環境で読み込めるよう、ビルド時に上流 `numericals.py` の描画設定だけを `TkAgg` から `Agg` に変更する。
- Python: 3.10.19（Bookworm のイメージをダイジェストで固定）。上流は 3.10.9 を使用しているが、その古いイメージでは Debian パッケージの取得が 404 になった。依存パッケージは上流の `requirements.txt` に記されたバージョンとハッシュで固定する。PyPI から消えた `jaxlib==0.4.6` は [JAX 公式配布先](https://storage.googleapis.com/jax-releases/jax_releases.html) を追加検索する。上流の固定一覧を `--no-deps --require-hashes` でそのまま入れ、`pip check` で依存の整合性を確認する。
- 学習済み重みと語彙は、公式配布先の Google Drive が CLI から取得不能だったため、公式 Drive フォルダから移したと明記する [Hugging Face の複製](https://huggingface.co/Wauplin/alphageometry/tree/main) の固定コミット `77df8a094307f7c2148f2b87b6d8986f65060b5f` から取得する。複製元とのバイト単位の照合は未実施。上流 README による重み・語彙のライセンス表記は CC BY 4.0 で、複製ページの表示とは異なる。`data/` と `artifacts/` はバージョン管理から除外する。

## 初回セットアップ

Windows ホストの PowerShell または SSH 接続先で、リポジトリのディレクトリに移動して実行する。

```powershell
docker compose build
docker compose run --rm -v <WORKDIR>/vlma-2/data:/data alphageometry huggingface-cli download Wauplin/alphageometry checkpoint_10999999 geometry.757.model geometry.757.vocab --revision 77df8a094307f7c2148f2b87b6d8986f65060b5f --local-dir /data/ag_ckpt_vocab --local-dir-use-symlinks False --resume-download
```

取得後に3ファイルがあることを確認する。検証スクリプトは取得した3ファイルの SHA-256 を固定値と照合する。配布物はリポジトリや Docker イメージに埋め込まない。

## R1 の検証

```powershell
docker compose run --rm alphageometry bash /usr/local/bin/r1_verify
```

このコマンドは公式 `run_tests.sh`、DDAR による `translated_imo_2000_p1`、学習済みモデルを使う `orthocenter` の順に実行する。公式テストの `test_lm_score_may_fail_numerically_for_external_meliad` だけは、[上流 README と Issue #14](https://github.com/google-deepmind/alphageometry/issues/14) に記載されたスコア差を許容する。その1件以外の失敗、DDAR の失敗、モデル付き探索の失敗では終了コードが非ゼロになり、`artifacts/r1-result.txt` は成功として更新されない。再検証前にはこのファイルを確認し、過去の成功記録と今回の実行を混同しないこと。

実行ログと両上流のコミット、配布物の SHA-256 は `artifacts/` に保存する。公式テストの結果は `artifacts/official-tests-result.txt` に記録する。`artifacts/alphageometry-orthocenter.log` に補助作図の `Translation:` と `Solved.` が残ることを合格条件とする。これは代表問題の動作確認であり、論文の全問題の解答数の再現ではない。

## 2026-10-06 の実行結果

- `<gpu-host>` で Docker イメージの構築と `pip check` が成功した。
- 取得した3ファイルは固定した SHA-256 に一致した。
- 公式テストはモデルのスコアを固定値と比較する既知の1件だけが失敗した。それ以外のテストに失敗は記録されていない。
- DDAR は `translated_imo_2000_p1` の証明を出力し、モデル付き AlphaGeometry は `orthocenter` に補助作図を提案して `Solved.` まで到達した。
- `docker compose run --rm alphageometry bash /usr/local/bin/r1_verify` は終了コード 0。生ログと SHA-256 一覧は接続先の `<WORKDIR>\vlma-2\artifacts\` に保存した。

## 再構築

同じ `Dockerfile` と `data/ag_ckpt_vocab/` から `docker compose build --no-cache` と上記検証コマンドを実行する。配布物を取り直した場合は SHA-256 一覧を比較して差分を調べる。
