# vlma-2

このリポジトリは、親リポジトリにあるvision-language-mathematics-action-modelの引き継ぎリポジトリです。vlma-1 で得た教訓をもとに、vlma を再開発します。

R1 では GenesisGeo を Docker で固定し、証明エンジン、図付き問題生成、公開モデルを検証します。セットアップと検証手順は [docs/R1_GENESISGEO_SETUP.md](docs/R1_GENESISGEO_SETUP.md) を参照してください。旧 AlphaGeometry v1 の検証記録は [docs/R1_SETUP.md](docs/R1_SETUP.md) に残しています。

## 構成

| パス | 内容 |
| --- | --- |
| `vlma/` | ライブラリ本体（データ生成、検証器、環境、ベンチマーク） |
| `scripts/` | 実行スクリプト（データ生成、学習、評価、ベンチマーク） |
| `tests/` | テスト（`tests/data/` は小さな試験用データ） |
| `docs/` | 設計と検証の記録（R1〜R4） |
| `patches/` | 外部ライブラリに当てるパッチ |
| `log/` | 作業ログ（`CLAUDE.log`、`CODEX.log`） |

## データとモデルの保存先（Hugging Face）

このリポジトリ（Git）には、コード、設定、文書、ログだけを置きます。**データと学習済みモデルは Git に入れず、Hugging Face（ユーザー: SuzukiHinata）に保存します。**

- `data/`、`artifacts/`、`checkpoints/`、`outputs/` と、重みファイル（`*.safetensors`、`*.pt`、`*.bin` など）は `.gitignore` で除外しています。
- 学習データは `scripts/r3_pack_for_hf.py` で、Hugging Face にアップロードしやすい形（図の tar 束ね、ファイル数の削減、データの説明 README）に整えてから保存します。データセットは公開（public）で保存します（[SuzukiHinata/vlma-2](https://huggingface.co/datasets/SuzukiHinata/vlma-2)）。
- 学習済みモデル（方式 A'、方式 B の学習結果）も、Hugging Face の model リポジトリに保存します。
- Hugging Face からの取得には `HF_TOKEN` を使います。トークンは環境変数で渡し、リポジトリには書きません。

| 種類 | 保存先 | 状態 |
| --- | --- | --- |
| 学習データ（R2 の幾何問題、図つき、10万問） | Hugging Face dataset: [SuzukiHinata/vlma-2](https://huggingface.co/datasets/SuzukiHinata/vlma-2)（public） | repo 作成済み。再パックしたデータをアップロード中 |
| 学習済みモデル（run1） | Hugging Face model: [SuzukiHinata/vlma2-run1](https://huggingface.co/SuzukiHinata/vlma2-run1) | アップロード済み（現在 private。公開は未定） |
| 学習済みモデル（runB） | Hugging Face model: [SuzukiHinata/vlma2-runB](https://huggingface.co/SuzukiHinata/vlma2-runB) | アップロード済み（現在 private。公開は未定） |
| 外部の公開モデル（GenesisGeo-2B など） | 取得元の Hugging Face（[ZJUVAI/GenesisGeo-2B](https://huggingface.co/ZJUVAI/GenesisGeo-2B) など）から直接取得 | 利用中 |

### 取得の例

```bash
# データ
hf download SuzukiHinata/vlma-2 --repo-type dataset --local-dir data/

# 学習済みモデル
hf download SuzukiHinata/vlma2-runB --local-dir artifacts/models/<name>
```

## ライセンス

コードは [Apache License 2.0](LICENSE) です。`patches/` は Apache-2.0 の [GenesisGeo](https://github.com/ZJUVAI/GenesisGeo) に当てる差分です。
