
# KNOWLEDGE

このリポジトリは親ディレクトリにある。vision-language-mathemacis-action-modelの引き継ぎリポジトリです。

vlma-1で得た教訓をもとにvlmaを再開発するのがこのリポジトリでのあなたの役割です。
COREは実装方針です。COREに基づいて実装を行って下さい。

## 版の管理（必ず守る。2026-10-15〜）

論文で PoC → v1 → v2 と発表するため、**1 つのコードに変更を焼き込まない**。`vlma/` と `scripts/`（直下）は PoC（baseline）として凍結し、新しい版の変更は `vlma_v1/`・`scripts/v1/`・`tests/v1/`・`docs/versions/v1/`（v2 も同様）に、継承・ラップ・差し替えで重ねる。baseline を変えてよいのは、不具合の修正と、既定の挙動を変えない拡張点だけ（log に理由を書き、`docs/versions/poc/MANIFEST.json` を更新）。詳細は **[docs/VERSIONING.md](docs/VERSIONING.md)**。コードを変える前に必ず読む。

## タスクの管理（2026-10-28〜）

これから出るタスクは、**[TASK.md](TASK.md)** に積んでいく。作業を始める前に必ず読む。

- 新しいタスク（ユーザーの指示、作業中に見つけた課題・不具合・未測定の項目）は、TASK.md の末尾に、決まった書式で追記する。番号は通し番号で、振り直さない。
- 着手したら状態を `doing`、終えたら `done`（結果の場所を書く）、やめたら `dropped`（理由を書く）に変える。項目は消さない。
- どのタスクを次にやるかはユーザーが決める。TASK.md にあることを理由に、指示なしで着手しない。

## LESSON

1. 車輪の再開発をしない。

## RULE

1. 実装内容はlog/*.logに記録していくこと(claudeの場合はCLAUDE.log, codexの場合はCODEX.logに記録していくこと)
