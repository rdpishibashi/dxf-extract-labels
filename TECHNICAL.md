# TECHNICAL.md — DXF-extract-labels

技術文書は `docs/` 以下に分割されている（2026-07-12、旧・単一ファイル約160KBを分割）。
まず [docs/INDEX.md](docs/INDEX.md) を読み、作業内容に応じて必要なファイルへ進むこと。

| ファイル | 内容 |
|---------|------|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | 概要・アーキテクチャ・Excel出力仕様・オプション仕様・FAQ等 |
| [docs/REGION_DETECTION.md](docs/REGION_DETECTION.md) | 矩形領域抽出（領域選択オプション）の全体 |
| [docs/REF_DESIGNATOR.md](docs/REF_DESIGNATOR.md) | 機器符号（候補）抽出パイプラインの全体 |
| [docs/TERMINAL_DETECTION.md](docs/TERMINAL_DETECTION.md) | 端子一覧抽出（「端子一覧を抽出」オプション）の全体 |
| [docs/VERSION_HISTORY.md](docs/VERSION_HISTORY.md) | v1.0.0〜現行までのバージョン履歴 |

このファイル自体は他プロジェクトの `CLAUDE.md` 等からの既存参照
（例: `Tools/CLAUDE.md` の「grouping (handles overlapping old/new title blocks —
see DXF-extract-labels TECHNICAL.md)」）を壊さないためのポインタとして維持している。

最終更新: 2026-07-14（docs/TERMINAL_DETECTION.md を追加、v1.8.0）
