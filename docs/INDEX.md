# INDEX.md — DXF-extract-labels 技術文書の歩き方

`TECHNICAL.md`（旧・単一ファイル、約160KB）を役割ごとに分割したもの。
作業内容に応じて必要なファイルだけを読めばよい。

| ファイル | 内容 | こんなときに読む |
|---------|------|----------------|
| [OVERVIEW.md](OVERVIEW.md) | 概要・ディレクトリ構成・アーキテクチャ（データフロー・MTEXT整形・図番/流用元判別・タイトル/サブタイトル抽出）・Excel出力仕様・オプション仕様・セッション状態・依存パッケージ・既知の制限・機能拡張ポイント・システム要件・トラブルシューティング・DXF-label-diffとの違い・ライセンス | 全体像を把握したい／`extract_labels.py`本体の抽出ロジックを触る |
| [REGION_DETECTION.md](REGION_DETECTION.md) | 矩形領域抽出（領域選択オプション）の処理フロー・検出アルゴリズム・Excel出力・設定パラメータ一覧 | `region_detector.py`・「領域選択オプション」まわりを触る |
| [REF_DESIGNATOR.md](REF_DESIGNATOR.md) | 機器符号（候補）抽出パイプライン・候補/除外/確定パターン・判断ログ（`decision_log.py`）・開発用ツール（`reference_designator_analyzer.py`） | `ref_designator.py`・「未確定ラベル」まわりを触る |
| [TERMINAL_DETECTION.md](TERMINAL_DETECTION.md) | 端子一覧抽出（「端子一覧を抽出」オプション）の対象ファイル判定・端子台矩形検出アルゴリズム・ラベル-矩形対応判定（隣接矩形衝突解消・90°回転対応）・Excel出力 | `terminal_detector.py`・「端子一覧を抽出」まわりを触る |
| [VERSION_HISTORY.md](VERSION_HISTORY.md) | v1.0.0〜現行までの変更履歴（1エントリ=1バージョン） | 過去の経緯・特定バージョンでの変更内容を調べたい |

ルート直下の `TECHNICAL.md` は本ファイルへの短いポインタとして維持している
（他プロジェクトの `CLAUDE.md` 等からの既存参照を壊さないため）。

---

最終更新: 2026-07-23（REGION_DETECTION.md に、図面枠が見つからない図面で
`analyze_dxf_regions()` の `labels` が空になり「機器符号（候補）以外も抽出」
ON時にラベルが丸ごと消える不具合の修正を反映、v1.9.10）

最終更新: 2026-07-23（REGION_DETECTION.md に、regionモードの Summary シートに
`図番`・`タイトル` 列が無かった欠落の修正を反映、v1.9.9）

最終更新: 2026-07-23（REGION_DETECTION.md に境界線ちょうどの点を内側扱いする
`_point_in_polygon` の `boundary_eps` 修正を反映、v1.9.8）

最終更新: 2026-07-15（`utils/` を `model/` にリネーム。OVERVIEW.md・REF_DESIGNATOR.md・
REGION_DETECTION.md・TERMINAL_DETECTION.md のパス表記を更新、v1.9.5）

最終更新: 2026-07-15（REGION_DETECTION.md にT字接合の局所修復・合体親のN子一般化・
名称先頭文字列指定・「領域を検出」オプション化を反映、v1.9.4）
