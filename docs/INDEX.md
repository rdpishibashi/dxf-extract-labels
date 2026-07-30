# INDEX.md — DXF-extract-labels 技術文書の歩き方

`TECHNICAL.md`（旧・単一ファイル、約160KB）を役割ごとに分割したもの。
作業内容に応じて必要なファイルだけを読めばよい。

| ファイル | 内容 | こんなときに読む |
|---------|------|----------------|
| [OVERVIEW.md](OVERVIEW.md) | 概要・ディレクトリ構成・アーキテクチャ（データフロー・MTEXT整形・図番/流用元判別・タイトル/サブタイトル抽出）・Excel出力仕様・オプション仕様・セッション状態・依存パッケージ・既知の制限・機能拡張ポイント・システム要件・トラブルシューティング・DXF-label-diffとの違い・ライセンス | 全体像を把握したい／`extract_labels.py`本体の抽出ロジックを触る |
| [REGION_DETECTION.md](REGION_DETECTION.md) | 矩形領域抽出（領域選択オプション）の処理フロー・検出アルゴリズム・Excel出力・設定パラメータ一覧 | `region_detector.py`・「領域選択オプション」まわりを触る |
| [REF_DESIGNATOR.md](REF_DESIGNATOR.md) | 機器符号抽出パイプライン（3候補パターン一致で全件出力）・図面情報枠（タイトルブロック）の幾何検出 | `ref_designator.py`まわりを触る |
| [TERMINAL_DETECTION.md](TERMINAL_DETECTION.md) | 端子一覧抽出（「端子一覧を抽出」オプション）の対象ファイル判定・端子台矩形検出アルゴリズム・ラベル-矩形対応判定（隣接矩形衝突解消・90°回転対応）・Excel出力 | `terminal_detector.py`・「端子一覧を抽出」まわりを触る |
| [VERSION_HISTORY.md](VERSION_HISTORY.md) | v1.0.0〜現行までの変更履歴（1エントリ=1バージョン） | 過去の経緯・特定バージョンでの変更内容を調べたい |

ルート直下の `TECHNICAL.md` は本ファイルへの短いポインタとして維持している
（他プロジェクトの `CLAUDE.md` 等からの既存参照を壊さないため）。

---

最終更新: 2026-07-30（v2.1.0、REGION_DETECTION.md に「領域一覧」グローバル選択
機能（全ファイル・全領域の名称候補を横断的に一覧化し、チェック済み名称に一致する
領域を一括決定する）と、ボタン表示制御の再設計（「領域を検出」「ラベルを抽出」の
排他表示化、結果表示中の全ウィジェットdisabled化）を反映。OVERVIEW.md にも
セッション状態キー・オプション説明の変更を反映）

最終更新: 2026-07-30（v2.0.0、機器符号抽出の判定条件を簡素化。除外/追加/確定
パターン・「未確定ラベル」UIでのレビュー・連動採用・判断ログ（`decision_log.py`）・
開発用ツール（`reference_designator_analyzer.py`）を廃止し、3候補パターン一致で
全件出力する方式に変更。図面情報枠（タイトルブロック）の除外も構造的除外から
幾何検出に置き換えた。REF_DESIGNATOR.md を全面書き換え、除外/追加/確定パターンの
回帰テストは新設の `tests/regression/test_ref_designator_patterns.py`
（`ref_designator_patterns.py` は `region_detector.py` の領域名候補フィルタ専用として
存続）へ移設。OVERVIEW.md・VERSION_HISTORY.md にも反映）

最終更新: 2026-07-27（REGION_DETECTION.md に、複数VIEWPORTを持つ組立図
〈`EE6892-455B.dxf`・`EE6892-916-01A.dxf`・`EE6888-339-03A.dxf`〉で
「図面枠が見つからない」となる既知の限界を追記。図面枠矩形がPaper Space
にのみ存在し複数VIEWPORT構成のため単純な矩形検出では対応できないことを
確認。Reference Designator抽出（既定モード）は既存のフォールバックで
正常動作することも確認済み。コード変更なし・ユーザー確認の上で対応見送り）

最終更新: 2026-07-27（REGION_DETECTION.md・TERMINAL_DETECTION.md に、「領域を検出」
ON時の出力Excelに `Total` シート（`Summary` 直後、全ファイル横断のラベル合計一覧）を
追加した機能追加を反映。従来は `領域一覧`・`領域別ラベル一覧` で代替する設計だったが、
ユーザー要望により追加した。TERMINAL_DETECTION.md の「領域モード（`Total`シートが無い）」
という記述も訂正。v1.9.19）

最終更新: 2026-07-26（モジュラリティ改善リファクタ v1.9.18: `select_layout_result()`
を `region_detector.py` から `common_utils.py` へ移設、`ref_designator.py` を
純粋パターン判定（`ref_designator_patterns.py`、新設）とDXF抽出パイプラインに分割
（`region_detector.py` との循環依存を解消）、`app.py` の809行単一 `app()` 関数を
`_render_*`/`_handle_*` の10関数へ分割。いずれもロジック変更なしの純粋な
コード移動で、全439件のテスト（Playwrightエンドツーエンドテスト2件含む）で
無回帰を確認。OVERVIEW.md・REGION_DETECTION.md・REF_DESIGNATOR.md に反映）

最終更新: 2026-07-26（REGION_DETECTION.md に、領域名候補の機器符号除外判定を
`filter_non_circuit_symbols()` から `ref_designator.py` ベース（CONFIRMED
reference designatorのみ除外、`unit_names`カテゴリは例外扱い）に変更した修正
（MFFX・CTCの誤除外バグ対応）を反映、v1.9.17）

最終更新: 2026-07-26（REGION_DETECTION.md に「領域選択しない」のOFF同期・
OVERVIEW.md に「除外符号」シート追加を反映、v1.9.16）

最終更新: 2026-07-26（REGION_DETECTION.md に、領域名称候補の出現頻度の集計
単位を「領域数」から「別フレーム数」に変更した修正を反映、v1.9.15）

最終更新: 2026-07-26（REGION_DETECTION.md に、`detect_drawing_frames()` の
高さの異なる複数枠が左辺を共有する場合の検出漏れ修正・入れ子領域の既知の限界
（両エスカレーション同時発動が必要）を反映、v1.9.14）

最終更新: 2026-07-26（REGION_DETECTION.md に、領域名称候補の並び順変更
（距離→中央→頻度→Tier）と既知の限界（`MPD RACK2 PANEL LIMIT`ケース）を反映、
v1.9.13）

最終更新: 2026-07-26（REF_DESIGNATOR.md に、ユーザー提供の速記記法による
追加除外/機器符号パターン（`ADDED_EXCLUSION_PATTERNS`/`ADDED_DESIGNATOR_PATTERNS`）
を反映、v1.9.12）

最終更新: 2026-07-23（REGION_DETECTION.md に、無名領域が1個だけの場合は
番号を付けず「no name」のみにする変更を反映、v1.9.11）

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
