# OVERVIEW.md — DXF-extract-labels 概要・アーキテクチャ

> 矩形領域抽出の詳細は [REGION_DETECTION.md](REGION_DETECTION.md)、
> 機器符号（候補）抽出パイプラインの詳細は [REF_DESIGNATOR.md](REF_DESIGNATOR.md)、
> 端子一覧抽出の詳細は [TERMINAL_DETECTION.md](TERMINAL_DETECTION.md)、
> バージョン履歴は [VERSION_HISTORY.md](VERSION_HISTORY.md) を参照。

## 概要

複数の DXF ファイルからテキストラベルを抽出し、Excel 形式で出力する Streamlit アプリ。
既定モードでは、図面枠内・図面情報欄外のラベルから Reference Designator（機器符号）
パターンに一致するものを「機器符号（候補）」として抽出し、未確定ラベルはユーザーが
選択して採用する（v1.6.0）。図番・タイトル抽出オプションも持つ。「UNIT内結線図」
図面から端子台の端子番号を抽出する「端子一覧を抽出」オプションも持つ（v1.8.0）。

---

## ディレクトリ構成

```
DXF-extract-labels/
├── app.py                      # Streamlit UI（ファイルアップロード・オプション・結果表示）
├── requirements.txt
├── utils/
│   ├── extract_labels.py       # DXF ラベル抽出コア（共有モジュール）
│   ├── ref_designator.py       # 機器符号（候補）抽出パイプライン（DXF-extract-labels 専用、v1.6.0）
│   ├── region_detector.py      # 矩形領域検出アルゴリズム（DXF-extract-labels 専用）
│   ├── terminal_detector.py    # 端子台(TB)矩形検出・端子番号抽出（DXF-extract-labels 専用、v1.8.0）
│   ├── excel_output.py         # Excel 出力生成（通常モード・領域モード両対応）
│   ├── decision_log.py         # 未確定ラベルの採用/非採用の判断ログ記録（DXF-extract-labels 専用、v1.7.0）
│   └── common_utils.py         # 共通ユーティリティ（共有モジュール）
└── tools/
    └── reference_designator_analyzer.py  # Reference Designator 抽出検討ツール（v1.6.0、判断ログ分析はv1.7.0、後述）
```

> `extract_labels.py` / `common_utils.py` は DXF-label-diff / DXF-diff-processor / DXF-tools 等と
> 同一ロジックを持つ共有モジュール。修正時は関連プロジェクトへの伝播が必要。
> `ref_designator.py` / `region_detector.py` / `terminal_detector.py` / `excel_output.py` は
> DXF-extract-labels 専用で共有対象外。`ref_designator.py` は `extract_labels.py` の
> `extract_text_from_entity()` / `_block_has_text_content()` のみ再利用し、図面枠・
> フォーマットブロック検出・ラベル収集の**アルゴリズム本体**は完全に自前実装する
> （`extract_labels.py`／`region_detector.py` の図面枠検出ロジックそのものを変更しない
> ため。詳細は「機器符号（候補）抽出パイプライン」節）。ただし `region_detector.py` が
> 提供する小さな汎用ヘルパー（`select_layout_result()`＝Model/Paper Space の
> 選択方針、`_label_rotation_angle()`＝90°回転判定）は `ref_designator.py`・
> `terminal_detector.py` の両方から再利用する（v1.8.4でレイアウト選択ロジックの
> 重複を解消する形で整理。ドメイン固有のアルゴリズム本体とは異なりレイアウト非依存の
> 汎用処理のため、独立性の原則を損なわない）。

### モジュール責務

| ファイル | 責務 |
|---------|------|
| `app.py` | Streamlit UI のみ（ファイルアップロード・オプション選択・結果表示） |
| `extract_labels.py` | DXF エンティティからのテキスト抽出・図番判別・タイトル抽出 |
| `ref_designator.py` | 機器符号（候補）パターン・除外リスト・確定リスト・図面枠/フォーマットブロック検出・候補/確定/未確定ラベルの分類と集計（v1.6.0、確定リストは v1.6.3、連動採用は v1.7.1）|
| `region_detector.py` | 図面枠検出・線分結合・閉領域（直交ポリゴン）探索・名称候補抽出・集計（`build_region_results` / `build_region_label_summary`。「以外も抽出」ON 時のみ使用）|
| `terminal_detector.py` | 端子台(TB)矩形検出（LINE+CIRCLE橋渡し）・ラベル-矩形対応判定・端子番号抽出・集計（`analyze_dxf_terminals` / `build_terminal_rows`。「端子一覧を抽出」オプション時のみ使用、v1.8.0）|
| `excel_output.py` | Excel ファイル生成（`create_excel_output` / `create_ref_designator_excel_output` / `create_region_excel_output`）|
| `decision_log.py` | 未確定ラベルの採用/非採用の記録（`build_entries` / `record` / `GitHubBackend` / `FileBackend`。v1.7.0）|
| `common_utils.py` | 機器符号フィルタリング（`filter_non_circuit_symbols`。領域名候補の除外判定にのみ使用）・ファイル保存・エラー処理 |

---

## アーキテクチャ

### データフロー

既定モード（「機器符号（候補）以外も抽出」OFF、v1.6.0）:

```
DXF ファイル複数アップロード
  → save_uploadedfile()                       # 一時ファイル保存
  → [ユーザーがオプション選択]
  → ref_designator.extract_ref_designator_data()  # ファイルごとに機器符号（候補）/未確定に分類
  → 「未確定ラベル」UI（st.data_editor、初期全OFF）→ 「選択完了」
  → create_ref_designator_excel_output() / create_region_excel_output()
  → st.download_button()
```

「機器符号（候補）以外も抽出」ON（従来の全量抽出。フィルタ・図面枠制限なし）:

```
DXF ファイル複数アップロード
  → save_uploadedfile()
  → [ユーザーがオプション選択]
  → process_multiple_dxf_files()    # 全ファイル一括処理（フィルタなし）
  → create_excel_output()           # Excel 生成
  → st.download_button()
```

### `process_multiple_dxf_files()` の内部フロー（「以外も抽出」ON 時のみ使用）

1. `ezdxf` で DXF を読み込み
2. MODEL_SPACE / PAPER_SPACE / BLOCKS の TEXT・MTEXT・INSERT エンティティを収集
3. MTEXT フォーマットコード（`\f`、`\H`、`\W`、`\C` 等）を除去
4. 円マーク（`¥`）をバックスラッシュ（`\`）に正規化（日本語環境対応）
5. 図面番号抽出（`XX0000-000-00X` パターン）
6. タイトル・サブタイトル抽出（`TITLE` ラベル周辺のテキスト）

> v1.6.0 でレイヤー選択・機器符号妥当性チェックの UI/機能を削除したため、
> `extract_labels()` の `selected_layers`/`validate_ref_designators` 引数は
> `app.py` から渡されなくなった（既定値 = 全レイヤー対象・チェックなし）。
> **v1.7.12 で `extract_labels.py`（`extract_title_and_subtitle()`）に変更が入った
> ため、DXF-diff-manager・DXF-visual-diff とのバイト一致は崩れている（2026-07-12
> 時点で未伝播）。伝播する場合は本節末尾の v1.7.12 の対策を移植し `diff -q`/`md5`
> で一致確認すること。**


### INSERT展開のスキップ最適化（`_block_has_text_content()`、2026-06 追加）

INSERT エンティティの展開（手順2）は `e.virtual_entities()`（変換・複製を伴う重い処理）
で行うが、手描き回路図ではテキストを持たないブロック（コネクタ等の記号）の INSERT が
非常に多い。`_block_has_text_content(doc, block_name, cache)` で「ブロックが
TEXT/MTEXT を含むか（ネストINSERTを再帰的にたどった先も含む）」をブロック名単位で
メモ化し、含まない INSERT は `virtual_entities()` を呼ぶ前にスキップする。判定不能時
（ブロックが見つからない等）は安全側（展開する）に倒すため、出力結果は変わらない。

サンプル161ファイル（DXF-diff-manager の回帰テストデータ）で最適化前後の `extract_labels()`
出力が完全一致することを確認済み（処理時間は計測環境で合計約10%短縮）。`DXF-diff-manager`
へ伝播済み（バイト一致、2026-06）。`DXF-visual-diff` も byte-identical な copy を持つため
将来伝播の検討対象（今回は未実施）。

### 図番・流用元図番の判別（`determine_drawing_number_types()`）

図番（比較先）と流用元図番（比較元）はラベル位置と座標から判別する。

- **図番**: ファイル名と一致する候補を優先。なければ `DWG No.` ラベル近傍／座標で判定。
- **流用元図番**: `流用元図番` ラベルに最も近い候補から判定（図番自身は除外）。

**重なったタイトルブロックへの対応（2026-06）**: 旧版と現行版の2つのタイトル
ブロック（INSERT）が同一座標に重なっている DXF では、流用元図番ラベルへの距離が
両ブロックで同一になり、旧ブロックの図番を誤って流用元として拾う問題があった
（例 `EE6888-602-01A.dxf`: `EE2505-602-26B` を誤抽出 → 正しくは `EE6492-602-02A`）。

対策として、各図番候補に**所属タイトルブロック（INSERT）のグループキー**（親
INSERT の handle）を付与し、図番がファイル名一致等で確定したら**同じグループ内**で
流用元図番を判定する。これにより別ブロック（旧版）の図番を拾わない。通常（ブロック
1つ）の挙動は不変。候補は `(図番, 座標, グループ)` 形式で、グループ無しの2要素
タプルとも後方互換。

> 回帰テスト: `tests/regression/test_drawing_number_types.py`

### タイトル・サブタイトルの抽出（`extract_title_and_subtitle()`）

`TITLE` ラベルの右側かつ `REVISION` より下にあるラベルを候補にし、Y座標で
グルーピングした上で「Y座標最上位グループ群のうちX座標最小」のグループを
タイトル、その直下（同じX範囲）をサブタイトルとする。

**重なったタイトルブロックへの対応（v1.7.9）**: 旧版と現行版の2つのタイトル
ブロックが同一座標に重なっている DXF では、タイトル行とサブタイトル行のY座標差
が `y_threshold`(10.0) 未満になるケースがあり、サブタイトル行もトップグループ
候補に入り込む。この状態で「X座標最小」のタイブレークを浮動小数点の実値の
大小だけで決めると、ノイズ（`373.0` vs `373.0000000000002`）次第でサブタイトル
行がタイトルとして選ばれてしまう問題があった（`EE6888-637-01A.dxf` 等）。

対策は2段構え:
1. **タイブレーク修正**: min_x が許容誤差(1.0)内で同一のグループが複数ある
   場合、その中でY座標が最も高い（＝最上段の）行を採用する。
2. **グループキーによる候補制限**: 図番判別（`determine_drawing_number_types()`）
   と同じ「所属タイトルブロック（INSERT）のグループキー」方式を流用。図番が
   確定したブロック（`main_drawing_group`）内に `TITLE` ラベルが存在する場合は、
   そのブロックのラベルのみを候補にする。ブロック化されずタイトルブロックが
   直接配置されている図面（グループ内に `TITLE` が無いケース）は、従来どおり
   全ラベルで判定する（安全側フォールバック）。

`determine_drawing_number_types()` は図番が属するグループキーを `main_group`
として返すよう拡張されており、`extract_labels()` がこれを
`extract_title_and_subtitle(..., main_drawing_group=...)` に橋渡しする。

**図面枠外ノイズ（位置記号・頁数）の除外（v1.7.12）**: 電気回路図以外（機構
部品図等）で、TITLE近傍のY座標グルーピング（許容誤差5.0）に、図面枠**外**に
置かれる位置記号（`F`/`L`/`H` 等の英大文字1字やグリッド参照番号）や、図番横の
頁数「1/1」（数字のみのラベルが2つ、斜線を挟んでフラクション状に配置）が
偶然近接して混入し、タイトル末尾に付着する・頁数がサブタイトルとして誤採用
される不具合があった（`EE2685-335-01D.dxf` 等。実データ94件中39件で発生を確認）。

原因は2経路:
1. 位置記号が TITLE 自身の行と同じ Y グループに混入 → タイトル末尾に付着
2. 混入したグループがサブタイトル側の候補としても選ばれる場合、末尾の英大文字
   1字だけを除去する既存ロジック（`is_single_uppercase_letter` によるサブタイトル
   末尾ストリップ）が働き、結果的にタイトルと同一内容のサブタイトルが生成される

対策（内容ベースではなく、位置記号については図面枠のジオメトリで判定）:
1. **枠外除外（幾何学的）**: `_titleblock_frame_bbox()` が `main_drawing_group`
   の指す タイトルブロック INSERT 内の図面枠（機器符号抽出＝`ref_designator.py`
   と同じ識別キー：lineweight=100 かつ color=7 の LINE、`virtual_entities()` で
   ワールド座標変換済み）のバウンディングボックスを検出する。`_is_titleblock_noise_label()`
   がこのbboxの外側にあるラベルを候補から除外する（位置記号・グリッド参照番号は
   常に枠の外側に配置されるため）。枠を検出できない図面（フォーマットブロック化
   されていない・非対応テンプレート等）は bbox が None になり、この条件を
   適用しない＝従来どおり内容ベースの判定にフォールバックする（副作用回避）。
2. **数字のみラベルの除外（内容ベース）**: `_is_titleblock_noise_label()` は
   frame_bbox の有無に関わらず、ラベルが数字のみ（半角・全角とも `str.isdigit()`
   で判定）の場合は常に候補から除外する。頁数「1/1」は枠内（右下隅、枠の右辺・
   下辺の両方に接する小矩形内）に配置されるため幾何学的な枠外判定だけでは
   除外できないが、内容が数字のみである点は普遍的なため、こちらは常時適用する。
3. **タイトル＝サブタイトルの安全策**: 上記1・2を適用してもなお同一内容が
   タイトル・サブタイトル両方に選ばれた場合に備え、`subtitle_text == title_text`
   のときはサブタイトルを `None` にする最終ガードを追加。

実データ94件（`339_Unit内結線図`/`405_展開接続図`）で before/after 比較を実施し、
44件が意図通り修正・50件（正規のサブタイトルを含む）は出力不変であることを確認。
全pytestスイート289件pass。

> 回帰テスト: `tests/regression/test_title_extraction.py`

---


## Excel 出力仕様

既定モード（機器符号（候補）パイプライン。`create_ref_designator_excel_output`）:
シート順 `Summary` → `Total` → `<ファイル名>...`

| シート | 内容 |
|--------|------|
| `Summary` | 全ファイルの集計（図面枠内ラベル数・機器符号数〔未確定ラベルUIで採用した件数〕・未採用件数・図番等）。ファイル名セルは各ファイルシートへの内部ハイパーリンク |
| `Total` | 全ファイルの機器符号（確定パターン一致の自動採用分＋未確定ラベルUIで採用したもの）をユニーク集計した合計一覧（ラベル・個数・図番）|
| `<ファイル名>` | ファイルごとの機器符号一覧（確定パターン一致の自動採用分＋未確定ラベルUIで採用したもの。ラベル・個数）|

「機器符号（候補）以外も抽出」ON（`create_excel_output`）:
シート順 `Summary` → `Total` → `<ファイル名>...`（フィルタなし・図面枠制限なしの全ラベル）

シート名: 元ファイル名の拡張子除去、31 文字以内に切り詰め。

**TB List シート（「端子一覧を抽出」ON、v1.8.0）**: `Total` シート直後
（領域モードでは `領域別ラベル一覧` 直後）に `TB List` シート（`端子台` /
`端子No.` / `図番`）を追加。タイトルが「UNIT内結線図」の図面を対象に、
`TB`で始まりその直後に英大文字・数字が続く端子台ラベルに対応する矩形
（LINE+CIRCLE橋渡し）内の端子番号を抽出する。「端子台」でユニークをとり、
複数ファイルにまたがる場合は端子番号・図番を統合する（同じ番号が複数回
登場する場合は `7(2)` のように件数を表示）。候補パターンには一致したが
対応する矩形が見つからないラベルは、末尾に空行を挟んで「端子検出不可」行
（端子No.列にラベル・図番列に図番）として記載する。詳細は
[TERMINAL_DETECTION.md](TERMINAL_DETECTION.md) を参照。

**ファイル順序**: Summary シートの行順・各ファイルシートの並びとも、アップロード順
（dict挿入順）ではなくファイル名昇順（`sorted()`）（v1.7.9）。UI側の表示順
（「領域の確認」「未確定ラベル」セクション）は v1.7.6/v1.7.7 で先に対応済みで、
これは Excel 出力側の対応。

**Total シートの図番列（v1.7.10）**: `Total` シート（`create_excel_output` /
`create_ref_designator_excel_output` の両方）の C 列に、そのラベルが登場する
ファイルの図番一覧を追加。ラベル集計の際、ファイルごとに
`info.get('main_drawing_number') or os.path.splitext(filename)[0]`
（図番未抽出時はファイル名〔拡張子なし〕にフォールバック、`build_region_label_summary`
と同じ方式）を識別子とし、ラベル→識別子集合の逆引きマップを構築する。複数ファイル
にまたがるラベルは `', '.join(sorted(...))`（既存の「領域」列結合と同じ方式）で
結合する。`領域別ラベル一覧`（`create_region_excel_output`）は元々ファイル横断の
図番列を持つため対象外。

**半角正規化**: 出力ファイルに記録するラベル・機器符号・矩形領域名称は
すべて `normalize_width()`（NFKC、v1.5.25）／ `ref_designator.normalize_label()`
（v1.6.0、NFKC+前後空白除去）で半角へ統一してから集計する。図面上の表記が
半角（`CN1`）でも全角（`ＣＮ１`）でも同じ語は同じ行に合算される（かな・漢字は不変）。

**全シートの先頭行固定（v1.8.0）**: `Summary`/`Total`/各ファイルシート/
`領域一覧`/`領域別ラベル一覧`/`TB List` の全シートに `freeze_panes(1, 0)`
を適用し、スクロールしてもヘッダー行が常に表示されるようにした
（`excel_output._write_header_row()` に集約。ヘッダー行を手書きしている
`領域別ラベル一覧` のみ個別に `freeze_panes` を呼ぶ）。

### ラベル集計コード例

```python
counter = Counter(labels)
label_data = [{'ラベル': lbl, '個数': counter[lbl]} for lbl in sorted(counter)]
```

---

## オプション仕様

| オプション | 説明 |
|-----------|------|
| 機器符号（候補）以外も抽出 | OFF（既定）: 図面枠内・図面情報欄外のラベルのうち機器符号パターンに一致し除外パターンに非該当のものを「未確定ラベル」として全件レビューUIに表示し、チェックして採用したものだけを出力する（自動確定はしない）。ON: 図面枠制限・フィルタなしで全ラベルを抽出（従来動作） |
| 図面番号抽出 | `XX0000-000-00X` パターンの図番を Summary に記録 |
| タイトル・サブタイトル抽出 | `TITLE` 近傍テキストを解析して Summary に記録 |
| ソート順 | 昇順 / 降順 / なし |
| 領域を検出 | OFF（既定）: 通常モード。ON: 「領域検出の詳細設定」「領域を検出」ボタン「領域の確認」セクションを表示し、図面枠内の矩形（直交ポリゴン）領域を検出して領域内ラベルに領域名を付与できる（独立セクション「領域選択オプション」。後述、v1.9.4でチェックボックス化）|
| 端子一覧を抽出 | タイトルが「UNIT内結線図」の図面（サブタイトルが「TB COMPONENT」の図面を除く）を対象に、`TB`で始まりその直後に英大文字・数字が続く端子台ラベルに対応する矩形（LINE+CIRCLE橋渡し）を検出し、矩形内の端子番号を「端子台」ユニークで `TB List` シートに出力（v1.8.0、詳細は [TERMINAL_DETECTION.md](TERMINAL_DETECTION.md)）|

> v1.6.0 で「特定のレイヤーのみを処理する」（未使用のため）・「機器符号妥当性チェック」
> （未確定ラベルUIでの人手選択に置き換え）の UI・機能を削除した。

---


## セッション状態

| キー | 型 | 内容 |
|------|-----|------|
| `excel_result` | bytes | 生成した Excel バイナリ（通常・領域モード共用）|
| `output_filename` | str | ダウンロードファイル名 |
| `is_region_mode` | bool | 最後に実行したのが領域モードか通常モードかの区別 |
| `processing_settings` | dict | 適用したオプション設定 |
| `results` | dict | ファイル名 → (labels, info) の辞書（通常モード）|
| `region_analyses` | dict | ファイル名 → `analyze_dxf_regions()` 結果（領域モード）。「図面番号・タイトル・サブタイトルを抽出」ON で検出した場合は `main_drawing_number`/`title`/`subtitle` キーを追加格納（v1.5.27）|
| `saved_region_cfg` | dict | 「設定完了」で確定した領域検出詳細設定 |
| `region_cfg_is_saved` | bool | 詳細設定が保存済みかどうか（✅キャプション表示制御）|
| `rc_<fname>_<reg_id>_<i>` | bool | 領域名チェックボックスの選択状態（ラジオ動作）|
| `region_results_summary` | dict | 領域抽出のサマリー表示用（rows/named キーは除いた軽量版）|
| `download_done` | bool | 「Excelをダウンロード」クリック後 True。ボタン配色の切り替えに使用（v1.5.29）|
| `uploader_version` | int | `file_uploader` の `key` に使うカウンタ。「新しい抽出を開始」でインクリメントしウィジェットを再生成することでアップロード済みファイルをクリアする（v1.5.29）|
| `ref_pending` | dict | 機器符号（候補）パイプラインの「未確定ラベル」選択待ちデータ（ファイル名 → `extract_ref_designator_data()` 結果 + 図番等）。「選択完了」まで保持し、確定後に削除（v1.6.0）|
| `ref_pending_mode` | str | `ref_pending` が通常モードか領域モードかの区別（`'normal'`/`'region'`、v1.6.0）|
| `ref_results_summary` | dict | 機器符号（候補）パイプライン（通常モード）のサマリー表示用（v1.6.0）|
| `unclassified_editor_<ver>_<fname>_<0..N>` | DataFrame | 「未確定ラベル」`st.data_editor` の描画結果（v1.6.0。ファイルごとに `UNCLASSIFIED_ROWS_PER_TABLE`（既定10行）ごとに分割、テーブル数はラベル数に応じて可変。キーに `unclassified_ver` を含むためチェック変更のたびに別ウィジェットとして再生成される、v1.7.1）|
| `unclassified_checked` | dict | 「未確定ラベル」の採用チェック状態の正本（ファイル名 → {ラベル: bool}）。data_editor はここから初期値を作るだけで、ウィジェット自身の state は直接書き換えない（v1.7.1）|
| `unclassified_ver` | int | `unclassified_checked` を書き換えるたびにインクリメントし、`unclassified_editor_*` の `key` に使うことで data_editor を別ウィジェットとして再生成する（v1.7.1、詳細は「連動採用（兄弟ラベル）」節）|
| `decision_log_result` | dict | 判断ログの記録結果（`ok`/`message`/`fallback_csv`）。抽出結果セクションで表示し、「新しい抽出を開始」または再度「ラベルを抽出」時にクリアする（v1.7.0）|
| `terminal_results` | dict | 「端子一覧を抽出」ON時の `analyze_dxf_terminals()` 結果（ファイル名 → 結果dict）。「ラベルを抽出」時に一度だけ計算し、既定モード（Excel生成が「選択完了」まで遅延する）でも使えるよう保持する。「新しい抽出を開始」または再度「ラベルを抽出」時にクリアする（v1.8.0）|

---

## 依存パッケージ

```
streamlit>=1.40.0, ezdxf>=1.4.2, pandas>=2.0.0
xlsxwriter>=3.0.0, openpyxl>=3.0.0, requests>=2.31.0
```

`requests` は判断ログの GitHub Contents API 呼び出し（`utils/decision_log.py`）に
使用する（v1.7.0 で追加）。

---

## 既知の制限

| 制限 | 詳細 |
|------|------|
| MTEXT 複雑書式 | v1.5.1 で ezdxf `plain_mtext()` ベースへ移行し、`\S` 分数・`%%c`/`%%d`/`%%p`・キャレットシーケンス等も処理可能になった。ごく稀な独自書式は残る可能性がある |
| タイトル抽出精度 | `TITLE` ラベルの配置パターンに依存するため、図面レイアウトが異なると誤抽出する場合がある |
| 大量ファイル | 10 ファイル超でメモリ消費が大きくなる可能性がある |
| フォーマットブロック非依存の図面情報欄（v1.6.0） | 図面情報欄がフォーマットブロック（INSERT）経由でなく modelspace に直置きされている DXF では、構造的除外が働かず図面情報欄のラベルが未確定ラベルに残る可能性がある（実データ18サンプルでは全て INSERT 経由だったため未検証） |
| **LINE 矩形の部品輪郭（未対応課題）** | `DE5434-563-03A.dxf` のように、lw=25/color=2 の LINE エンティティ 4 本で形成された「細い閉じた矩形（部品輪郭）」（実例: x=81~90, y=98~394.5、幅9単位）が領域境界線と同じ属性を持つ場合、アルゴリズムが部品輪郭と領域境界線を区別できない。部品輪郭の両端が領域中央の縦仕切りを形成するケースでは、子領域の境界が「正確な直線」にならず、左右の子領域を閉領域として独立して検出できない（結果として合体した親領域のみが検出される）。**将来対応案**: lw=25/color=2 が形成する「縦横比が高い閉矩形」を部品輪郭として自動判定し検出対象から除外する。あわせて、合体親領域に付与する名称を「子領域の名称候補を除外した上で、底辺中央により近いラベルを優先して探索する」ロジックを追加することで、合体親に本来の領域名（例: `FX CHAMBER`）を正しく割り当てられるようになる。|

---

## 機能拡張ポイント

| テーマ | 実装アプローチ |
|--------|--------------|
| Excel 列幅の自動調整 | xlsxwriter の `set_column()` でラベル文字列長に応じた幅を計算 |
| 進捗バー | `process_multiple_dxf_files()` をファイル単位でループ化して `st.progress()` で表示 |
| カスタムフィルタパターン | UI 上でユーザーが正規表現を入力できるようにする |
| 重複ラベル警告 | 同一ファイル内の高頻度ラベルを Summary で強調表示 |

---

## システム要件

- Python 3.9 以上
- 最小 512MB RAM（一般的な DXF ファイルの場合）
- JavaScript が有効な最新のウェブブラウザ

---

## トラブルシューティング

| 症状 | 確認事項 |
|------|---------|
| DXF ファイルの読み取りエラー | ファイルが有効な DXF 形式か確認。破損・パスワード保護がないか確認 |
| ラベルが抽出されない | TEXT/MTEXT エンティティの有無を確認。「機器符号（候補）以外も抽出」ONで全ラベルを見て切り分ける |
| 処理タイムアウト | ファイル数を減らして再試行 |
| 「図面枠内の制約なしに全ラベルを抽出しました」と表示される（機器符号（候補）パイプライン） | 「領域検出の詳細設定」の「図面枠の太さ」が実際の図面枠の lineweight と一致しているか確認（color=7 も必須条件）。Model Space に実際のラベル・図形がある図面では、図面枠が Paper Space レイアウトにしか無い場合、意図的にこのメッセージを出し図面枠フィルタなしでの抽出にフォールバックする（Model Space と Paper Space は独立した座標系のため、Paper Space の図面枠を Model Space のラベルに適用すると正しいラベルまで誤って除外されるため。v1.8.2で確定した仕様）。抽出自体は正常に行われるため、失敗ではない（v1.8.3で「見つかりませんでした」という失敗を思わせる文言から変更） |
| 「領域探索を実施することができませんでした」と表示される（矩形領域抽出） | 領域検出には図面枠が必須（面積比の基準となるため）で、機器符号（候補）パイプラインのような制約なしへのフォールバックが無い。「図面枠の太さ」の設定を確認するか、対象ファイルの図面枠が Paper Space レイアウトにしか無い場合は領域検出の対象外となる（v1.8.3で文言変更） |
| 期待した機器符号が候補に出ない | 「未確定ラベル」一覧に出ていれば手動で採用できる。除外パターン（`ExclusionPatterns` シート）に該当していないか確認 |

---


## DXF-label-diff との違い

| アプリ | 用途 |
|--------|------|
| DXF-label-diff | 2 つの DXF ファイルのラベルを比較し差分を抽出 |
| DXF-extract-labels | 単一または複数の DXF ファイルからラベルを抽出 |

両プロジェクトは同一の `utils/extract_labels.py` を共有している（バイト一致、伝播ルールは
`Tools/CLAUDE.md` 参照）。`utils/common_utils.py` は v1.6.0 で本プロジェクト側のみ
`validate_circuit_symbols()` を削除したため、他プロジェクトのコピーとは差異がある
（伝播前に `diff`/`md5` で確認すること）。

---


---

## ライセンス

株式会社 RDPi 所有。特定顧客向け開発品につきコピー・改版禁止。

---

最終更新: 2026-07-14 (v1.9.0) — 詳細な変更履歴は [VERSION_HISTORY.md](VERSION_HISTORY.md) 参照。
