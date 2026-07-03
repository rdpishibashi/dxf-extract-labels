# TECHNICAL.md — DXF-extract-labels

## 概要

複数の DXF ファイルからテキストラベルを抽出し、Excel 形式で出力する Streamlit アプリ。
機器符号フィルタリング、妥当性チェック、図面番号・タイトル抽出、レイヤー選択などの高度なオプションを持つ。

---

## ディレクトリ構成

```
DXF-extract-labels/
├── app.py                      # Streamlit UI（ファイルアップロード・オプション・結果表示）
├── requirements.txt
├── utils/
│   ├── extract_labels.py       # DXF ラベル抽出コア（共有モジュール）
│   ├── region_detector.py      # 矩形領域検出アルゴリズム（DXF-extract-labels 専用）
│   ├── excel_output.py         # Excel 出力生成（通常モード・領域モード両対応）
│   └── common_utils.py         # 共通ユーティリティ（共有モジュール）
```

> `extract_labels.py` / `common_utils.py` は DXF-label-diff / DXF-diff-processor / DXF-tools 等と
> 同一ロジックを持つ共有モジュール。修正時は関連プロジェクトへの伝播が必要。
> `region_detector.py` / `excel_output.py` は DXF-extract-labels 専用で共有対象外。

### モジュール責務

| ファイル | 責務 |
|---------|------|
| `app.py` | Streamlit UI のみ（ファイルアップロード・オプション選択・結果表示） |
| `extract_labels.py` | DXF エンティティからのテキスト抽出・図番判別・タイトル抽出 |
| `region_detector.py` | 図面枠検出・線分結合・閉領域（直交ポリゴン）探索・名称候補抽出・集計（`build_region_results` / `build_region_label_summary`）|
| `excel_output.py` | Excel ファイル生成（`create_excel_output` / `create_region_excel_output`）|
| `common_utils.py` | 機器符号フィルタリング・妥当性チェック・ファイル保存・エラー処理 |

---

## アーキテクチャ

### データフロー

```
DXF ファイル複数アップロード
  → save_uploadedfile()             # 一時ファイル保存
  → get_layers_from_dxf()           # 第 1 ファイルのレイヤー一覧取得
  → [ユーザーがレイヤー/オプション選択]
  → process_multiple_dxf_files()    # 全ファイル一括処理
  → create_excel_output()           # Excel 生成
  → st.download_button()
```

### `process_multiple_dxf_files()` の内部フロー

1. `ezdxf` で DXF を読み込み
2. MODEL_SPACE / PAPER_SPACE / BLOCKS の TEXT・MTEXT・INSERT エンティティを収集
3. MTEXT フォーマットコード（`\f`、`\H`、`\W`、`\C` 等）を除去
4. 円マーク（`¥`）をバックスラッシュ（`\`）に正規化（日本語環境対応）
5. レイヤーフィルタ適用（`selected_layers` 引数）
6. 機器符号フィルタ適用（正規表現パターン）
7. 図面番号抽出（`XX0000-000-00X` パターン）
8. タイトル・サブタイトル抽出（`TITLE` ラベル周辺のテキスト）

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

---

## Excel 出力仕様

シート順: `Summary` → `Total` → `<ファイル名>...` → `Invalid`（オプション有効時のみ）

| シート | 内容 |
|--------|------|
| `Summary` | 全ファイルの集計（総抽出数・フィルタ除外数・最終ラベル数・図番等）。ファイル名セルは各ファイルシートへの内部ハイパーリンク |
| `Total` | 全ファイルのラベルをユニーク集計した合計一覧（ラベル・個数）|
| `<ファイル名>` | ファイルごとのラベル一覧（ラベル・個数）|
| `Invalid` | 妥当性チェック失敗の機器符号を集計（機器符号・個数・ファイル名）。オプション有効時のみ作成 |

シート名: 元ファイル名の拡張子除去、31 文字以内に切り詰め。

**半角正規化（v1.5.25）**: 出力ファイルに記録するラベル・機器符号・矩形領域名称は
すべて `normalize_width()`（NFKC）で半角へ統一してから集計する。図面上の表記が
半角（`CN1`）でも全角（`ＣＮ１`）でも同じ語は同じ行に合算される（かな・漢字は不変）。

### Invalid シートのカラム仕様

| カラム | 内容 |
|--------|------|
| 機器符号 | 非適合と判定された機器符号 |
| 個数 | 全ファイルを通じた出現合計数 |
| ファイル名 | その機器符号が含まれるファイル名（`, ` 区切り）|

### ラベル集計コード例

```python
counter = Counter(labels)
label_data = [{'ラベル': lbl, '個数': counter[lbl]} for lbl in sorted(counter)]
```

---

## オプション仕様

| オプション | 説明 |
|-----------|------|
| 機器符号のみ抽出 | 正規表現でパターン一致するラベルのみを抽出 |
| 機器符号妥当性チェック | 標準フォーマット非適合の機器符号を Invalid シートに出力 |
| 図面番号抽出 | `XX0000-000-00X` パターンの図番を Summary に記録 |
| タイトル・サブタイトル抽出 | `TITLE` 近傍テキストを解析して Summary に記録 |
| ソート順 | 昇順 / 降順 / なし |
| レイヤー選択 | 処理対象レイヤーを multiselect で絞り込み |
| 矩形領域抽出 | 図面枠内の矩形（直交ポリゴン）領域を検出し、領域内ラベルに領域名を付与（独立セクション「領域選択オプション」。後述）|

---

## 矩形領域抽出（領域選択オプション）

電気回路 DXF 内の閉領域（直交ポリゴン。四角形に限らない）を検出し、領域内ラベルに
領域名を付与する機能。`utils/extract_labels.py` の `analyze_dxf_regions()` /
`assign_region_labels()`、`app.py` の「領域選択オプション」セクションで実装。

### 処理フロー（UI）

```
「領域を検出」ボタン  → analyze_dxf_regions() を各ファイルに実行（session_state 保持）。
                        「図面番号・タイトル・サブタイトルを抽出」オプションが ON の
                        場合は extract_labels() も各ファイルに実行し、図番・タイトル・
                        サブタイトルを analysis dict に格納（v1.5.27）
「領域の確認」セクション → ファイル名見出しの直下に「図番：… / タイトル：… /
                           サブタイトル：…」をキャプション表示（上記オプション ON 時
                           のみ。空の項目は省略）。
                           各領域の名称候補をチェックボックス（ラジオ動作）で確定。
                           他の図面/領域で選択済みの名称があればそれをデフォルト選択
                           （初回描画時のみ）。チェック後にユーザーが選択を変更した
                           場合も、同じ名称候補を持つ他領域（候補2件以上の領域に限る）
                           には `_on_change_radio`（v1.5.5）が即時に選択を伝播する。
                           名称候補なし → Excel 出力時に「no name #」（連番）で自動命名。
「ラベルを抽出」ボタン → build_region_results() → create_region_excel_output()
                        （領域検出なし時は process_multiple_dxf_files() + create_excel_output()）
```

> 「ラベルを抽出」ボタンは領域モード・通常モード共用。領域検出済み（`region_analyses` あり）
> かどうかで内部処理を切り替え、共通の「Excelをダウンロード」ボタンで出力する。

### ボタンの色分け（v1.5.26）

3つの操作ボタン（「領域を検出」「ラベルを抽出」「Excelをダウンロード」）は、
`session_state` の完了状況から `type`（primary=青 / secondary=白）を動的に切り替える。
`disabled=` は使わない（設定を変えて何度でも撮り直せるよう、常にクリック可能なままにする）。

```python
detect_done = 'region_analyses' in st.session_state
extract_done = bool(st.session_state.get('excel_result'))
detect_btn_type  = "secondary" if (detect_done or extract_done) else "primary"
extract_btn_type = "secondary" if extract_done else "primary"
```

- 初期状態（ファイルアップロード直後）: 「領域を検出」「ラベルを抽出」の両方が青
  （領域検出は任意手順のため、どちらから始めてもよい）。
- 「領域を検出」実行後: 「領域を検出」は白、「ラベルを抽出」は引き続き青
  （まだ必須の次操作が残っているため）。
- 「ラベルを抽出」実行後（領域検出の有無を問わず）: 両方白になり、
  「Excelをダウンロード」（常に `type="primary"`）が青になる。

各ボタンの `if st.button(...):` 処理ブロックの末尾で `st.rerun()` を呼んでいる。
ウィジェットは処理ブロックより前に描画されるため、`st.rerun()` を呼ばないと
クリックした瞬間の画面には古い状態の色が残ったまま表示され、次に何か別の操作をするまで
色が切り替わって見えないため（ユーザー報告により追加）。

### 検出アルゴリズム

1. **図面枠検出**: `lineweight=100` の線分で囲まれた枠を検出（`detect_drawing_frames`）。
   枠内のみ処理対象。複数図面は枠が横並び。枠面積は全図面同一。
2. **領域境界線**: `lineweight=25` かつ `color=2`(ACI黄) の線分。**かつ線種(linetype)が
   実質的に Continuous（実線）であること（`_is_continuous_linetype`、v1.5.10）**。
   `linetype='ByLayer'` の場合はレイヤーの既定線種まで解決する。PHANTOM（二点鎖線）等の
   装飾的な線種は、lineweight/color が境界線条件に一致していても閉領域の壁を表すもの
   ではないため除外する（`EE6313-546-01E.dxf` で、実体の小さな矩形`MX CHAMBER`
   〈handle 21AB/21AC/219A/219E、Continuous〉の周囲に、別の handle
   〈21AE/21A1/21A9/2198等、PHANTOM〉で描かれた二点鎖線の矩形が重なっており、これも
   境界線として誤認識し、実体矩形を「くり抜いた」形状の存在しない領域が誤検出される
   不具合をユーザーが報告。DXF-viewer で座標リストを確認した際、抽出された境界の一部が
   実体の直線ではなく二点鎖線だったことから発覚）。
3. **共線セグメント結合**（`_merge_collinear`）:
   - レベル座標一致は厳密（`merge_level_tol=0.5`）。別レベルの線（別矩形）を結合しない。
   - **ギャップ橋渡しは既定で縦線分のみ**（`bridge_vertical_gaps=True` /
     `bridge_horizontal_gaps=False`）。部品は縦線分だけを途切れさせるため。
   - **コーナー相手判定**: 縦ギャップの両端のどちらかに横線分の端点が一致する場合
     （＝境界が折れるステップ）は橋渡ししない（`corner_tol=0.5`）。部品の切れ目は
     コーナー相手が無いので橋渡しする。これにより閾値バンド無しで段差と部品切れ目を区別。
   - 縦ギャップが `CIRCLE`（接続点）で繋がる場合は橋渡ししない（配線ループ）。
   - **横線分ギャップ橋渡し（v1.5.4・フォールバックのみ）**: `_detect_regions` は
     `bridge_horizontal_gaps=True` 指定時、縦線分の端点を（x/y を入れ替えて）コーナー
     相手として用い、縦ギャップ橋渡しと全く同じ安全条件（コーナー相手無し・CIRCLE無し）
     で横線分のギャップも橋渡しする。既定では無効。`analyze_dxf_regions` は、通常の
     検出（縦線分のみ橋渡し）で閾値超え候補がゼロ **かつ** `_is_globally_rotated()` が
     真（ラベルの過半数が90°回転＝図面全体が90°回転して描かれている）の場合のみ、
     このフォールバックを有効化する。「検出ゼロ件」だけを条件にすると、通常向きの
     図面で偶然ゼロ件になったときに無関係な隣接矩形を誤って結合する副作用があるため、
     回転判定を明示的な条件として併用する（ユーザー指摘により条件を追加）。
4. **閉領域検出**（`_find_rectilinear_faces`）: 接続は**線分の端点が相手の線分に乗る
   箇所（角・T字）のみ**で作る（中ほど同士の交差では繋がない／`face_snap=0.1`）。
   平面グラフの面探索（half-edge）で閉路を列挙。

   **行き止まり枝（dangling edge）の除去（v1.5.7）**: 境界線と同じ線種
   （lineweight=25/color=2）を持ちながら、どこにも閉じていない線分（次数1の
   ノードに繋がる枝）があると、半面探索はその枝を折り返すしかなく（次数1の
   ノードでは戻る辺が1本しかないため）、生のポリゴンに「同じ頂点が2回連続する」
   アーティファクトを生む（実例: `EE6313-546-01E.dxf` の `頂点の座標` リストに
   `(660.53, 129.56)` が2回連続して現れる不具合として報告。原因は handle
   `214F`/`2199` の2本の短い枝線が、本来繋がるべき相手まで約5単位届かず行き止まり
   になっていたこと）。面探索前に次数1のノードとその辺を再帰的に除去する2-core
   抽出を追加（戻り値が `faces` 単独から `(faces, dangling_branches)` に変更）。

   **枝（連結成分）単位のグルーピングと領域単位への絞り込み（v1.5.8）**: 当初
   実装は除去した辺をフレーム単位の1本のフラットなリストとして報告していたため、
   (1) 1本の枝が複数の短い線分の連なりで構成される場合に別々の行として現れる、
   (2) その領域の探索とは無関係な部品・他領域の枝まで同じリストに混在する、
   という2つの問題があった（実例: `EE6313-546-01E.dxf` で「行き止まり枝158件」
   と表示され、特定の領域に関係する枝を見分けられないとユーザーから指摘）。

   対策として `_find_rectilinear_faces` の次数1ノード除去ループを、除去した辺
   `(leaf_key, other_key)` を即座に座標へ変換するのではなく、まず Union-Find で
   連結成分（＝1本の枝）にまとめ、各枝について「現存する境界グラフ（除去後の
   `adj`）に含まれるノード」を探して取り付け点（`attachment`）を1つ求める
   （`_resolve_dangling_handles` も枝単位に対応し、枝を構成する全セグメントの
   延長線上にある全エンティティを `entities` にまとめる）。`analyze_dxf_regions`
   は各枝の `attachment` 座標が各領域の `polygon` 境界上（許容誤差
   `max(face_snap, 0.5)`）に乗るかどうかで、その領域の `dangling_edges` に
   絞り込んで割り付ける（戻り値は領域ごとの `regions[i]['dangling_edges']` に
   変更、フレーム単位のフラットなトップレベル `dangling_edges` キーは廃止）。
   `app.py` の表示もファイル単位から各領域カードの直下（📐ボタンの下）に移動。

   結果として `EE6313-546-01E.dxf` の図面1/領域1（最大の領域）は、ちょうど2本の
   枝（handle `214F` の単独枝と、`2199`→`21AD`→`219B`→`21AA`→`219F`→`21A7` が
   1本に連結した枝）だけが報告されるようになった。`21A7` は領域の上端境界
   （`150.22≦x≦660.53`）としても使われている1本のLINEで、その延長
   （`660.53≦x≦812.24`）が行き止まりになっている部分のため、同じ枝の構成
   エンティティとして自然に含まれる。他の4領域、および無関係な部品（小さな
   端子ボックス等）の枝は、いずれの領域の境界にも取り付かないため報告されない。

   **副次効果（同一領域の重複検出バグも解消）**: 行き止まり枝の往復は面積には
   正味ゼロしか寄与しないため、同一の物理境界が「綺麗な内側面」と「枝の往復で
   座標が汚れ bounding box が変わった外側面」の2つの別領域として重複検出される
   ケースがあった（座標の汚れにより既存の bbox 重複除外をすり抜けていた）。除去後は
   両者が同一 bbox になり正しく1領域に統合される（`EE6313-546-01E.dxf`:
   regions 6→5。汚れた版は迂回経路上の無関係なラベルまで誤って名称候補に
   取り込んでいたため、名称候補も正しくなった）。
5. **絞り込み**: 単独の閉領域は面積 ≥ 図面枠面積 × `area_ratio`(0.20)。**同名の複数
   ピース（≥2）の合算**が ≥ 図面枠面積 × `group_area_ratio`(0.10) なら、その名称を
   第1図面で「ターゲット名称」とし、他図面では面積不問で採用（例 MPD RACK2＝内側+外側）。
   タイトルブロック枠（図番・TITLE 等を含む）と、境界に接続点(円)を持つ領域は除外。
6. **領域名候補**（`region_name_candidates`、優先順位＝Tier制 v1.5.9）: ラベルは
   優先順位（Tier）→距離の順に提示し、デフォルトは最有力候補（`default_name`）。
   UI のチェックボックスでユーザーが上書き確定できる。

   - **Tier 1**: 下端横エッジ最近傍（図面全体が90°回転している場合は右端/左端の
     いずれか一方の縦エッジ。後述）。
   - **Tier 2**: 上端横エッジ最近傍（回転時はもう一方の縦エッジ）。**v1.5.27 から、
     L字型等の非矩形ポリゴンの「切り欠き部の下向き横エッジ」（最下端レベル以外に
     ある、直上が領域内・直下が領域外のエッジ。`_notch_bottom_edges`）も Tier2 の
     探索対象に含める**（後述）。
   - **Tier 3**: Tier1/2 のいずれでも候補がゼロの場合のみ、ポリゴン境界全体
     （任意の辺）への最短距離でフォールバック評価する（`_dist_point_to_polygon`）。

   いずれの Tier も `name_min_dist`(1.0)〜`name_max_dist`(10.0) の範囲のみ採用し、
   `min_dist` 未満（境界線分上＝コネクタ符号等が偶然線上に乗っただけの無関係な
   ラベル。例 `CN24POW04`/`CN24POW05`）は除外する。ラベル自体の条件は変更なし
   （英字3字以上・英小文字を含まない・`NOTE`/`☆`を含まない・機器符号候補でない。
   ただし `RACK` を含む語は名称として残す）。同じテキストが複数 Tier・複数距離で
   見つかった場合は最も優先度の高い Tier・距離のものを残す。

   **90°回転時の Tier1/2 の対応（`_rotated_edge_roles`、v1.5.9）**:
   図面全体が90°回転して描かれているファイル（ラベルの大半が `text_direction`/
   `rotation` で90°回転）では、領域名が下端/上端の横エッジでなく左右いずれかの
   縦エッジ脇に置かれる。`_label_rotation_angle()` の符号付き角度（-180°〜180°）
   の多数派が +90° 付近か -90° 付近かを判定し、
   - 回転角+90°が多数派（実例: `DE5434-553-10B.dxf`） → Tier1=右端、Tier2=左端
   - 回転角-90°が多数派 → Tier1=左端、Tier2=右端（左右反転。ユーザー確認による
     推奨対応。実例は未確認＝今後の図面確認で変更の可能性あり）
   - 回転は検出されたが方向が判定できない（角度が分散）場合は `rotated_edge_roles=None`
     として通常通り下端/上端で評価する。
   図面枠・領域境界線自体の検出（lineweight/color）は回転の影響を受けない。

   **複数行MTEXTは分割しない**: MTEXT内の `\P`（段落区切り）で区切られた複数行
   （例: `"CN I/F B.D TYPE3\P(CN-IF3-1A)"`）は同一エンティティ（同一 handle）に属する
   ため、`clean_mtext_format_codes` の既定どおり改行をスペースに変換した1つの
   結合済み文字列（`"CN I/F B.D TYPE3 (CN-IF3-1A)"`）のまま名称候補にする
   （ユーザー確認済み。一時的に行ごとに分割する実装を試したが、同一エンティティ内の
   行は分割しないという方針が正しいとの確認を得て撤回した）。

   **入れ子/隣接領域の選択同期バグの修正（v1.5.9）**: `app.py` は「他の図面/領域で
   選択済みの名称があれば、それを候補に持つ自分のデフォルトにも引き継ぐ」同期
   （`selected_elsewhere`、v1.5.3/v1.5.5、複数ピース合算の MPD RACK2 等を想定）を
   持つが、入れ子/隣接する2領域が互いの候補リストに相手の名称を（優先度の低い候補
   として）含む場合、片方が先に確定した名称を相手側が誤って引き継いでしまう不具合が
   あった（ユーザー報告: `EE6313-546-01E.dxf` の図面1/領域1,2 が同じ選択に同期され、
   本来は領域1=`B CHAMBER`、領域2=`BAKE HEATER UNIT RX` で別々が正しい）。各領域の
   `default_name_tier`（`region_name_candidates()` の Tier をそのまま反映）が
   1 または 2（確信度の高い自前の候補）の場合は同期で上書きしないように変更し、
   同期は元々の想定どおり Tier3（確信度の低い候補）の領域同士でのみ発動するように
   した。

   **重なる領域同士の同期禁止（`regions_overlap`、v1.5.11）**: 上記のTierガードは
   「自分自身がTier1/2の確信度の高い候補を持つ場合」に限られるため、ユーザーが
   手動でデフォルト以外の候補をチェックした場合（`_on_change_radio`）や、両領域が
   Tier3同士のケースでは依然として誤同期し得る。`EE6313-546-01E.dxf` の領域1
   （`B CHAMBER`、外側）・領域2（`BAKE HEATER UNIT RX`、内側、完全内包）で、
   デフォルトでない候補を手動選択すると、内包関係にあるもう片方の領域も同じ名称に
   同期されてしまうとユーザーから報告（当初は完全な内包のみを想定したが、部分的な
   重複も同期禁止の対象にすべきとの指摘により一般化）。`regions_overlap(poly_a,
   poly_b, tol=1.0)` を追加し、両ポリゴンの頂点＋各辺の中点をサンプル点として
   「一方のサンプル点が他方の内部に（境界から `tol` より離れて）存在するか」を
   判定する（完全な内包・部分的な重複の両方を検出。境界が接するだけ＝隣接して
   壁を共有する領域は重なりとみなさない）。`app.py` の `_on_change_radio`
   （手動選択の伝播）と `selected_elsewhere`（初期デフォルト同期）の両方に、
   同一ファイル内で重なりのある領域同士は同期しないガードを追加した。MPD RACK2
   のような空間的に分離した（重ならない）複数ピース合算のケースは、互いに
   `regions_overlap()` が False になるため、同期は引き続き正しく機能する。

   **重なる領域の確定名を候補から除去（`_remove_overlap_claimed_candidates`、
   v1.5.14）**: v1.5.11 までの対策は「選択の同期」を防ぐのみで、`region_name_candidates`
   が領域ごとに独立して評価する以上、重なる領域の確定名がもう片方の候補リストに
   残ること自体は止めていなかった。`EE6313-546-01E.dxf` の領域1（`B CHAMBER`、外側）
   の候補に、内包される領域2の確定名 `BAKE HEATER UNIT RX` が残っていたのは矛盾
   （重なる領域同士は同期しない＝選んでも意味がないのに選択肢として見える）と
   ユーザーが指摘。`analyze_dxf_regions` の最終段で、重なる(`regions_overlap`)
   領域どうしの候補に同じテキストがある場合、距離がより小さい（その名称を確信度
   高く保持している）側にのみ残し、距離が大きい側の候補からは除去する（距離が
   等しい場合はどちらからも除去しない）。`default_name`/`default_name_tier` も
   除去結果に応じて再計算する。MPD RACK2 のような重ならない複数ピース合算は
   `regions_overlap()` が False のため対象外で、同名共有は引き続き機能する。

   **L字型領域の切り欠き下向きエッジを Tier2 に追加（`_notch_bottom_edges`、
   v1.5.27）**: Tier1/2 の従来実装は「最下端レベル（最小y ±2.0）の横エッジ」
   「最上端レベル（最大y ±2.0）の横エッジ」しか探索対象にしないため、L字型領域で
   名称ラベルが切り欠き部の下向き横エッジ（最下端ではない）の直上に置かれている
   場合に候補から漏れていた（ユーザー報告: `EE6491-039-04A.dxf` の
   `SYSTEM I/F BOX`。FLAT CABLE 部と一体のL字型領域〔71.3%〕で、ラベルは切り欠き
   水平線 LINE #7DE〔y=124.76、最下端は y=13.24〕の直上 3.5 にあり、上端近傍の
   ケーブル型式 `FLAT CABLE` が default になっていた）。`_notch_bottom_edges()`
   （最下端レベル以外の横エッジのうち、エッジ中点の 0.5 直上が領域内・直下が
   領域外のもの）を追加し、Tier2 スキャンの対象を `_top_edges() +
   _notch_bottom_edges()` に拡張した。長方形では下向きエッジは最下端にしか
   存在しないため挙動不変。この図面では最下端エッジ近傍（Tier1）の
   `HEATER CTRL B.D` がいったん default になるが、同名をより近距離で持つ
   ネスト領域が `_remove_overlap_claimed_candidates`（v1.5.14、前述）で候補を
   引き取るため、最終的な default は `SYSTEM I/F BOX` になる。また「同名複数
   ピース合算」の保持条件（手順5）はこの候補整理**前**の default 名で集計される
   ため、ネストされた `HEATER CTRL B.D` 領域（8.5%、単独では閾値未満）の保持も
   従来どおり機能する。全サンプル137件の before/after 比較で、default 名の変化は
   同型パターンの改善2件（`EE6097-039-06C`/`EE6321-039-06A`: ケーブル型式→
   `SYSTEM I/F BOX`）のみであることを確認済み。回帰テスト
   `test_lshape_notch_bottom_edge_label_is_candidate` が本ケースを固定する。
7. **ラベル所属**（`assign_region_labels`）: 点-多角形内包判定。1ラベルが複数領域に
   所属可。出力の「領域」列に所属領域名（複数はカンマ区切り、領域外は空欄）。

### Excel 出力（`create_region_excel_output`）

| シート | 内容 |
|--------|------|
| `Summary` | ファイルごとの図面枠数・検出領域数・確定領域数・枠内/領域内ラベル数 |
| `領域一覧` | ファイル名・ページ No.・領域名・面積率・領域内ラベル数 |
| `領域別ラベル一覧` | 領域名ごとに全ファイル横断でラベルを集計（v1.5.28、後述） |
| `<ファイル名>` | `ラベル` / `個数` / `領域` 列。図面枠内の全ラベルを出力 |

**`領域一覧` の `ページ No.` 列（v1.5.28、旧名 `図面`）**: `r['frame'] + 1` — その
ファイル内の図面枠の**通し番号**（1ファイル複数図面の場合の何ページ目か）であり、
DXF図番ではない。当初「図面番号」への改称が検討されたが、実際の図番（`図番`）と
混同されるためユーザー指示により `ページ No.` に確定した。

**`領域別ラベル一覧`（`build_region_label_summary()`、v1.5.28）**: 検出した領域名
ごとに、全ファイル横断でラベルと出現個数を一覧できるシート。同名の領域は複数
ファイルにまたがって1グループに合算される（矩形領域抽出は元々「同名複数ピース
合算」「他図面でも同名採用」等、複数ファイルで同じ領域名を共有する前提の設計の
ため、これは自然な拡張）。

ヘッダー行: `領域名` / `ラベル` / `合計個数` / (`図番` / `個数`) × ファイル数
（`領域一覧`直後にシートを作成）。データ行はラベル1件につき1行で、`合計個数` は
全ファイルでのそのラベル・その領域名内での出現数の合計、以降はファイルごとの
個数が並ぶ。**あるファイルにそのラベルが存在しない場合は 0**（ユーザー指定）。

各ファイル列の `図番` はDXFから抽出した図番（`region_analyses` の
`main_drawing_number`。「図面番号・タイトル・サブタイトルを抽出」オプションON時
のみ取得可能）。未抽出の場合はファイル名（拡張子なし）にフォールバックする
（ユーザー確認: 「ファイル名が正しければ図番と一致するはず」）。

内部実装: `build_region_results()` が各ファイルの `assigned`（ラベル→所属領域の
対応）を集計する際に、`region_label_counts: {領域名: {ラベル: 個数}}` と
`drawing_number`（抽出済み図番 or 空文字列）を新たに `region_results[fname]` へ
追加する。`build_region_label_summary(region_results)` はこれを入力に、
`per_file` の内部キーを**ファイル名**（表示用の図番/フォールバック名ではなく）に
することで、2ファイルが同じ図番（またはフォールバック時の同名ファイル名）を
持つ場合の値衝突を避けている。ヘッダー・セルへの書き出し時にのみファイル名→
表示用図番のマッピングを適用する。

### 設定パラメータ一覧（`DEFAULT_REGION_CONFIG`）— 役割と作用箇所

処理段階は次の順。各パラメータが**どの段階で作用するか**を「作用箇所」に示す。

```
(A) 図面枠検出 → (B) 線分収集 → (C) 共線セグメント結合 → (D) 閉領域(面)検出
  → (E) 面積絞り込み → (F) 領域除外 → (G) 名称候補 → (H) ラベル所属
```

**(A) 図面枠検出**（`detect_drawing_frames`）

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `frame_lineweight` | 100 | この太さの線分を図面枠の辺として扱う |
| `snap` | 2.0 | 枠辺の軸平行判定・x位置クラスタの許容誤差（軸分類全般にも使用）|

**(B) 領域境界線の収集**

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `region_lineweight` | 25 | この太さの線分を領域境界の候補とする |
| `region_color` | 2 | かつこの色(ACI)の線分のみを領域境界候補とする |

**(C) 共線セグメント結合**（`_merge_collinear`）

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `merge_level_tol` | 0.5 | 同一線とみなすレベル座標(縦=x/横=y)の一致許容。別レベル(別矩形)を繋がない |
| `bridge_vertical_gaps` | True | 縦線分のギャップ（部品で途切れた箇所）を橋渡しする |
| `bridge_horizontal_gaps` | False | 横線分のギャップは橋渡ししない（別矩形取り込み防止）|
| `corner_tol` | 0.5 | 縦ギャップ端に横線端点が一致（コーナー）かの許容。一致＝段差で橋渡ししない |
| `connection_point_margin` | 0.1 | 縦ギャップ上に円(接続点)があるかの判定距離。あれば橋渡ししない（配線）※(F)と共用 |

**(D) 閉領域(面)検出**（`_find_rectilinear_faces`）

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `face_snap` | 0.1 | 矩形を構成する線分どうしの接続点(角・T字)の座標一致マージン |

> 面探索内部では、座標を `max(face_snap, 0.2)` の許容でクラスタ正規化し、round 境界での
> 一致点分裂を防ぐ（手描きの微小ズレ対策）。

**(E) 面積絞り込み**

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `min_face_ratio` | 0.005 | 面として残す最小面積（枠面積比、ノイズ除去の下限）|
| `area_ratio` | 0.20 | 単独領域として採用する最小面積（枠面積比）|
| `group_area_ratio` | 0.10 | 同名複数ピース(≥2)を合算して採用する最小合計面積（枠面積比）|

**(F) 領域除外**

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `exclude_titleblock` | True | 図番/TITLE/流用元 等を含む図番枠を領域から除外 |
| `exclude_connection_point_regions` | True | 境界に接続点(円)を持つ領域（配線ループ）を除外する機能のON/OFF |
| `connection_point_threshold` | 1 | 境界上の接続点が**何個**以上で領域を除外するか（個数）|
| `connection_point_margin` | 0.1 | 円が境界線から**何座標**以内なら「境界上」と数えるか（距離）※(C)と共用 |

**(G) 名称候補**（`region_name_candidates`）

| パラメータ | 既定 | 役割 |
|-----------|------|------|
| `name_min_dist` | 1.0 | 通常パス: 下端横線分からこの距離未満のラベルは候補から除外。フォールバック時は 0 |
| `name_max_dist` | 10.0 | 下端横線分（フォールバック時は全横エッジ）からこの距離以内のラベルを候補にする |
| `name_min_letters` | 3 | 名称候補に必要な英字数 |
| `name_exclude_terms` | (NOTE, ☆) | これらの語を含むラベルを候補から除外 |
| `name_exclude_lowercase` | True | 英小文字を含むラベルを候補から除外（領域名は大文字想定）|
| `exclude_circuit_symbols` | True | 機器符号(候補)パターンのラベルを候補から除外 |
| `circuit_symbol_keep_terms` | (RACK,) | 上記除外の例外。この語を含むラベルは機器符号扱いしない（例 RACK1）|

> (H) ラベル所属（`assign_region_labels`）は閾値を持たず、点-多角形内包判定のみ。

### 機器符号フィルタ正規表現パターン

| パターン種別 | 例 |
|------------|-----|
| 英文字のみ（2 文字以上）| `FB`、`CNCNT` |
| 英文字+数字 | `R10`、`CN3` |
| 英文字+数字+英文字 | `X14A`、`RMSS2A` |
| 括弧付き | `FB()`、`R10(2.2K)`、`MSS(MOTOR)` |

### 機器符号妥当性チェックパターン

| カテゴリ | パターン例 |
|---------|----------|
| CB 系 | `CB001`、`ELB(CB)001`、`MCCB001`、`NFB001` |
| 基本部品 | `R`、`R1`、`C1`、`L1`、`Q1` |
| IC | `U`、`U1`、`U10A` |
| 電源 | `PS`、`PSW`、`DC1`、`AC1` |
| モータ | `M`、`M1`、`MOT1` |
| リレー | `K`、`K1`、`MC1` |
| スイッチ | `S`、`S1`、`SW1`、`PB1` |
| 表示 | `H`、`H1`、`HL1`、`PL1` |
| 端子 | `X`、`X1`、`CN1`、`TB1` |

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

---

## 依存パッケージ

```
streamlit>=1.40.0, ezdxf>=1.4.2, pandas>=2.0.0
xlsxwriter>=3.0.0, openpyxl>=3.0.0
```

---

## 既知の制限

| 制限 | 詳細 |
|------|------|
| レイヤー取得 | 第 1 ファイルのみからレイヤーを取得。ファイルごとにレイヤーが異なる場合は考慮されない |
| MTEXT 複雑書式 | v1.5.1 で ezdxf `plain_mtext()` ベースへ移行し、`\S` 分数・`%%c`/`%%d`/`%%p`・キャレットシーケンス等も処理可能になった。ごく稀な独自書式は残る可能性がある |
| タイトル抽出精度 | `TITLE` ラベルの配置パターンに依存するため、図面レイアウトが異なると誤抽出する場合がある |
| 大量ファイル | 10 ファイル超でメモリ消費が大きくなる可能性がある |
| **LINE 矩形の部品輪郭（未対応課題）** | `DE5434-563-03A.dxf` のように、lw=25/color=2 の LINE エンティティ 4 本で形成された「細い閉じた矩形（部品輪郭）」（実例: x=81~90, y=98~394.5、幅9単位）が領域境界線と同じ属性を持つ場合、アルゴリズムが部品輪郭と領域境界線を区別できない。部品輪郭の両端が領域中央の縦仕切りを形成するケースでは、子領域の境界が「正確な直線」にならず、左右の子領域を閉領域として独立して検出できない（結果として合体した親領域のみが検出される）。**将来対応案**: lw=25/color=2 が形成する「縦横比が高い閉矩形」を部品輪郭として自動判定し検出対象から除外する。あわせて、合体親領域に付与する名称を「子領域の名称候補を除外した上で、底辺中央により近いラベルを優先して探索する」ロジックを追加することで、合体親に本来の領域名（例: `FX CHAMBER`）を正しく割り当てられるようになる。|

---

## 機能拡張ポイント

| テーマ | 実装アプローチ |
|--------|--------------|
| ファイルごとのレイヤー選択 | `get_layers_from_dxf()` を全ファイルに適用し、ファイルごとに選択 UI を追加 |
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
| ラベルが抽出されない | TEXT/MTEXT エンティティの有無、レイヤー選択、機器符号フィルタ設定を確認 |
| 処理タイムアウト | ファイル数を減らして再試行 |
| 機器符号が認識されない | フィルタパターンを確認。妥当性チェックを無効にして試行 |

---

## DXF-label-diff との違い

| アプリ | 用途 |
|--------|------|
| DXF-label-diff | 2 つの DXF ファイルのラベルを比較し差分を抽出 |
| DXF-extract-labels | 単一または複数の DXF ファイルからラベルを抽出 |

両プロジェクトは同一の `utils/extract_labels.py`・`utils/common_utils.py` を共有している。

---

## バージョン履歴

| バージョン | 変更内容 |
|-----------|---------|
| v1.5.28 | ユーザー要望2件。**(1) `領域一覧` の `図面` 列を `ページ No.` に改称**（中身は変更なし、ファイル内の図面枠通し番号）。当初「図面番号」への改称が候補に挙がったが、実際の図番（`図番`）と混同されるとの指摘を受け `ページ No.` に確定。**(2) `領域別ラベル一覧` シートを新設**（`領域一覧` の直後）: 検出した領域名ごとに全ファイル横断でラベルと出現個数を集計する。同名の領域は複数ファイルにまたがって1グループに合算（「同名複数ピース合算」「他図面でも同名採用」等、既存の複数ファイル前提の設計を踏襲）。ヘッダーは `領域名`/`ラベル`/`合計個数`/(`図番`/`個数`)×ファイル数。`図番` はDXFから抽出した図番（未抽出時はファイル名にフォールバック、ユーザー確認: 「ファイル名が正しければ図番と一致するはず」）。ラベルが存在しないファイル欄は 0（ユーザー指定）。`build_region_results()` に `region_label_counts`/`drawing_number` を追加し、新関数 `build_region_label_summary()` で集計する。`per_file` の内部キーはファイル名（表示用図番ではない）とし、2ファイルが同じ図番/フォールバック名を持つ場合の値衝突を避けた。回帰テスト3件追加（`test_excel_output.py`）、実データ（`EE6892-039-05B.dxf`+`EE6492-039-38A.dxf`）でのE2E手動検証済み、全83件 pass。 |
| v1.5.27 | 2件の変更。**(1) 「領域の確認」に図番・タイトル・サブタイトルを表示**: 「図面番号・タイトル・サブタイトルを抽出」オプションが ON のとき、「領域を検出」実行時に `extract_labels()` も各ファイルへ実行して図番・タイトル・サブタイトルを `region_analyses` の analysis dict（`main_drawing_number`/`title`/`subtitle` キー）に格納し、「領域の確認」のファイル名見出し直下に「図番：… / タイトル：… / サブタイトル：…」を `st.caption`（「図面枠 x 個 / 検出領域 x 個」と同じ書式）で表示する。空の項目は省略。オプション OFF 時は抽出も表示もしない（解析コストをかけない）。**(2) L字型領域の切り欠き下向きエッジ直上の名称ラベルを候補化**（`_notch_bottom_edges`）: `EE6491-039-04A.dxf` の `SYSTEM I/F BOX`（FLAT CABLE 部と一体のL字型領域）が検出されないとユーザー報告。名称候補の Tier1/2 探索が最下端/最上端レベルの横エッジしか見ておらず、切り欠き部の下向きエッジ（LINE #7DE）直上のラベルが候補から漏れていた（詳細は「検出アルゴリズム」手順6）。`_notch_bottom_edges()` を追加し Tier2 スキャン対象を `_top_edges() + _notch_bottom_edges()` に拡張。長方形領域では挙動不変。全サンプル137件の before/after 比較で default 名の変化は同型パターンの改善2件（`EE6097-039-06C`/`EE6321-039-06A`）のみ。当初案の「下端/上端エッジ全体の向きベース化＋同名グループ集計へのTier1候補追加」は差分が37ファイルに及んだため破棄し、Tier2への追加のみに絞り込んだ。`sample-dxf/EE6491-039-04A.dxf` を追加、回帰テスト `test_lshape_notch_bottom_edge_label_is_candidate` を追加、全80件 pass。DXF-viewer の `core/region_detector.py` にも `_notch_bottom_edges` を移植済み（併せて viewer 未移植だった `_remove_overlap_claimed_candidates`〔v1.5.14〕も移植。default_name のみ照合する Search Boundary で `SYSTEM I/F BOX` がヒットするために必要）。 |
| v1.5.26 | 「領域を検出」「ラベルを抽出」「Excelをダウンロード」ボタンの配色をユーザー指定仕様に合わせて修正。従来は3ボタンとも `type` 未指定（secondary=白）または `width='stretch'`（Excelダウンロードのみ横幅いっぱい）で、有効な次操作が視覚的に分からなかった。`session_state`（`region_analyses`・`excel_result` の有無）から各ボタンの `type` を動的に計算するよう変更（詳細は「ボタンの色分け」節）。Excelダウンロードボタンは `width='stretch'` を外し、他の主要ボタンと同程度の横幅で左寄せ表示に変更。処理ブロック末尾に `st.rerun()` を追加し、クリック直後に新しい色が即座に反映されるよう修正（`st.rerun()` が無いと、ボタンウィジェットが処理より前に描画されるため次の操作まで旧い色が残って見えるユーザー報告あり）。`~/.claude/skills/streamlit/SKILL.md` にも本パターン（状態に応じた動的な `type` 計算・`st.rerun()` の必要性）を汎用ノウハウとして追記。 |
| v1.5.25 | 出力ファイル（Excel）のラベル・矩形領域名称を**すべて半角に正規化して集計・記録**するよう変更（ユーザー指定の仕様）。`common_utils.py` に `normalize_width()`（NFKC 正規化。全角英数字・記号・スペース→半角、かな・漢字は不変）を追加。**集計への適用**: 通常モードは `create_excel_output()` が Counter 集計前にラベルを正規化（半角 `CN1` と全角 `ＣＮ１` が同一行に合算される）、Invalid シートの機器符号も正規化。領域付きモードは `build_region_results()` が集計前にラベルと確定領域名を正規化（各ファイルシートの `ラベル`/`領域` 列・`領域一覧` の `領域名` 列すべて半角）。**判定への適用**: `filter_non_circuit_symbols()`・`validate_circuit_symbols()` は判定のみ半角相当で行い（返り値の表記は不変）、全角の機器符号（`ＣＮ１` 等）が「機器符号のみ抽出」フィルタで欠落したり妥当性チェックで誤って invalid になる問題を解消。整合のため `region_detector._is_valid_name_candidate()` の除外語・keep-term（`RACK`）照合も半角相当に変更（全角 `ＲＡＣＫ１` が機器符号除外と keep-term のすき間に落ちるのを防止）。UI の領域名候補表示は図面の表記（全角のまま）を維持し、出力時のみ正規化する。実データ検証: `EE6492-039-38A.dxf`（全角のみの図面）で領域付き/通常（フィルタON/OFF・妥当性チェックON）の全出力に全角が残らないことを openpyxl で確認。回帰テスト6件追加（`test_excel_output.py`）、全79件 pass。 |
| v1.5.24 | 領域名候補の英字判定（`_count_letters`）を**全角英字（Ａ-Ｚ, ａ-ｚ）にも対応**するよう修正。従来は `ch.isascii() and ch.isalpha()` で ASCII 半角英字のみを英字とみなしていたため、領域名ラベルが全角文字のみで書かれた図面（例: `ＳＹＳＴＥＭ　Ｉ／Ｆ　ＢＯＸ`）では `name_min_letters`(3) 条件を常に満たせず、名称候補が一切検出できなかった。ユーザー報告（`EE6492-039-38A.dxf` で「以前は検出できていたのに検出できなくなった」）を受け調査したところ、`git worktree` で region 検出機能導入時点（v1.4.0, `094ff71`）まで遡っても同じ結果であり、退行ではなく機能導入当初からの未対応（全角のみラベルへの非対応）と判明。`_is_letter()`（全角対応の英字判定）・`_is_lowercase_letter()`（全角小文字も含めた小文字判定、`exclude_lowercase` フィルタで使用）を追加し、`is_single_uppercase_letter()`（`extract_labels.py`）で既に採用されていた全角対応の考え方を踏襲。`sample-dxf/problems/EE6492-039-38A.dxf` を追加、回帰テスト `test_zenkaku_only_label_is_valid_name_candidate` を追加、全74件 pass 確認。DXF-viewer の `core/region_detector.py` にも同じ修正を移植済み。 |
| v1.5.23 | `analyze_dxf_regions()` に**レベル汚染フォールバック（4パス目）**を追加。既存3パス（LINE→+LWPOLYLINE→横ギャップ橋渡し）の後に、「閾値超えゼロの図面枠」が存在する場合に限り、スパン単位レベル（グループ全体の平均でなく、そのスパンを構成した線分だけの平均）で再検出し、名称一致で採用判定する。**発動ゲート条件**: (a) 閾値超えゼロの枠が1枚以上ある かつ (b) 他の枠に閾値超え領域が存在する（全枠ゼロの電源基板回路図等では発動しない）。**採用条件**: 回復した領域の `default_name` が他枠で検出済みの名称と一致する枠のみ置き換える（1ファイル複数図面は同名領域が枠をまたぐことを根拠とする）。**根本原因**: `merge_level_tol=0.5` の共線セグメント結合で、スパンが重ならない近接線分（例: 境界線 y=122.00 の 0.37 上にあるコネクタ箱底辺 y=122.37）が同一レベルクラスタに取り込まれてクラスタ平均がシフト（y≈122.25）し、縦線端点（y=122.00）との接続が `face_snap=0.1` 許容を超えて切断→閉路不成立、という「レベル汚染」（Level Contamination）現象。`_merge_collinear` に `span_levels` 引数を追加（スパン単位レベル算出を有効化する）、`DEFAULT_REGION_CONFIG` に `span_level_merge: False`（既定値・通常時は全体平均のまま）を追加。全変更はフォールバックとして閉じており、通常向き3パスへの副作用はゼロ（135サンプルで DIFF 1件のみ＝対象ファイル EE6892-039-05B.dxf の2ページ目 SYSTEM I/F BOX が新規検出）。`sample-dxf/EE6892-039-05B.dxf`（4ページ構成）を追加。回帰テスト `test_level_pollution_fallback_recovers_frame`・`test_level_pollution_fallback_not_triggered_on_schematic` を追加、全73件 pass 確認。 |
| v1.5.22 | コード品質リファクタリング（ロジック変更なし・出力は不変）。`region_detector.py`: `_label_ok()` クロージャの重複（`region_name_candidates` / `_name_union_parent` / Tier3 フォールバックの3か所）を module-level の `_is_valid_name_candidate()` に統合。`_detect_regions` 内のマジックナンバー `5` を `_FRAME_MARGIN = 5` 定数に、`_trace_faces` 内の `200000` を `_MAX_FACE_NODES = 200_000` 定数に置き換え。5公開 API 関数（`detect_drawing_frames` / `region_name_candidates` / `analyze_dxf_regions` / `assign_region_labels` / `build_region_results`）に Python 3.10+ 型アノテーションを追加。`build_region_results()` を `excel_output.py`（I/O 層）から `region_detector.py`（ビジネスロジック層）へ移動（モジュール責務の正常化）。`app.py` の import 先を更新。`excel_output.py`: モジュール docstring 更新、不要 import 削除（`assign_region_labels` / `filter_non_circuit_symbols`）。回帰テスト 25 件追加（`test_circuit_symbol_filter.py` 16 件・`test_excel_output.py` 9 件）、全 71 件 pass 確認。DXF-viewer の `core/region_detector.py` にも ①-④（`_is_valid_name_candidate` 統合・import 整理）および ⑤（マジックナンバー定数化）を移植済み。 |
| v1.5.21 | `_name_union_parent()` に `exclude_names` パラメータを追加。`_resolve_union_parents()` で**同一フレーム内**の非親・非子領域がすでに `default_name` として使用している名称を `exclude_names` に渡し、合体親がそれらを誤って取得しないように修正。フレームをまたいだ場合（例: `DE5434-563-03A.dxf` の frame0・frame1 が同じ 'FX CHAMBER' を名乗る）は除外対象にしない（`parent_claimed_by_frame` でフレーム別に管理）。背景: EE6888-631-01A.dxf では frame0 に正規の 'SYSTEM I/F BOX' 領域が2件存在し、その内部の合体親（2件）が v1.5.20 で 'SYSTEM I/F BOX' を誤取得して 'SYSTEM' クエリの一致数が 2→4 に増加する回帰が発生した。修正後、EE6888-631-01A.dxf・EE6492-631-02A.dxf の回帰テストが再 PASS（'SYSTEM' クエリ = 2件）。`DE5434-563-03A.dxf` (5% 閾値) は引き続き frame0・frame1 の合体親がそれぞれ 'FX CHAMBER' を取得して保持。全25件の回帰テスト PASS。DXF-viewer の `core/region_detector.py` にも同じ変更を移植済み。|
| v1.5.20 | `_resolve_union_parents()` を**除去から命名**に変更。結合親領域（2兄弟矩形の合体）が検出された場合、子領域の名称候補をすべて除外した上で、底辺中央近接条件（中心距離を第2ソートキーとする距離ソート）を加味した専用の名称探索関数 `_name_union_parent()` で親固有のラベルを探索し、見つかった場合は名称を更新して親を**残す**。見つからなかった場合は従来通り除去する。底辺近傍の探索は `require_inside` を緩和し、領域外（底辺の下方向）も探索対象にする（例: `DE5434-563-03A.dxf` の 'FX CHAMBER' @ y=76.4 は polygon 外だが底辺y=83 から 6.6 ユニット）。`_detect_union_parents()` の戻り値を `[parent_idx, ...]` から `{parent_idx: (child_j, child_k)}` に変更（内部 API のみ）。`analyze_dxf_regions()` の呼び出しを `_resolve_union_parents(regions, labels=frame_labels, cfg=cfg)` に更新。検証: `DE5434-563-03A.dxf` (5% 閾値) で合体親 [1] が 'FX CHAMBER' (pos=(98.6,76.4)) として残り、子 [3]='SB-1A(FX1)'・[4]='CN I/F B.D TYPE3' は独立して保持。`DE5401-405-21B.dxf` (20% 閾値) は未採用ラベルなしのため従来通り合体親が除去され、全25件の既存回帰テストが引き続き PASS。DXF-viewer の `core/region_detector.py` にも同じ変更（`name_candidate_positions` 用途の `_name_union_parent` 版）を移植済み。|
| v1.5.19 | `_split_axis_aligned` の長さ比較を `> eps` から `>= eps` に変更し、長さがちょうど `snap`(eps=2.0) ユニットの極短スタブも V/H 線分として検出されるよう修正。実例: `DE5434-563-03A.dxf` の縦スタブ #10D0 (x=86, y=400~402, len=2.0) が `> eps` 条件を満たさず除外されていたため、x=86 の縦仕切り線分（INSERT #188E 部品上下のスタブ #10CF + #10D0 の橋渡し）が成立せず、frame[0] の 'SB-1A(FX1)'（左領域）と 'CN I/F B.D TYPE3'（右領域）が独立した閉領域として検出されていなかった。修正後、5% 閾値で両領域が正しく検出される（[3] 7.7% bbox=(32,83,86,402) 'SB-1A(FX1)'・[4] 7.6% bbox=(86,83,140,402) 'CN I/F B.D TYPE3'）。DXF-viewer の `core/region_detector.py` にも同じ変更を適用済み。|
| v1.5.18 | `analyze_dxf_regions()` に `_resolve_union_parents()` を追加し、**横線分（または縦線分）で2分割された兄弟矩形の合体親矩形が planar graph の半面として誤検出されるケースを自動除去**するよう修正。`DE5401-405-21B.dxf` では、y=265 の横線分で親矩形（52.7%）が上部 L CHAMBER（26.3%）と下部 FX CHAMBER（26.3%）に分割されているが、外側の親矩形も有効な planar graph 面として検出され（全3頂点とも4頂点矩形）、`_detect_complement_pairs` の「large の頂点数 > small の頂点数」という条件を満たさないため v1.5.17 の `_resolve_complement_faces` では検出できなかった。`_detect_union_parents()` は「area(P) ≈ area(Q) + area(R)」「P の全頂点が Q.corners ∪ R.corners に含まれる」「regions_overlap(P,Q) かつ regions_overlap(P,R)」「NOT regions_overlap(Q,R)」の4条件を満たす P を結合親として検出・除去する。`_resolve_union_parents()` は `_resolve_complement_faces()` の直後（`_remove_overlap_claimed_candidates` より前）に呼ぶ。848ファイルの回帰テストでは新規 DUPE ファイル 0件、14ファイルの既存 DUPE が解消、4ファイルが部分改善（DUPE 名称数が1減少）。回帰テスト `test_horizontal_sibling_union_parent_removed` を追加（`tests/regression/test_region_extraction.py`）。DXF-viewer の `core/region_detector.py` にも同じ2関数（`_detect_union_parents`・`_resolve_union_parents`）を移植済み。|
| v1.5.17 | `analyze_dxf_regions()` に `_resolve_complement_faces()` を追加し、**兄弟矩形の部分共有辺（B CHAMBER 右辺 ＝ FX CHAMBER 左辺 等）が生む補完面（complement face）を自動解消**するよう修正。planar graph の半面探索では、2つの矩形が縦辺を部分共有すると「2矩形を合体した補完面（頂点数＝両矩形の頂点数合計マイナス共有頂点数）」が必ず生成される。FX CHAMBER（面積14.6%）は20%閾値を下回るため直接検出されず、結果として補完面（78.2%）と実際の B CHAMBER（63.6%）の両方に `B CHAMBER` が等距離で候補として付き、`_remove_overlap_claimed_candidates` がtie-break を解けずに B CHAMBER が2件検出されるDUPEになっていた（ユーザー報告: `EE6313-545-01D.dxf`）。`_detect_complement_pairs()`（small の全頂点が large の頂点集合に含まれ、かつ large が larger で重なっているペアを検出）・`_extract_complement_subpolygons()`（補完面の境界を辿り small にない「追加頂点列」の各連続区間をサブ領域ポリゴンとして切り出す）・`_resolve_complement_faces()`（補完面を除去し、サブ領域には補完面の name_candidates のうち base_face に claimed されていない候補を継承）の3関数を `_remove_overlap_claimed_candidates` の呼び出し前に追加。`EE6313-545-01D.dxf` の検出結果が `['B CHAMBER', 'B CHAMBER', 'BAKE HEATER UNIT FX']`（DUPE）→ `['B CHAMBER', 'BAKE HEATER UNIT FX', 'FX CHAMBER']`（OK）に改善。846ファイルの回帰テストでは `EE6313-545-01D.dxf` が OK に修正され、他ファイルへの退行は0件（`EE6313-546-01E.dxf` 等の既存DUPE は補完面ペア=0件で本変更の影響を受けない）。回帰テスト `test_sibling_fx_chamber_extracted_from_complement_face` を追加（`tests/regression/test_region_extraction.py`）。DXF-viewer の `core/region_detector.py` にも同じ3関数＋`regions_overlap` 等の必要なヘルパー関数を移植済み（`_resolve_complement_faces` の DXF-viewer 版は `dangling_edges`/`_tier_by_text` を持たず `name_candidate_positions: {}` を付与）。|
| v1.5.16 | 図面枠の識別条件に `color=7`（ACI白）を追加し、`detect_drawing_frames` の `min_side=400` 固定閾値を撤廃（既定値0=フィルタなし）。従来は lineweight=100 のみで図面枠線を識別していたが、枠とは無関係な短い lineweight=100 線分（色5の小さな線分群等）が混在するファイルがあり、`min_side` で高さによる足切りをしていた。これが原因で、縦辺の高さが400未満の枠（実例: `EE6097-039-06C.dxf`、高さ277、INSERT `#E02`/`#E03`/`#E04` 内のLINE 4本×3枠）が「図面枠が見つかりませんでした」エラーになっていた。ユーザーが実図面を調査し「図面枠はすべて lineweight=100 かつ color=7」と報告。`DEFAULT_REGION_CONFIG` に `frame_color: 7` を追加し `_collect_region_geometry()` の `handle_line` に色判定を追加したところ、サンプル137件（`Tools/sample-dxf/` 非pairC 27件+pairC 110件）で検証した結果、既存の正しい検出への退行は0件、従来「枠が見つからない」だった22件すべてが解消した（lineweight=100のみで`min_side`を単純に撤廃すると無関係な短い線分どうしが偽の枠を作り `EE6868-500-01C.dxf` で13→31フレームに崩壊することを確認、color条件追加でこれを回避）。DXF-viewer の `core/region_detector.py` にも同じ修正を移植済み。|
| v1.5.15 | 回帰テストのサンプルDXF探索先を、3プロジェクト(DXF-viewer/DXF-extract-labels/DXF-diff-manager)で共用する `Tools/sample-dxf/`（プロジェクト直下の `sample-dxf` symlink 経由）に変更。`test_region_extraction.py` の `MULTI`/`SINGLE`/`ROTATED`/`DANGLING`・`test_drawing_number_types.py` の `EE6888-602-01A.dxf` は、ファイル名を直接ハードコードして参照していたため、ユーザーが `sample-dxf/` 内のファイルを `viewer-error/`/`problems/` 等のサブフォルダへ再編成した際に発見できなくなり、`test_region_extraction.py` で2件 FAIL・9件 skip が発生した。`_find_sample(name)`（直接パスを先に試し、無ければ `os.walk` で再帰的にファイル名一致を探す）を両ファイルに導入し、サブフォルダへの移動・将来の新規フォルダ追加に追従できるようにした。修正後は全30件 pass に復旧。|
| v1.5.14 | `analyze_dxf_regions()` に `_remove_overlap_claimed_candidates()` を追加。重なる(`regions_overlap`)領域同士の名称候補に同じテキストがある場合、距離がより小さい（確信度の高い）側にのみ残し、もう片方の候補からは除去する。`EE6313-546-01E.dxf` の領域1（`B CHAMBER`、外側）の候補に、内包される領域2の確定名 `BAKE HEATER UNIT RX`（領域2への距離2.0、領域1への距離8.6）が残っていたのは、v1.5.11で同期は防止したものの選択肢として表示され続けるのは矛盾だとユーザーが指摘。修正後は領域1の候補が `B CHAMBER` のみになる。`default_name`/`default_name_tier` も除去結果に応じて再計算するよう変更（除去により1位の候補が変わる場合に対応）。MPD RACK2 のような重ならない複数ピース合算（`regions_overlap()` が False）は対象外で同名共有は引き続き機能。回帰テスト `test_overlapping_region_does_not_offer_other_regions_confirmed_name` を追加、`test_nested_regions_each_get_own_confident_default_name` を更新（領域1も領域2の名称を候補に持たなくなったことを反映）。|
| v1.5.13 | `region_name_candidates()` の Tier1/2 を**領域内側のラベルに限定**するよう修正。`_scan()` に `require_inside` 引数を追加し、Tier1/2（下端/上端、回転時は右端/左端の最近傍）のスキャン時に `_point_in_polygon()` で領域内側かどうかを確認する（Tier3 フォールバックは内外を問わず従来通り）。`DE5434-553-10B.dxf` の回転領域(id 3,4)で、領域の**外側**にあるラベル`EFEM UPPER`（右端から距離3.9）が、領域の**内側**にある正しいラベル`CONTROL BOX CORE FX`（距離5.2）より単純な距離比較で優先されてしまうバグを解消（default_name が `EFEM UPPER`→`CONTROL BOX CORE FX` に変化）。DXF-viewer の Search Boundary で「最上位候補（default_name）のみで照合」するよう変更した際にユーザーが発見（"CONTROL BOX CORE FX" で検索しても領域がヒットしない不具合の調査中、"EFEM UPPER は領域の外側なので優先順位が低いはず" との指摘）。全テストファイルの既存 default_name を検証し、このケース以外は変化なし（=他の全領域の正しい default_name は元々すべて領域内側のラベルだった）。回帰テスト `test_tier1_candidate_must_be_inside_the_region` を追加、`test_nested_regions_each_get_own_confident_default_name` を更新（領域2は領域1の名称`B CHAMBER`を、そのラベルが領域2の外側にあるため、候補に持たなくなった）。DXF-viewer の `core/region_detector.py` にも同じ修正を移植済み。|
| v1.5.12 | コード品質レビュー（モジュール性・可読性向けリファクタ。ロジック変更なし、出力は不変）。`region_detector.py`（1161行）にセクション見出しコメントを追加し、設定／ジオメトリ収集／ポリゴン幾何ユーティリティ／線分結合／図面枠検出／閉領域検出／名称候補／回転判定／タイトルブロック除外／トップレベル解析の10ブロックに整理。最も複雑だった `_find_rectilinear_faces`（175行）を `_build_planar_graph`（平面グラフ構築）・`_peel_dangling_branches`（行き止まり枝の除去・連結成分化）・`_trace_faces`（半面探索）の3関数に分割し、`_find_rectilinear_faces` 自体は3段のオーケストレーションのみに簡素化。`analyze_dxf_regions` 内の入れ子クロージャ（`_run_detection`/`_hits`）を `_run_region_detection`/`_count_threshold_hits` としてモジュールレベルに抽出し、3パス検出ロジックの見通しを改善。`app.py` の「領域の確認」ループからも、座標ポップオーバー（`_render_corners_popover`）・行き止まり枝表示（`_render_dangling_edges_section`）・デフォルト候補インデックス計算（`_compute_default_candidate_index`）の3関数を抽出し、ファイル→領域の本流ループを見やすくした。DXF-viewer の `core/region_detector.py` にも同じ構造改善（DXF-viewer独自の `_filter_eligible_labels` キャッシュ・`_label_position_for_candidate` 等は保持）を移植済み。回帰テスト41件・DXF-viewer側の `test_region_search.py` 等は全て同じ結果で通過を確認。|
| v1.5.9 | `region_name_candidates()` に優先順位（Tier）制を導入：Tier1=下端横エッジ最近傍（90°回転時は右端/左端のいずれか）、Tier2=上端横エッジ最近傍（回転時はもう一方）、Tier3=Tier1/2が空の場合のみポリゴン境界全体への最短距離でフォールバック。回転方向（+90°/-90°多数派）の判定は `_rotated_edge_roles()` を追加（`DE5434-553-10B.dxf` で確認した実例: +90°多数派→Tier1=右端,Tier2=左端）。各領域に `default_name_tier`（1/2/3）を追加し、`app.py` の他領域への選択同期（`selected_elsewhere`）がTier1/2（確信度の高い自前の候補）を上書きしないように変更。ユーザー報告: `EE6313-546-01E.dxf` の図面1/領域1,2（互いの候補リストに相手の名称を含む入れ子/隣接領域）が同じ選択に同期されてしまい、本来は領域1=`B CHAMBER`、領域2=`BAKE HEATER UNIT RX` で別々が正しい不具合を解消。|
| v1.5.11 | `regions_overlap()` を追加し、`app.py` の名称選択同期（`_on_change_radio`の手動選択伝播・`selected_elsewhere`の初期デフォルト同期）が、同一ファイル内で重なりのある（完全な内包も部分的な重複も含む）領域同士を誤って同期しないように修正。`EE6313-546-01E.dxf` の領域1（`B CHAMBER`、外側）・領域2（`BAKE HEATER UNIT RX`、内側、完全内包）で、デフォルトでない候補を手動選択すると、もう片方も同じ名称に同期されてしまうとユーザーが報告（v1.5.9で対応したTierガードは自領域の確信度に依存するため、手動選択やTier3同士のケースでは依然発生し得た）。当初は完全な内包のみを対象とする実装で提案したが、部分的な重複も対象にすべきとの指摘により、両ポリゴンの頂点＋辺の中点をサンプルに用いる一般的な重なり判定に変更。MPD RACK2のような空間的に分離した複数ピース合算は重ならないため、同期は引き続き機能する。|
| v1.5.10 | 領域境界線の収集条件に線種(linetype)チェックを追加（`_is_continuous_linetype`）。`lineweight=25`/`color=2`を満たしても線種がPHANTOM（二点鎖線）等のCircuit以外の場合は除外する。`EE6313-546-01E.dxf`で、実体の小さな矩形`MX CHAMBER`（handle 21AB/21AC/219A/219E、Continuous、面積1.8%）の周囲に重なるPHANTOM線種の矩形（21AE/21A1/21A9/2198等）が誤って境界線として認識され、実体矩形を「くり抜いた」形状の存在しない領域（10角形、面積4.6%）が誤検出されていた不具合をユーザーが報告（DXF-viewerで座標リストを確認した際に発覚）。修正後はPHANTOM由来の誤検出領域が消え、実体の矩形のみが残る（regions 5→4）。|
| v1.5.8 | 行き止まり枝の報告を「フレーム単位のフラットなリスト」から「枝（連結成分）単位、かつ各領域の `dangling_edges` に絞り込み」へ変更。`EE6313-546-01E.dxf` で「行き止まり枝158件」と表示され特定領域に無関係な部品の枝まで混在するとユーザーから指摘（v1.5.7時点では1本の枝が複数行に分かれ、かつファイル内の全枝が無差別に列挙されていた）。`_find_rectilinear_faces` の次数1ノード除去を Union-Find で連結成分化し、各枝の取り付け点（`attachment`）を求めるよう変更。`analyze_dxf_regions` は取り付け点が各領域のポリゴン境界上に乗るものだけをその領域の `dangling_edges` に割り付ける（トップレベルのフラットな `dangling_edges` キーは廃止）。結果、`EE6313-546-01E.dxf` の最大領域はちょうど2本の枝（handle `214F` の単独枝、`2199`→`21AD`→`219B`→`21AA`→`219F`→`21A7` の連結枝）に絞られた。DXF-viewer にもアルゴリズム部分（枝グルーピング。handle解決・領域絞り込み・UI表示は無し）を移植済み。|
| v1.5.7 | `region_detector.py` の閉領域検出（`_find_rectilinear_faces`）に行き止まり枝（dangling edge）の除去（2-core抽出）を追加。境界線と同じ線種を持つがどこにも閉じていない線分があると、半面探索がその枝を折り返すため頂点座標に「同じ点が2回連続する」アーティファクトが生じていた（`EE6313-546-01E.dxf` で報告）。副次効果として、同一物理境界が「綺麗な内側面」と「枝の往復で座標が汚れた外側面」の2領域として重複検出されるバグも解消（`EE6313-546-01E.dxf`: regions 6→5）。除去した枝は handle・座標を解決し `analyze_dxf_regions()` の `dangling_edges` キーに記録、`app.py` の「領域の確認」に「⚠️ 行き止まり枝」セクションとして表示するようにした。DXF-viewer にもアルゴリズム部分（handle解決・UI表示は除く）を移植済み。|
| v1.5.6 | `extract_labels.py` に INSERT展開のスキップ最適化（`_block_has_text_content()`）を追加。テキストを持たないブロックの INSERT は `virtual_entities()` を呼ぶ前にスキップし、手描き回路図（記号INSERTが多い）の抽出処理を高速化（サンプル161ファイルで処理時間約10%短縮、出力結果は完全一致を確認）。DXF-diff-manager の Step 2「ファイルを読み込む」高速化要望が発端。`DXF-diff-manager/utils/extract_labels.py` へ伝播済み（バイト一致）。|
| v1.5.5 | `app.py` の `_on_change_radio` に他領域への選択伝播を追加。従来は「他の図面/領域で選択済みの名称をデフォルトにする」（`selected_elsewhere`）が**チェックボックスの初回生成時のみ**有効で、生成後にユーザーが選択を変更しても他領域には反映されなかった（`st.session_state` の既存キーは初期値設定をスキップするため）。コールバックに選択した名称テキスト (`clicked_text`) を渡すよう変更し、同じ名称候補を持つ他領域（候補2件以上＝選択肢ありの領域に限る）の選択状態を即時に揃えるようにした（ユーザー指摘により追加）。回帰テストはStreamlitの`session_state`を要するためpytestでは追加せず、`st.session_state`をフェイク辞書に置き換えた手動シミュレーションで動作確認済み。コードレビュー時に `region_detector.py` 内 `analyze_dxf_regions` で `_is_globally_rotated()` を2回呼んでいた冗長呼び出しを1回（`rotated` 変数の再利用）に修正、また `tests/regression/test_region_extraction.py::test_connection_point_region_excluded` の `_collect_region_geometry()` 戻り値アンパック数が v1.5.2 の `region_lines_lp` 追加以降4個のままズレていた（無関係な既存バグ）のを5個に修正し、全テスト green 化。|
| v1.5.4 | 図面全体が90°回転して描かれているファイルへの対応を4点追加。(1) `region_name_candidates` に縦エッジフォールバック（`_all_vertical_edges`/`_dist_to_vertical_edge`）を追加。領域名が（通常の下端/上端横エッジでなく）左右いずれかの縦エッジ脇に置かれるファイルで名称候補が常にゼロになっていた問題を解消。(2) `_detect_regions`／`analyze_dxf_regions` に横線分ギャップ橋渡しのフォールバック（`bridge_horizontal_gaps=True`、縦線分ギャップ橋渡しと同じコーナー相手・CIRCLE安全条件を x/y 入れ替えて適用）を追加。回転図面では部品が横線分（本来の縦線分に相当）を途切れさせるため、既定のギャップ橋渡し方針（縦線分のみ）では大きな矩形が閉じずに検出漏れしていた問題を解消（実例: `CONTROL BOX CORE FX` を囲む矩形、handle 1EAF/1EB0/1E59/2748/1EA3/1EAE で構成）。(2)は「検出ゼロ件」だけでなく `_is_globally_rotated()`（ラベルの過半数が90°回転＝図面全体が回転していると判定）も条件に加え、通常向き図面で誤って隣接矩形を結合する副作用を防止（ユーザー指摘により追加）。(3) `region_name_candidates` に `also_scan_vertical` パラメータを追加し、回転図面では横エッジ側で候補が見つかった場合でも縦エッジ候補を常に追加合算するよう変更（候補ゼロのときだけのフォールバックでは、横エッジ側に1件でも候補があると縦エッジ側が完全に隠れてしまうため）。(4) フォールバック1/2の `min_dist=0`（線分上のラベルも含む）をユーザー指摘により撤回し、常に `name_min_dist`（既定1.0）を適用するよう変更。境界線分上(d=0)に偶然乗った無関係なラベル（コネクタ符号 `CN24POW04`/`CN24POW05`）が誤って名称候補・デフォルト名に選ばれ、(3)の対策後も本来の名称（`CONTROL BOX CORE FX`/`RX`）より優先されてしまう問題を解消。いずれも既存の正常向きファイル（EE6868/EE6888）の検出結果は変化なし（`also_scan_vertical` は回転判定が真のときのみ True、(4)の変更は元々 d=0 候補が存在しなかったため無影響）。回帰テスト `test_rotated_drawing_name_candidates_via_vertical_edge` / `test_rotated_drawing_horizontal_gap_bridging_closes_large_box` / `test_rotated_drawing_on_line_label_excluded_and_does_not_hide_real_name` を追加（サンプル DXF はサイズの都合で git 管理対象外）。|
| v1.5.3 | `app.py` の領域確認 UI を修正。(1) `selected_elsewhere` 収集で候補 1 件のみ（選択肢なし・自動確定）の領域をスキップするよう変更し、隣接領域が同じラベルを誤ってデフォルト選択する問題を解消。(2) 複数ファイル処理時に各ファイルの先頭へ h3 見出し（`### fname`）を追加し、2 ファイル目以降の前には `st.divider()` を挿入することで、ファイル境界を視覚的に明示。|
| v1.5.2 | `region_detector.py` の図面枠検出・LWPOLYLINE 境界対応を修正。`detect_drawing_frames` に `_merge_collinear(bridge=False)` を追加し、枠の縦辺が同一 x 上で複数線分に分断されていても正しく統合・高さ判定できるよう修正（EE6888-631-01A.dxf の右辺が y=367.5 で 2 分割されていた問題を解消）。`_collect_region_geometry` に `handle_lwpolyline_lp` を追加し LWPOLYLINE (lw=25/color=2) の辺を `region_lines_lp` として別収集。`analyze_dxf_regions` に `_run_detection` ヘルパーと LINE 優先 2 パス検出を導入（LINE のみで閾値超え候補ゼロかつ LWPOLYLINE 境界線がある場合に LINE+LWPOLYLINE で再検出）。DXF-viewer のアルゴリズムを移植。|
| v1.5.1 | MTEXT 整形（`clean_mtext_format_codes`）を手書き正規表現から ezdxf の `plain_mtext()` ベースへ移行。関数シグネチャ・呼び出し側は不変。実データ 12,145 件の MTEXT で旧実装と出力完全一致を確認、実ファイル E2E 抽出結果も同一ハッシュ（`cb112d6d…`）。加えて旧実装が未対応だった `\S` 分数・`%%c`/`%%d`/`%%p`（Ø/°/±）・`^I`/`^J`/`^M` キャレットシーケンスを正しく処理。回帰テスト `tests/regression/test_mtext_cleaning.py`（14件）追加。|
| v1.5.0 | コードベース モジュール分割リファクタリング: `utils/extract_labels.py`（1327行）を `extract_labels.py`（621行）＋ `region_detector.py`（707行）に分割。`app.py` から Excel 出力ロジックを `utils/excel_output.py`（264行）に分離。ロジック変更なし・回帰テスト全14件 PASS。|
| v1.4.0 | UI 全面再構築: オプション統合（機器符号のみがデフォルト ON・機器符号妥当性チェック連動・図面番号/タイトル統合チェック）、詳細設定フォーム（`st.form` + 「設定完了」ボタン + `st.toast` フィードバック）、「領域を検出」常時表示、「ラベルを抽出」が領域/通常モード共用、レイヤー取得の `@st.cache_data` 化（再処理遅延解消）。UI 改善: 「領域の確認」名称チェックボックスのラジオ動作・他領域選択名をデフォルト優先・`st.popover` 角座標表示・領域間 divider。領域なし領域の Excel 出力を「no name #」（連番）で自動命名。`region_name_candidates` に全横エッジフォールバック（上端近傍ラベルの取得）を追加。|
| v1.3.0 | 矩形領域抽出（領域選択オプション）追加。図面枠(lw100)内の直交ポリゴン閉領域を検出（コーナー相手判定による縦ギャップ橋渡し・端点接続ベース面探索・面積20%絞り込み）。領域名候補（下端横線分近傍・機器符号/小文字/NOTE/☆除外・RACK例外）をチェックボックスで確定。各領域の角座標表示。領域付き Excel（領域列・領域一覧シート）出力。回帰テスト `tests/regression/test_region_extraction.py`。※本機能は DXF-extract-labels のみ。他プロジェクトへは未伝播（動作確認後に検討）|
| v1.2.0 | Total シート追加（全ファイルのラベル合計集計）。Summary のファイル名に各シートへの内部ハイパーリンクを設定。Invalid シートを機器符号ごとの集計形式（機器符号・個数・ファイル名）に変更。「抽出結果統計」をテーブル表示に変更しデフォルト折りたたみ化。処理後の「処理オプション」表示ブロックを削除 |
| v1.1.0 | タイトル・サブタイトル自動抽出機能追加。ラベルベース図面番号検出（80 単位近接）。複数図面対応（最も右側の図面を自動選択）。座標ベース検出の精度向上。Excel 出力に Title・Subtitle カラム追加 |
| v1.0.0 | 初回リリース。複数ファイル対応・機器符号フィルタリング・妥当性チェック・図面番号抽出・レイヤー選択・Excel 出力 |

---

## ライセンス

株式会社 RDPi 所有。特定顧客向け開発品につきコピー・改版禁止。

---

最終更新: 2026-07-04 (v1.5.28)
