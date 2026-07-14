# REF_DESIGNATOR.md — 機器符号（候補）抽出

> 全体構成・アーキテクチャは [OVERVIEW.md](OVERVIEW.md) を参照。

## 機器符号（候補）抽出パイプライン（`utils/ref_designator.py`、v1.6.0・既定モード）

`reference_designator_candidates.xlsx`（`Patterns` / `ExclusionPatterns` シート）を
正としたパターン・除外リストを実装する。処理はファイル単位で以下の順に行う。

1. **図面枠検出**: 「領域検出の詳細設定」の「図面枠の太さ」（`frame_lineweight`）と
   `color=7` を満たす LINE（modelspace 直置き + フォーマットブロックの INSERT 展開）を
   `region_detector.detect_drawing_frames()` で 4 辺 1 組に集約する。
   **color 条件は必須**（lineweight 単独では無関係な線分を拾って誤検出することを
   `EE6868-500-01C.dxf` で確認済み: lineweight単独=772本→誤検出31枠、
   lineweight+color=7=52本→正しく13枠。2026-07-10）。
   **Model Space に何らかの内容（frame_lines・label_entities のいずれか）が
   あれば常に Model Space のみを対象にする**（v1.8.1・v1.8.2で方針確定）。
   図面枠線を直接の子に持つフォーマットブロック（`JZB_0001`）が Model Space
   ではなく `ICADSX Layout` という Paper Space レイアウトにのみ INSERT
   されている図面が実データに存在する（`EE5322-455-01B.dxf`・
   `EE5322-455-07A.dxf` 等）。Model Space が**完全に空の場合のみ**、他の
   レイアウトを順に試し、内容が見つかったレイアウト自身の frame_lines・
   label_entities のペアを使う（レイアウトをまたいで混在させない）。

   **重要**: Model Space と Paper Space は独立した座標系のため、一方の
   レイアウトの図面枠bboxをもう一方のレイアウトのラベルに適用してはならない。
   v1.8.0では両レイアウトの図形を無条件にまとめて収集していたが、
   `EE6892-455B.dxf`（実際の機器符号ラベル`CB001`等はModel Spaceにあり、
   図面枠だけPaper Spaceにある図面）で、Paper Space由来の枠bboxをModel
   Spaceのラベルに適用してしまい、`CB001`等の大半が「枠外」と誤判定され
   出力から消える不具合が発生した（2026-07-14ユーザー報告）。v1.8.2で
   「Model Spaceに何かあれば常にModel Spaceのみを使う」方針に修正して
   解消。この場合`EE5322-455-01B.dxf`等は再び「図面枠が見つかりません
   でした」という警告が表示されるが、次項7の枠なしフォールバックで
   Model Space全ラベルが正しく抽出されるため実害はない。
2. **フォーマットブロックの構造的除外**: 図面枠線を直接の子として持つブロック
   （実データでは `JZB_*`）を「フォーマットブロック」と判定し、その INSERT 由来の
   TEXT/MTEXT を丸ごと除外する。図面情報欄（タイトル・日付・改訂履歴・設計者名等）と
   図面枠外の位置記号（A-F, 1-8 等）はいずれもこのブロック内に存在することを
   サンプル18件で検証済み。**人名の個別リストは持たない**（増減するため。図面情報欄が
   丸ごと除外されるので人名も自動的に除外される）。
3. **図面枠内判定**: 残った TEXT/MTEXT のうち、検出済み図面枠の bbox（±1 マージン）
   内にあるものだけを対象にする（複数図面ファイルは全枠が対象）。
4. **正規化・パターン判定**: NFKC 正規化（全角→半角）・前後空白除去した上で、
   **括弧より前の部分**でパターン判定する（`R10(2.2K)` → `R10` で判定、出力は
   `R10(2.2K)` のまま）。3パターンのいずれかに一致するか判定する:
   `^(?:[A-Z]+-[A-Z]+[0-9]+[A-Z0-9-]*|[A-Z]+[0-9]+[A-Z0-9-]*|[A-Z]+)$`
5. **除外パターン適用**: 4 で一致したものから、除外パターン（後述）に該当するものを
   さらに除く。残ったものが「機器符号（候補）」（`reference_designator_candidates.xlsx`
   の RemainingUnclassified シートと同じ母集団）。3パターンいずれにも一致しない
   文字列（`(2/5)` 等の記号・注記）・除外パターンに該当したもの（`GND`・`TITLE`・
   `N24` 等）はいずれも機器符号（候補）に含めない＝画面に一切表示しない。
6. **確定パターン適用**（v1.6.3）: 5 の機器符号（候補）のうち、確実に Reference
   Designator と判定してよい形（`CONFIRMED_PATTERN_CATEGORIES`、後述）に一致した
   ものは「確定」として自動採用され、「未確定ラベル」UI には表示しない。
7. **「未確定ラベル」UI でのレビュー**: 6 で確定しなかった残りを「未確定ラベル」と
   して `st.data_editor` に表示する（初期状態は全て未選択）。ユーザーがチェックし
   「選択完了」した分と、6 で自動採用された確定分を合わせたものが最終的な
   機器符号として出力される。表示レイアウト（固定幅テーブルのブラウザー幅に
   応じたレスポンシブな横並び）は v1.6.6 参照。
7. **図面枠が検出できない場合**: 警告を出し、図面枠フィルタなしでファイル全体
   （Model Space + Model 以外の全レイアウトの TEXT/MTEXT 全件）を対象に
   フォールバックする。

> 「機器符号（候補）以外も抽出」ON のときは、このパイプラインを使わず
> `process_multiple_dxf_files()`（図面枠制限なし・フィルタなし）で全ラベルを抽出する。

#### 連動採用（兄弟ラベル・全ファイル横断、v1.7.1）

「未確定ラベル」UI で1件チェック/解除すると、以下の条件に一致する他のラベルへ
アップロード済み**全ファイルにわたって**即座に連動する（`utils/ref_designator.py`）。

- **兄弟ラベル**（`sibling_key()`）: NFKC 正規化後、末尾が数字1〜2桁で、その前の
  文字列が一致するラベル（例: `CN1`・`CN2`・`CN10` は同じキー `CN`）。末尾3桁以上
  （`CB001`）・末尾が数字でない（`X14A`）・数字のみ（`10`）は対象外（`sibling_key()`
  が `None` を返す）。
- **同一ラベル**（全角/半角の表記違いを含む、NFKC正規化後の一致）: 兄弟ラベルの
  対象外の形でも、別ファイルに同じラベルがあれば同期する。
- 他ファイルに存在しないラベルへ勝手に追加されることはない
  （`propagate_selection_all_files()` は既存キーのみ更新）。

**実装方式（正本 dict + キー再生成）**: 初回実装は `st.data_editor` の
`on_change` コールバックから**別インスタンスの `session_state`（`edited_rows`）を
直接書き換える**方式だったが、ブラウザ⇔サーバー間の無限同期フィードバックループで
画面全体が例外なしにフリーズする不具合が発生し撤回した。代わりに、チェック状態の
正本を通常の `session_state`（`unclassified_checked`）に持ち、data_editor の
`key` に `unclassified_ver`（バージョンカウンタ）を含める方式に変更した。

```
1. data_editor は正本 unclassified_checked から作った初期値で描画
   （key = f"unclassified_editor_{unclassified_ver}_{fname}_{suffix}"）
2. rerun時、data_editor の返り値と正本を比較して差分（deltas）を検出
3. 差分があれば ref_designator.propagate_selection_all_files() で正本を更新
   → unclassified_ver を +1 → st.rerun()
4. 次のrunでは新しいkeyの別ウィジェットとして再生成される
   （旧インスタンスはブラウザ側でも破棄され、同期し返す相手が存在しない）
```

差分がない run（再描画直後・「選択完了」クリック時）は何もしないため、再実行が
連鎖することはない。「選択完了」時の採用集合は、data_editor の返り値ではなく
`unclassified_checked`（正本）から直接収集する。汎用パターンとしては
`~/.claude/skills/streamlit/SKILL.md` §6「st.data_editor 間のライブ連動」に記録済み。

> 単体テスト: `tests/unit/test_sibling_autocheck.py`（16件）。


## 機器符号（候補）抽出パターン（`utils/ref_designator.py`、v1.6.0）

`reference_designator_candidates.xlsx`（`Patterns` / `ExclusionPatterns` シート）が
定義の正。同ファイルは `/Users/ryozo/Dropbox/Workspace/Reference Designator/Labels/` に
ある（本リポジトリ外・実データからパターンを検討した作業ファイル）。

### 候補パターン（`CANDIDATE_PATTERN`）

判定は NFKC 正規化・括弧より前の部分（`R10(2.2K)` → `R10`）に対して行う。

```
^(?:[A-Z]+-[A-Z]+[0-9]+[A-Z0-9-]*|[A-Z]+[0-9]+[A-Z0-9-]*|[A-Z]+)$
```

| パターン種別 | 例 |
|------------|-----|
| 英字繰返し-英字繰返し+数字繰返し+英数字/ハイフン任意 | `CN-IF2-1`、`AAC1B4-07` |
| 英字繰返し+数字繰返し+英数字/ハイフン任意 | `R10`、`CN3`、`AAC1B4-07` |
| 英字繰返しのみ | `FB`、`CNCNT` |
| 括弧付き（括弧より前で判定、出力は原文） | `FB()`、`R10(2.2K)`、`MSS(MOTOR)` |

### 除外パターン（`EXCLUSION_REGEX_CATEGORIES` / `EXCLUSION_EXACT_CATEGORIES`）

| カテゴリ | 種別 | パターン/内容 | 例 |
|---------|------|--------------|-----|
| 図形枠外の位置記号 | 正規表現 | `^[A-Z]$` | `A` |
| 電源端子（末尾記号） | 正規表現 | `.*[+-]$` | `R10-` |
| ケーブル線径 | 正規表現 | `^AWG[0-9]*$` | `AWG14` |
| ユニット名（RACK） | 正規表現 | `^RACK[0-9]*(-[0-9]+)?$` | `RACK1`、`RACK1-2` |
| 図番 | 正規表現 | `^[A-Z]{2}[0-9]{4}-[0-9]{3}(-[0-9]{2})?[A-Z]?$` | `EE1234-500-01A` |
| 図面情報枠（JIS/DWG） | 正規表現 | `^(JIS\|DWG)[A-Z0-9]*$` | `JIS123` |
| 機器端子の行番号 | 正規表現 | `^[AB][0-9]+$` | `A1`、`B12` |
| 保護接地端子 | 正規表現 | `^PE[0-9]+$` | `PE1`、`PE2` |
| 相線・電源レール | 正規表現（v1.6.4で拡張） | `^[LNP][0-9]+[A-Z]*$` | `L1`、`N24`、`P24`、`N24AB`（末尾英大文字は0字以上） |
| PLC/内部信号名 | 正規表現 | `^X[A-Z]+$` | `XRST`、`XMCON`（`X1`等 `X+数字` は除外対象外） |
| 回路の説明 | 正規表現（v1.6.3、語追加はv1.6.4） | キーワード+数字1桁まで許容（例 `^(?:GND\|POWER\|...)[0-9]?$`） | `GND`、`POWER`、`COM`、`OUT2`、`IN1`、`AOUT`、`LG`、`MR`、`MRR`、`RX`、`TX`、`CLR`、`ZERO`、`YOUT`、`AG`（`OUT12`等2桁以上は対象外） |
| 普通名詞 | 完全一致（約100語） | 端子/スイッチの機能説明語 | `ALARM`、`MOTOR` |
| ユニット/モジュール名 | 完全一致 | `CTC`、`EFEM`、`SHIELD`、`LA`、`LB`、`CASE`、`SH` 等 | |
| ケーブル色 | 完全一致 | JIS配線色略号 | `BK`、`WH`、`RD` 等 |
| 図面情報枠タイトル項目 | 完全一致 | `TITLE`、`DATE`、`REVISION` 等 | |

**構造的除外が優先**: 図面情報欄・図面枠外の位置記号は、上記コンテンツベースの除外
リストに加えて「機器符号（候補）抽出パイプライン」節の手順2（フォーマットブロック
除外）で構造的にも除外される。人名は増減するため個別リストを持たず、構造的除外のみで
対応する。

> 除外パターンの検討過程・実データでの件数検証は
> `reference_designator_candidates.xlsx` の `ExclusionPatterns` シートを参照。

### 確定パターン（`CONFIRMED_PATTERN_CATEGORIES`、v1.6.3・4パターン追加はv1.6.4）

機器符号（候補）のうち、確実に Reference Designator と判定してよい形。一致した
ラベルは「未確定ラベル」UI でのレビューを経ずに自動採用される（`matched_confirmed_category()`
/ `is_confirmed_designator()` / `split_confirmed()`）。より限定的なパターンを先に
判定する順でリストされており（下表の掲載順＝判定順）、いずれか1つでも一致すれば
確定扱いとなる（複数一致した場合は最初に一致したカテゴリ名が記録される）。

| カテゴリ | 判定基準 | 正規表現 | 例 |
|---------|---------|--------|-----|
| cn_single_digit（v1.6.4） | judgment | `^CN[0-9]$` | `CN1`、`CN3` |
| cn_if_prefix（v1.6.4） | judgment | `^CN-IF.*$` | `CN-IF21`、`CN-IF2-1` |
| r_paren_suffix（v1.6.4） | full | `^R[0-9]+\(.*\)$` | `R10(2.2K)`、`R1000(2.2K)` |
| vr_paren_suffix（v1.6.4） | full | `^VR[0-9]+\(.*\)$` | `VR5(10K)` |
| letters_digits_2or3 | judgment | `^[A-Z]+[0-9]{2,3}$` | `R10`、`D100` |
| letters_digits_2or3_letter | judgment | `^[A-Z]+[0-9]{2,3}[A-Z]$` | `R10A`、`A007A` |
| hyphen_letters_digits_notail | judgment | `^[A-Z]+-[A-Z]+[0-9]+$` | `CN-IF21`（cn_if_prefixが先に一致するため実際はcn_if_prefix扱い） |
| single_letter_digits_except_ab | judgment | `^[C-Z][0-9]+$` | `D1`、`D1000`（桁数不問） |

A,B の除外は `single_letter_digits_except_ab` のみに適用する（`letters_digits_2or3`
系には適用しない。A1/B12 等は既存の `terminal_row_letter_digit` 除外パターンで
確定パターン判定より前に除外されるため実害はない）。

**判定基準（`basis`）**: ほとんどのカテゴリは括弧より前の判定用文字列（`judgment`、
候補パターン・除外パターンと同じ基準）に対して正規表現を適用するが、
`r_paren_suffix`/`vr_paren_suffix` は括弧の中身自体を問うパターンのため、
正規化済みラベル全体（`full`、括弧を含む）に対して判定する。このため
`matched_confirmed_category()` の引数は v1.6.4 で「判定用文字列（judgment）」から
「正規化済みラベル全体」に変更した（`judgment` のみが必要なカテゴリは内部で
`_judgment_text()` を都度計算する）。

---

## 判断ログ（採用/非採用の記録・蓄積、`utils/decision_log.py`、v1.7.0）

「未確定ラベル」UI でユーザーが行った採用/非採用の判断を蓄積し、確定パターン・
除外パターンの完成度向上（`CONFIRMED_PATTERN_CATEGORIES`/`EXCLUSION_*_CATEGORIES`
の見直し）に活用するための記録機能。「選択完了」ボタン押下時、レビュー対象だった
未確定ラベル全件（採用したもの・しなかったもの両方）をエントリ化して記録する
（確定パターンで自動採用された分は対象外＝人手判断の記録のみ）。

### 記録エントリのスキーマ

`build_entries()` が生成する CSV 行（列: `timestamp, source, file_name,
drawing_number, label, decision, count, app_version, patterns_version`）。

- `decision`: `adopted`（採用）/ `rejected`（非採用）
- `count`: そのファイル内でのラベルの出現個数
- `patterns_version`: 記録時点の `ref_designator.PATTERNS_VERSION`
  （パターンリストを変更したら値を上げる。パターン更新後もまだ人手判断され続けて
  いるラベルの特定に使う）
- 追記型・重複排除なし（同一ラベルの判断頻度が分析上の信号になるため）

### バックエンドの自動選択（`pick_backend()`）

| 環境 | バックエンド | 保存先 |
|------|------------|--------|
| Streamlit Community Cloud | `GitHubBackend` | `st.secrets['github']`（`token`/`repo`/`path`/`branch`）で指定したログ専用リポジトリの CSV（GitHub Contents API で追記）|
| ローカル・Windows アプリ | `FileBackend`（既定） | `~/Documents/DXF-extract-labels/decision_log.csv`。環境変数 `DXF_DECISION_LOG_PATH` で変更可（Dropbox 等の共有フォルダを指定すれば複数 PC のログを一元化できる）|

`st.secrets['github']` が読めない・未設定（`secrets.toml` が存在しないローカル環境
を含む）の場合は例外を握りつぶし `FileBackend` にフォールバックする。

**ログ専用リポジトリはアプリ本体のリポジトリとは別にする**（現在の運用先:
`rdpishibashi/dxf-label-decisions`、private）。本体リポジトリの `decision_log.csv`
にコミットすると、コミットのたびに Streamlit Community Cloud がアプリを
再デプロイしてしまうため。

`GitHubBackend.append()` は競合（他セッションが同時に追記し `sha` が古くなった
場合、HTTP 409/422）時に `sha` を取り直して1回だけリトライする。

### Streamlit Cloud での Secrets 設定例

```toml
# .streamlit/secrets.toml（ローカル動作確認用。Cloud では管理画面の Secrets に設定）
[github]
token = "ghp_..."            # 対象リポジトリへの contents:write 権限を持つ PAT
repo = "rdpishibashi/dxf-label-decisions"
# path, branch は省略可（既定 decision_log.csv / main）
```

### 記録失敗時のフォールバック

`decision_log.record()` は例外を外に出さず `(成功?, メッセージ)` を返す。記録に
失敗しても抽出処理・Excel ダウンロードは止めない。失敗時は抽出結果セクションに
警告を表示し、そのエントリを CSV として個別ダウンロードできるボタン
（`entries_to_csv_bytes()`）を提供する。

### 判断ログの分析（`tools/reference_designator_analyzer.py`）

蓄積した `decision_log.csv` を集計し、確定パターン・除外パターンの候補を機械的に
提案する「判断ログ分析」タブ。詳細は「開発用ツール」節を参照。

回帰テスト: `tests/unit/test_decision_log.py`（バックエンド・エントリ構築の単体
テスト）。

---


## 開発用ツール（`tools/`）

本体アプリ（`app.py`）とは別に、機器符号（候補）パターン・除外パターンを検討する
ための独立した Streamlit ツールを `tools/` 配下に置く。deploy 対象の本体アプリの
一部ではなく、開発者がパターンを見直す際に使うローカル/内部向けツール。

### `reference_designator_analyzer.py`（v1.6.0、確定パターンは v1.6.2〜v1.6.3）

```
streamlit run tools/reference_designator_analyzer.py
```

複数の `extracted_labels*.xlsx`（本体アプリの「機器符号（候補）以外も抽出」ON で
出力した Excel。`Total` シートにラベル・個数を持つ）を入力とし、
`reference_designator_candidates.xlsx` と同じ7シート構成（`ReferenceDesignators` /
`Patterns` / `PatternSignatures` / `ExclusionPatterns` / `ConfirmedPatterns` /
`ConfirmedDesignators` / `RemainingUnclassified`）の分析用 Excel を生成する。

- **入力**: `st.file_uploader`（複数可）とローカルフォルダパス（1行1つ、glob パターン・
  再帰検索の有無を指定可）を併用できる。
- **パターン・除外・確定リストの定義**: `utils/ref_designator.py`（本体アプリの判定
  ロジック）を単一の正として参照する。本ツール独自にパターン・除外語・確定パターンを
  定義し直さない。`utils/ref_designator.py` に `PATTERN_CATEGORIES` /
  `EXCLUSION_REGEX_CATEGORIES` / `EXCLUSION_EXACT_CATEGORIES` /
  `CONFIRMED_PATTERN_CATEGORIES` / `classify_judgment_detailed()` /
  `matched_pattern_name()` / `matched_confirmed_category()` を公開 API として
  追加し、本ツールと本体アプリが常に同じ判定結果を返すようにしている
  （確定パターンは v1.6.2 で本ツール専用として導入し、v1.6.3 で
  `utils/ref_designator.py` 側にも取り込んで本体アプリの判定に反映した。
  「確定パターン」節参照）。
- **`ConfirmedPatterns` シート**（`ExclusionPatterns` と `ConfirmedDesignators` の間）:
  確定パターン4カテゴリの正規表現・該当ラベル数・該当個数合計。
- **`ConfirmedDesignators` シート**（v1.6.3、`ConfirmedPatterns` の直後）: 確定パターンに
  実際に一致したラベルの一覧（ラベル・個数・出現ファイル数・一致した確定カテゴリ、
  個数降順）。
- **パターン表記**（`PatternSignatures` シート）: 数字列の前の部分はそのまま、直後の数字は
  桁数表記（1桁→`1`、2桁→`12`…）、後ろに続きがなければ `+1*` に集約、続きがあれば
  `+` の後に英字繰返し=`A*`／数字繰返し=`1*`／ハイフンはそのまま、というノーテーション
  （2026-07 ユーザーと確定。本体アプリでは使わない分析専用の表記のため
  `utils/ref_designator.py` には持ち込まず本ツールに閉じている）。
- **本体アプリとの違い（既知の制限）**: 本ツールは集計済みラベル一覧（座標情報なし）
  だけを見るため、本体アプリが行う図面枠・図面情報欄の構造的除外はできない。人名は
  個別リストを持たない設計のため、図面情報欄由来のラベル（人名等）がパターン一致・
  除外非該当のまま `RemainingUnclassified` に残ることがある。パターン表記のみ本ツール
  専用の分析ロジック（本体アプリの実際の抽出結果には影響しない）。

回帰テスト: `tests/regression/test_reference_designator_analyzer.py`（集計・分類・
Excel出力の配線を検証。パターン/除外/確定リストの定義自体は `test_ref_designator.py` 側で検証）。

#### 判断ログ分析タブ（v1.7.0）

`app()` はタブ構成（「extracted_labels 集計」/「判断ログ分析」）になり、上記は
`_app_extracted_labels()` に、判断ログ分析は `_app_decision_log()` に分離されている。

本体アプリが蓄積した `decision_log.csv`（「判断ログ」節参照）を読み込み、ラベル
ごとの採用率・非採用率から確定パターン候補・除外パターン候補を提案する。

- **入力**: `st.file_uploader`（複数可）・ローカル/Dropbox フォルダパス指定（glob
  `decision_log*.csv`）・GitHub のログ専用リポジトリから直接取得（`repo`/`path`/
  `branch`/トークンを指定。トークン省略時は `st.secrets['github']['token']` を使用）
  の3方式を併用できる。
- **集計**（`aggregate_decision_log()`）: 正規化ラベル単位で `adopted`/`rejected`
  の個数を合算する。
- **提案**（`build_decision_log_suggestions()`）: 合計出現回数が
  `最小出現回数`（既定3）未満は「様子見（サンプル不足）」。採用率が
  `確定パターン候補とする採用率の下限`（既定1.0=常に採用）以上なら「確定パターン
  候補」。非採用率が `除外パターン候補とする非採用率の下限`（既定1.0）以上なら
  「除外パターン候補」。閾値は UI のスライダーで調整可能。
- **出力**: `DecisionLogSummary` 1シートの Excel（`build_decision_log_workbook()`）。
- 提案はあくまで機械的な集計に基づく候補であり、実際にパターンへ反映するかは
  `utils/ref_designator.py` の `CONFIRMED_PATTERN_CATEGORIES` /
  `EXCLUSION_*_CATEGORIES` を人手で判断・編集する（自動反映はしない）。

回帰テスト: `tests/regression/test_reference_designator_analyzer.py` の
`test_aggregate_decision_log_*` / `test_build_decision_log_*`。

---

