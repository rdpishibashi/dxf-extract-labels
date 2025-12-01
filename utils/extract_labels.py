import ezdxf
import re
import os
import sys
from typing import List, Tuple, Dict, Optional

# 共通ユーティリティをインポート
from .common_utils import process_circuit_symbol_labels


def get_layers_from_dxf(dxf_file):
    """
    DXFファイルからレイヤー一覧を取得する

    Args:
        dxf_file: DXFファイルパス

    Returns:
        list: レイヤー名のリスト
    """
    try:
        doc = ezdxf.readfile(dxf_file)
        # レイヤーテーブルからすべてのレイヤー名を取得
        layer_names = [layer.dxf.name for layer in doc.layers]
        return sorted(layer_names)  # アルファベット順にソート
    except Exception as e:
        print(f"レイヤー一覧の取得中にエラーが発生しました: {str(e)}")
        return []


def clean_mtext_format_codes(text: str, debug=False) -> str:
    r"""
    MTEXTのフォーマットコードを除去して完全なテキスト内容を保持する

    Args:
        text: MTEXTの生テキスト
        debug: デバッグ情報を出力するかどうか

    Returns:
        str: 完全なテキスト内容（\Pなどの構造を保持）
    """
    if not text:
        return ""

    # バックスラッシュと円マークの両方に対応
    # 日本語環境では ¥ (円マーク, Unicode U+00A5) が使われることがある
    # 英語環境では \ (バックスラッシュ, Unicode U+005C) が使われる

    # まず、円マークをバックスラッシュに正規化
    normalized_text = text.replace('¥', '\\')

    # フォーマット制御コードのみを除去し、テキスト構造（\\Pなど）は保持
    cleaned = normalized_text

    # フォント制御コード \f...;を除去
    cleaned = re.sub(r'\\f[^;]*;', '', cleaned)

    # 高さ制御コード \H...;を除去
    cleaned = re.sub(r'\\H[^;]*;', '', cleaned)

    # 幅制御コード \W...;を除去
    cleaned = re.sub(r'\\W[^;]*;', '', cleaned)

    # カラー制御コード \C...;を除去
    cleaned = re.sub(r'\\C[^;]*;', '', cleaned)

    # 配置制御コード \A...;を除去
    cleaned = re.sub(r'\\A[^;]*;', '', cleaned)

    # 追跡制御コード \T...;を除去
    cleaned = re.sub(r'\\T[^;]*;', '', cleaned)

    # その他の制御コード（文字;形式）を除去
    # ただし、\\Pは保持する（テキスト構造として重要）
    cleaned = re.sub(r'\\(?!P)[^\\;]*;', '', cleaned)

    # スペース制御 \~ を通常のスペースに変換
    cleaned = cleaned.replace('\\~', ' ')

    # バックスラッシュエスケープを処理
    cleaned = cleaned.replace('\\\\', '\\')
    cleaned = cleaned.replace('\\{', '{')
    cleaned = cleaned.replace('\\}', '}')

    # 複数の空白を単一の空白に変換
    cleaned = re.sub(r' +', ' ', cleaned)

    result = cleaned.strip()

    if debug:
        print(f"MTEXT cleaning: '{text}' -> '{result}'")

    return result


def extract_text_from_entity(entity, debug=False) -> Tuple[str, str, Tuple[float, float]]:
    """
    TEXTまたはMTEXTエンティティからテキストと座標を抽出する

    Args:
        entity: DXFエンティティ（TEXTまたはMTEXT）
        debug: デバッグ情報を出力するかどうか

    Returns:
        tuple: (生テキスト, クリーンテキスト, (X座標, Y座標))
    """
    try:
        # 座標を取得 - MTEXTとTEXTで異なる属性を使用
        x, y = 0.0, 0.0

        if entity.dxftype() == 'MTEXT':
            # MTEXTの場合、グループコード10,20を確認
            if hasattr(entity.dxf, 'insert'):
                x, y = entity.dxf.insert[0], entity.dxf.insert[1]
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'x') and hasattr(entity.dxf, 'y'):
                x, y = entity.dxf.x, entity.dxf.y
            else:
                # 直接属性を確認
                try:
                    x = getattr(entity.dxf, 'x', 0.0)
                    y = getattr(entity.dxf, 'y', 0.0)
                except:
                    x, y = 0.0, 0.0
        elif entity.dxftype() == 'TEXT':
            # TEXTの場合
            if hasattr(entity.dxf, 'insert'):
                x, y = entity.dxf.insert[0], entity.dxf.insert[1]
            elif hasattr(entity.dxf, 'location'):
                x, y = entity.dxf.location[0], entity.dxf.location[1]

        # 生テキストを取得
        raw_text = ""

        if entity.dxftype() == 'TEXT':
            # TEXTエンティティの場合
            if hasattr(entity.dxf, 'text'):
                raw_text = entity.dxf.text
        elif entity.dxftype() == 'MTEXT':
            # MTEXTエンティティの場合、複数の方法でテキストを取得

            # 方法1: entity.dxf.text
            if hasattr(entity.dxf, 'text'):
                raw_text = entity.dxf.text

            # 方法2: entity.text (ezdxfのプロパティ)
            if not raw_text and hasattr(entity, 'text'):
                try:
                    raw_text = entity.text
                except:
                    pass

            # 方法3: plain_text() メソッド
            if not raw_text and hasattr(entity, 'plain_text'):
                try:
                    raw_text = entity.plain_text()
                except:
                    pass

        # フォーマットコードをクリーンアップ（エンティティタイプに応じて）
        if raw_text:
            if entity.dxftype() == 'MTEXT':
                # MTEXT の場合はフォーマットコードを除去
                clean_text = clean_mtext_format_codes(raw_text, debug)
            else:
                # TEXT の場合はそのまま使用（フォーマットコードは含まれない）
                clean_text = raw_text.strip()
        else:
            clean_text = ""


        return raw_text, clean_text, (x, y)

    except Exception as e:
        return "", "", (0.0, 0.0)


def extract_drawing_numbers(text: str, debug=False) -> List[str]:
    """
    テキストから図面番号フォーマットに一致する文字列を抽出する

    Args:
        text: 検索対象のテキスト（クリーンテキスト）
        debug: デバッグ情報を出力するかどうか

    Returns:
        list: 図面番号のリスト
    """
    # 図面番号の正確なパターンを定義
    # 例: DE5313-008-02B（英大文字x2+数字x4+"-"+数字x3+"-"+数字x2+英大文字）
    patterns = [
        r'[A-Z]{2}\d{4}-\d{3}-\d{2}[A-Z]',  # DE5313-008-02B 型（正確なフォーマット）
    ]

    drawing_numbers = []


    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            # 重複を避けて追加
            if match.upper() not in [dn.upper() for dn in drawing_numbers]:
                drawing_numbers.append(match.upper())


    return drawing_numbers


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    2点間の距離を計算する

    Args:
        coord1: (X座標, Y座標)
        coord2: (X座標, Y座標)

    Returns:
        float: ユークリッド距離
    """
    import math
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    return math.sqrt(dx * dx + dy * dy)


def is_single_uppercase_letter(text: str) -> bool:
    """
    英大文字1文字かどうかを判定（半角・全角両対応）

    Args:
        text: 判定対象の文字列

    Returns:
        bool: 英大文字1文字ならTrue
    """
    if len(text) != 1:
        return False
    # 半角英大文字 A-Z
    if 'A' <= text <= 'Z':
        return True
    # 全角英大文字 Ａ-Ｚ（\uff21-\uff3a）
    if '\uff21' <= text <= '\uff3a':
        return True
    return False


def extract_title_and_subtitle(
    all_labels: List[Tuple[str, Tuple[float, float]]],
    drawing_numbers: Optional[List[Tuple[str, Tuple[float, float]]]],
    debug: bool = False
) -> Dict[str, Optional[str]]:
    """
    テキストラベルの位置関係からタイトルとサブタイトルを抽出する

    抽出条件:
    - 「TITLE」という文字列の右側近辺にタイトルとサブタイトルが配置
    - タイトルは「REVISION」の下方向にある
    - サブタイトルはタイトルの直下に配置
    - サブタイトルは図番のすぐ上方に配置
    - タイトルは複数単語の場合、半角スペースで結合
    - サブタイトルも複数単語の場合、半角スペースで結合
      ただし、最後が英大文字1文字の場合は除外

    Args:
        all_labels: (ラベル, (X座標, Y座標))のリスト（全テキスト）
        drawing_numbers: (図面番号, (X座標, Y座標))のリスト（図番の位置参照用）
        debug: デバッグ情報を出力するかどうか

    Returns:
        dict: {'title': 'タイトル文字列', 'subtitle': 'サブタイトル文字列'}
    """
    if not all_labels:
        return {'title': None, 'subtitle': None}

    title_text = None
    subtitle_text = None

    # 1. 「TITLE」と「REVISION」の位置を特定
    # 複数のTITLEラベルがある場合は、一番右側（X座標が最大）のものを採用
    title_label_positions = []
    revision_label_positions = []

    for label, coords in all_labels:
        label_upper = label.upper().strip()
        if label_upper == 'TITLE':
            title_label_positions.append(coords)
            if debug:
                print(f"TITLE ラベル発見: ({coords[0]:.2f}, {coords[1]:.2f})")
        elif label_upper == 'REVISION':
            revision_label_positions.append(coords)
            if debug:
                print(f"REVISION ラベル発見: ({coords[0]:.2f}, {coords[1]:.2f})")

    if not title_label_positions:
        if debug:
            print("TITLE ラベルが見つかりませんでした")
        return {'title': None, 'subtitle': None}

    # 一番右側のTITLEラベルを選択（X座標が最大）
    title_label_pos = max(title_label_positions, key=lambda pos: pos[0])
    if debug:
        print(f"採用されたTITLE: ({title_label_pos[0]:.2f}, {title_label_pos[1]:.2f}) (X座標が最大)")

    # 一番右側のREVISIONラベルを選択（複数ある場合）
    revision_label_pos = None
    if revision_label_positions:
        revision_label_pos = max(revision_label_positions, key=lambda pos: pos[0])
        if debug:
            print(f"採用されたREVISION: ({revision_label_pos[0]:.2f}, {revision_label_pos[1]:.2f}) (X座標が最大)")

    # 2. タイトル候補を収集
    # 条件: TITLEの右側（X座標 > title_label_pos[0]）
    #       REVISIONがあれば、その下方向（Y座標 < revision_label_pos[1]）
    title_candidates = []

    # TITLEラベルの右側の閾値（TITLEの右側近辺）
    title_right_threshold = title_label_pos[0] + 10  # TITLEのX座標より右側
    title_proximity_x = 80  # X方向の近辺範囲（TITLEのすぐ右側のみ）
    title_proximity_y = 50   # Y方向の近辺範囲（同じ行とみなす範囲）

    for label, coords in all_labels:
        # TITLEやREVISIONのラベル自体は除外
        label_upper = label.upper().strip()
        if label_upper in ['TITLE', 'REVISION']:
            continue

        # 図面番号も除外
        if drawing_numbers:
            is_drawing_number = any(dn == label for dn, _ in drawing_numbers)
            if is_drawing_number:
                continue

        # TITLEの右側近辺かチェック
        x_diff = coords[0] - title_label_pos[0]
        y_diff = abs(coords[1] - title_label_pos[1])

        # TITLEの右側かつ近辺
        if x_diff > 10 and x_diff < title_proximity_x:
            # REVISIONがある場合、その下方向かチェック
            if revision_label_pos:
                # REVISIONより下方向（Y座標が小さい）
                if coords[1] < revision_label_pos[1]:
                    title_candidates.append((label, coords))
                    if debug:
                        print(f"タイトル候補: '{label}' at ({coords[0]:.2f}, {coords[1]:.2f})")
            else:
                # REVISIONがない場合は、TITLEの右側近辺を候補とする
                title_candidates.append((label, coords))
                if debug:
                    print(f"タイトル候補: '{label}' at ({coords[0]:.2f}, {coords[1]:.2f})")

    if not title_candidates:
        if debug:
            print("タイトル候補が見つかりませんでした")
        return {'title': None, 'subtitle': None}

    # 2.5. 重複するテキスト（座標が近く、ラベルが同じ）を除去
    # 座標が非常に近い（許容誤差内）かつラベルが同じものを1つにまとめる
    deduplicated_candidates = []
    coord_tolerance = 1.0  # 座標の許容誤差

    for label, coords in title_candidates:
        is_duplicate = False
        for existing_label, existing_coords in deduplicated_candidates:
            # ラベルが同じで、座標が非常に近い場合は重複とみなす
            if label == existing_label:
                x_diff = abs(coords[0] - existing_coords[0])
                y_diff = abs(coords[1] - existing_coords[1])
                if x_diff <= coord_tolerance and y_diff <= coord_tolerance:
                    is_duplicate = True
                    if debug:
                        print(f"重複除去: '{label}' at ({coords[0]:.2f}, {coords[1]:.2f})")
                    break

        if not is_duplicate:
            deduplicated_candidates.append((label, coords))

    if debug:
        print(f"重複除去後: {len(title_candidates)} -> {len(deduplicated_candidates)} 候補")

    title_candidates = deduplicated_candidates

    if not title_candidates:
        if debug:
            print("重複除去後、タイトル候補がなくなりました")
        return {'title': None, 'subtitle': None}

    # 3. タイトル候補をY座標でグルーピング（同じ行の単語を結合）
    # Y座標が近い（同じ行）ものをグループ化
    y_tolerance = 5.0  # Y座標の許容誤差
    grouped_candidates = []

    sorted_candidates = sorted(title_candidates, key=lambda x: (-x[1][1], x[1][0]))  # Y降順、X昇順

    current_group = []
    current_y = None

    for label, coords in sorted_candidates:
        if current_y is None or abs(coords[1] - current_y) <= y_tolerance:
            current_group.append((label, coords))
            current_y = coords[1]
        else:
            if current_group:
                grouped_candidates.append(current_group)
            current_group = [(label, coords)]
            current_y = coords[1]

    if current_group:
        grouped_candidates.append(current_group)

    if debug:
        print(f"グルーピング結果: {len(grouped_candidates)} グループ")
        for i, group in enumerate(grouped_candidates):
            print(f"  グループ {i+1}: {[label for label, _ in group]}")

    # 4. TITLEラベルに最も近いグループをタイトル行とする
    # 各グループの最小X座標を計算し、TITLEに最も近い（最小X座標が最も小さい）グループを選択
    if grouped_candidates:
        # 各グループの最小X座標を計算
        groups_with_min_x = []
        for group in grouped_candidates:
            min_x = min([coords[0] for _, coords in group])
            avg_y = sum([coords[1] for _, coords in group]) / len(group)
            groups_with_min_x.append((group, min_x, avg_y))

        # Y座標が最も大きい（上の方にある）グループ群を特定
        max_y = max([avg_y for _, _, avg_y in groups_with_min_x])
        y_threshold = 10.0  # タイトルとサブタイトルの間の許容範囲

        # 上の方にあるグループの中で、X座標が最小（TITLEに最も近い）グループをタイトルとする
        title_candidates_filtered = [(group, min_x, avg_y) for group, min_x, avg_y in groups_with_min_x if avg_y >= max_y - y_threshold]
        title_group_info = min(title_candidates_filtered, key=lambda x: x[1])
        title_group = title_group_info[0]

        if debug:
            print(f"タイトルグループ選択: 最小X={title_group_info[1]:.2f}, Y={title_group_info[2]:.2f}")
        # X座標順にソートして結合
        title_group_sorted = sorted(title_group, key=lambda x: x[1][0])
        title_text = ' '.join([label for label, _ in title_group_sorted])
        title_y_coord = title_group_sorted[0][1][1]  # タイトルのY座標

        if debug:
            print(f"タイトル: '{title_text}' (Y={title_y_coord:.2f})")

        # 5. サブタイトルを探す（タイトルの直下）
        # タイトルの直下 = Y座標がタイトルより小さく、かつTITLEに近い位置にあるグループ
        subtitle_candidates = []

        # タイトルグループのX座標範囲を取得
        title_min_x = min([coords[0] for _, coords in title_group])
        title_max_x = max([coords[0] for _, coords in title_group])
        x_tolerance = 100.0  # サブタイトルとタイトルのX座標の許容範囲

        for group, min_x, avg_y in groups_with_min_x:
            if avg_y < title_y_coord:  # タイトルより下
                # タイトルと同じようなX座標範囲にあるグループのみを候補とする
                group_max_x = max([coords[0] for _, coords in group])
                # グループのX座標範囲がタイトルと重なる、または近い場合のみ
                if min_x <= title_max_x + x_tolerance and group_max_x >= title_min_x - x_tolerance:
                    subtitle_candidates.append((group, min_x, avg_y))

        if subtitle_candidates:
            # Y座標がタイトルに最も近い（最も大きい）グループを選択
            # 同じY座標の場合は、X座標が最も小さい（TITLEに近い）グループを選択
            subtitle_group_info = max(subtitle_candidates, key=lambda x: (x[2], -x[1]))
            subtitle_group = subtitle_group_info[0]

            # サブタイトルは通常1つのテキスト要素だが、複数ある場合は結合
            subtitle_group_sorted = sorted(subtitle_group, key=lambda x: x[1][0])
            subtitle_labels = [label for label, _ in subtitle_group_sorted]

            # 最後の要素が英大文字1文字の場合は除外
            if len(subtitle_labels) > 1 and is_single_uppercase_letter(subtitle_labels[-1]):
                if debug:
                    print(f"サブタイトルから英大文字1文字を除外: '{subtitle_labels[-1]}'")
                subtitle_labels = subtitle_labels[:-1]

            subtitle_text = ' '.join(subtitle_labels)
            subtitle_y_coord = subtitle_group_sorted[0][1][1]

            if debug:
                print(f"サブタイトルグループ選択: 最小X={subtitle_group_info[1]:.2f}, Y={subtitle_group_info[2]:.2f}")
                print(f"サブタイトル: '{subtitle_text}' (Y={subtitle_y_coord:.2f})")

            # 6. 図番との位置関係で検証（オプション）
            if drawing_numbers:
                # 図番のY座標の最大値を取得
                max_drawing_y = max([coords[1] for _, coords in drawing_numbers])

                # サブタイトルが図番の上方にあるか確認
                if subtitle_y_coord > max_drawing_y:
                    if debug:
                        print(f"検証OK: サブタイトル(Y={subtitle_y_coord:.2f}) は 図番(Y={max_drawing_y:.2f}) の上方")
                else:
                    if debug:
                        print(f"警告: サブタイトル(Y={subtitle_y_coord:.2f}) が 図番(Y={max_drawing_y:.2f}) の下方にあります")
                    # 警告のみで、サブタイトルは採用

    return {'title': title_text, 'subtitle': subtitle_text}


def determine_drawing_number_types(
    drawing_numbers: List[Tuple[str, Tuple[float, float]]],
    all_labels: Optional[List[Tuple[str, Tuple[float, float]]]] = None,
    filename: Optional[str] = None,
    debug: bool = False
) -> Dict[str, str]:
    """
    ラベルと座標に基づいて図番と流用元図番を判別する（改善版）

    判別ルール:
    1. ファイル名と一致する図面番号を「図番」とする
    2. 「流用元図番」ラベルの近くにある図面番号を「流用元図番」とする
    3. 「DWG No.」ラベルの近くにある図面番号を「図番」として確認
    4. フォールバック: 座標ベース（右下が図番、左上が流用元図番）

    Args:
        drawing_numbers: (図面番号, (X座標, Y座標))のリスト
        all_labels: (ラベル, (X座標, Y座標))のリスト（全テキスト）
        filename: ファイル名（拡張子なし）
        debug: デバッグ情報を出力するかどうか

    Returns:
        dict: {'main_drawing': '図番', 'source_drawing': '流用元図番'}
    """
    if len(drawing_numbers) == 0:
        return {'main_drawing': None, 'source_drawing': None}

    if len(drawing_numbers) == 1:
        return {'main_drawing': drawing_numbers[0][0], 'source_drawing': None}

    main_drawing = None
    source_drawing = None

    # ファイル名から推定される図番を取得
    file_stem = None
    if filename:
        from pathlib import Path
        file_stem = Path(filename).stem

    # 1. ファイル名と一致する図面番号を図番とする
    if file_stem:
        for dn, coords in drawing_numbers:
            if dn == file_stem or dn in file_stem or file_stem in dn:
                main_drawing = dn
                if debug:
                    print(f"図番をファイル名から判別: {main_drawing}")
                break

    # 2. ラベルベースの判別（all_labelsが提供されている場合）
    if all_labels:
        # 「流用元図番」ラベルを探す
        source_label_positions = []
        for label, coords in all_labels:
            # 「流用元図番」「流用元」などのラベルを検出
            if '流用元図番' in label or '流用元' in label:
                source_label_positions.append(coords)
                if debug:
                    print(f"流用元図番ラベル発見: '{label}' at ({coords[0]:.2f}, {coords[1]:.2f})")

        # 「DWG No.」ラベルを探す
        dwg_label_positions = []
        for label, coords in all_labels:
            label_upper = label.upper().replace('\n', '').replace('\r', '').replace(' ', '')
            if 'DWG' in label_upper and 'NO' in label_upper:
                dwg_label_positions.append(coords)
                if debug:
                    print(f"DWG No.ラベル発見: '{label}' at ({coords[0]:.2f}, {coords[1]:.2f})")

        # 流用元図番を「流用元図番」ラベルに最も近い図面番号から判別
        if source_label_positions:
            min_distance = float('inf')
            closest_dn = None

            for dn, dn_coords in drawing_numbers:
                # 既に図番として判定されている場合はスキップ
                if main_drawing and dn == main_drawing:
                    continue

                for label_coords in source_label_positions:
                    distance = calculate_distance(dn_coords, label_coords)
                    if debug:
                        print(f"  {dn} から 流用元ラベル までの距離: {distance:.2f}")

                    if distance < min_distance:
                        min_distance = distance
                        closest_dn = dn

            # 距離が妥当な範囲内（80単位以内）であれば採用
            if closest_dn and min_distance < 80:
                source_drawing = closest_dn
                if debug:
                    print(f"流用元図番をラベルから判別: {source_drawing} (距離: {min_distance:.2f})")

        # 図番を「DWG No.」ラベルに最も近い図面番号から確認
        if dwg_label_positions and not main_drawing:
            min_distance = float('inf')
            closest_dn = None

            for dn, dn_coords in drawing_numbers:
                for label_coords in dwg_label_positions:
                    distance = calculate_distance(dn_coords, label_coords)
                    if debug:
                        print(f"  {dn} から DWG No.ラベル までの距離: {distance:.2f}")

                    if distance < min_distance:
                        min_distance = distance
                        closest_dn = dn

            # 距離が妥当な範囲内（80単位以内）であれば採用
            if closest_dn and min_distance < 80:
                main_drawing = closest_dn
                if debug:
                    print(f"図番をDWG No.ラベルから判別: {main_drawing} (距離: {min_distance:.2f})")

    # 3. フォールバック: 座標ベースの判別
    if not main_drawing or not source_drawing:
        # 複数図面対応: 最も右側の図面番号群のみを対象とする
        # 最も右側のX座標を取得
        max_x = max([coords[0] for _, coords in drawing_numbers])
        x_tolerance = 100.0  # 同一図面とみなすX座標の許容範囲

        # 最も右側の範囲内にある図面番号のみをフィルタリング
        rightmost_numbers = [(dn, coords) for dn, coords in drawing_numbers
                            if coords[0] >= max_x - x_tolerance]

        if debug:
            print(f"座標ベース判別: 最大X={max_x:.2f}, 対象範囲={len(rightmost_numbers)}個")
            for dn, coords in rightmost_numbers:
                print(f"  {dn} at ({coords[0]:.2f}, {coords[1]:.2f})")

        # 最も右側の範囲内で座標ソート（右下が図番、それ以外が流用元図番）
        sorted_numbers = sorted(rightmost_numbers, key=lambda x: (x[1][0] + x[1][1]), reverse=True)

        if not main_drawing:
            # 最も右下にあるものを図番とする
            main_drawing = sorted_numbers[0][0]
            if debug:
                print(f"図番を座標から判別: {main_drawing} (右下)")

        if not source_drawing and len(sorted_numbers) > 1:
            # main_drawing以外で最も座標値が大きいものを流用元図番とする
            for dn, coords in sorted_numbers[1:]:
                if dn != main_drawing:
                    source_drawing = dn
                    if debug:
                        print(f"流用元図番を座標から判別: {source_drawing}")
                    break

    return {'main_drawing': main_drawing, 'source_drawing': source_drawing}


def extract_labels(dxf_file, filter_non_parts=False, sort_order="asc", debug=False,
                  selected_layers=None, validate_ref_designators=False,
                  extract_drawing_numbers_option=False, extract_title_option=False):
    """
    DXFファイルからテキストラベルを抽出する

    Args:
        dxf_file: DXFファイルパス
        filter_non_parts: 回路記号以外のラベルをフィルタリングするかどうか
        sort_order: ソート順 ("asc"=昇順, "desc"=降順, "none"=ソートなし)
        debug: デバッグ情報を表示するかどうか
        selected_layers: 処理対象とするレイヤー名のリスト。Noneの場合は全レイヤーを対象とする
        validate_ref_designators: 回路記号の妥当性をチェックするかどうか
        extract_drawing_numbers_option: 図面番号を抽出するかどうか
        extract_title_option: タイトルとサブタイトルを抽出するかどうか

    Returns:
        tuple: (ラベルリスト, 情報辞書)
    """
    # 処理情報を格納する辞書
    info = {
        "total_extracted": 0,
        "filtered_count": 0,
        "final_count": 0,
        "processed_layers": 0,
        "total_layers": 0,
        "filename": os.path.basename(dxf_file),
        "invalid_ref_designators": [],  # 妥当性チェック用
        "main_drawing_number": None,     # 図番
        "source_drawing_number": None,   # 流用元図番
        "all_drawing_numbers": [],       # 抽出されたすべての図面番号
        "title": None,                   # タイトル
        "subtitle": None                 # サブタイトル
    }

    try:
        # DXFファイルを読み込む
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()

        # 全レイヤー数を記録
        all_layers = [layer.dxf.name for layer in doc.layers]
        info["total_layers"] = len(all_layers)

        # 選択されたレイヤーの処理
        if selected_layers is None:
            # 選択されたレイヤーが指定されていない場合は全レイヤーを対象とする
            selected_layers = all_layers

        # 処理対象のレイヤー数を記録
        info["processed_layers"] = len(selected_layers)

        # すべてのテキストエンティティを抽出（選択されたレイヤーのみ）
        labels = []
        drawing_number_candidates = []  # 図面番号候補を座標付きで保存
        all_labels_with_coords = []  # 全ラベル（座標付き）- 図面番号判別用


        # 実際の抽出処理 - 全ての場所からエンティティを収集
        all_entities_to_process = []

        # 1. MODEL_SPACEからエンティティを収集
        msp = doc.modelspace()
        for e in msp:
            if e.dxftype() in ['TEXT', 'MTEXT']:
                all_entities_to_process.append(e)

        # 2. BLOCKSから直接は収集しない - INSERT経由でのみ処理する

        # 3. PAPER_SPACEからエンティティを収集
        try:
            for layout in doc.layouts:
                if layout.name != 'Model':  # Model space以外のレイアウト
                    for e in layout:
                        if e.dxftype() in ['TEXT', 'MTEXT']:
                            all_entities_to_process.append(e)
        except Exception as e:
            pass

        # 4. INSERT エンティティを処理してブロック参照を展開
        try:
            # ブロック定義内のテキストエンティティをキャッシュ
            block_text_cache = {}
            for block in doc.blocks:
                block_texts = []
                for entity in block:
                    if entity.dxftype() in ['TEXT', 'MTEXT']:
                        block_texts.append(entity)
                if block_texts:
                    block_text_cache[block.name] = block_texts

            # INSERT エンティティを処理
            for e in msp:
                if e.dxftype() == 'INSERT':
                    # INSERT エンティティのブロック名を取得
                    block_name = e.dxf.name

                    # そのブロック内のテキストエンティティを取得
                    if block_name in block_text_cache:
                        for text_entity in block_text_cache[block_name]:
                            # INSERT エンティティのレイヤーをチェック
                            if e.dxf.layer in selected_layers:
                                all_entities_to_process.append(text_entity)

            # ペーパースペースの INSERT エンティティも処理
            for layout in doc.layouts:
                if layout.name != 'Model':
                    for e in layout:
                        if e.dxftype() == 'INSERT':
                            block_name = e.dxf.name
                            if block_name in block_text_cache:
                                for text_entity in block_text_cache[block_name]:
                                    if e.dxf.layer in selected_layers:
                                        all_entities_to_process.append(text_entity)

        except Exception as e:
            pass  # INSERT処理エラーは無視して続行

        # 重複を除去（ここではINSERT経由の重複も許可する）
        processed_entity_ids = set()
        unique_entities = []
        for e in all_entities_to_process:
            # INSERT経由の同じブロック内エンティティは重複として扱わない
            unique_entities.append(e)


        # 実際の抽出処理
        for e in unique_entities:
            # エンティティのレイヤーが選択されたレイヤーに含まれているか確認
            if e.dxf.layer in selected_layers:
                # テキストと座標を抽出
                raw_text, clean_text, coordinates = extract_text_from_entity(e, debug)

                if clean_text:  # クリーンテキストがある場合のみ処理
                    # 座標付きラベル情報の保存（図面番号抽出またはタイトル抽出が有効な場合）
                    if extract_drawing_numbers_option or extract_title_option:
                        # 全ラベルを座標付きで保存
                        all_labels_with_coords.append((clean_text, coordinates))

                    # 図面番号抽出オプションが有効な場合の処理
                    if extract_drawing_numbers_option:
                        # クリーンテキストから図面番号を抽出
                        drawing_numbers = extract_drawing_numbers(clean_text, debug)
                        for dn in drawing_numbers:
                            drawing_number_candidates.append((dn, coordinates))

                    # 通常のラベルとして追加（クリーンテキストを使用）
                    labels.append(clean_text)

        # 総抽出数を記録
        info["total_extracted"] = len(labels)

        # 図面番号の判別（改善版ロジックを使用）
        if extract_drawing_numbers_option and drawing_number_candidates:
            drawing_info = determine_drawing_number_types(
                drawing_number_candidates,
                all_labels=all_labels_with_coords,
                filename=dxf_file,
                debug=debug
            )
            info["main_drawing_number"] = drawing_info['main_drawing']
            info["source_drawing_number"] = drawing_info['source_drawing']
            info["all_drawing_numbers"] = [dn[0] for dn in drawing_number_candidates]

        # タイトルとサブタイトルの抽出
        if extract_title_option and all_labels_with_coords:
            title_info = extract_title_and_subtitle(
                all_labels_with_coords,
                drawing_numbers=drawing_number_candidates if extract_drawing_numbers_option else None,
                debug=debug
            )
            info["title"] = title_info['title']
            info["subtitle"] = title_info['subtitle']



        # 回路記号処理（フィルタリング、妥当性チェック）
        symbol_result = process_circuit_symbol_labels(
            labels,
            filter_non_parts=filter_non_parts,
            validate_ref_designators=validate_ref_designators,
            debug=debug
        )

        # 処理結果を取得
        processed_labels = symbol_result['labels']
        info["filtered_count"] = symbol_result['filtered_count']
        info["invalid_ref_designators"] = symbol_result['invalid_ref_designators']

        # ソート
        if sort_order == "asc":
            processed_labels.sort()
        elif sort_order == "desc":
            processed_labels.sort(reverse=True)
        final_labels = processed_labels

        # 最終的なラベル数を記録
        info["final_count"] = len(final_labels)

        return final_labels, info

    except Exception as e:
        print(f"エラー: {str(e)}")
        info["error"] = str(e)
        return [], info


def process_multiple_dxf_files(dxf_files, filter_non_parts=False, sort_order="asc", debug=False,
                              selected_layers=None, validate_ref_designators=False,
                              extract_drawing_numbers_option=False, extract_title_option=False):
    """
    複数のDXFファイルからラベルを抽出する

    Args:
        dxf_files: DXFファイルパスのリスト
        filter_non_parts: 回路記号以外のラベルをフィルタリングするかどうか
        sort_order: ソート順 ("asc"=昇順, "desc"=降順, "none"=ソートなし)
        debug: デバッグ情報を表示するかどうか
        selected_layers: 処理対象とするレイヤー名のリスト。Noneの場合は全レイヤーを対象とする
        validate_ref_designators: 回路記号の妥当性をチェックするかどうか
        extract_drawing_numbers_option: 図面番号を抽出するかどうか
        extract_title_option: タイトルとサブタイトルを抽出するかどうか

    Returns:
        dict: ファイルパスをキー、(ラベルリスト, 情報辞書)をバリューとする辞書
    """
    results = {}

    for dxf_file in dxf_files:
        # ディレクトリの場合は、中のDXFファイルを処理
        if os.path.isdir(dxf_file):
            for root, _, files in os.walk(dxf_file):
                for file in files:
                    if file.lower().endswith('.dxf'):
                        file_path = os.path.join(root, file)
                        labels, info = extract_labels(
                            file_path, filter_non_parts, sort_order, debug,
                            selected_layers, validate_ref_designators,
                            extract_drawing_numbers_option, extract_title_option
                        )
                        results[file_path] = (labels, info)
        # 単一のDXFファイルの場合
        elif os.path.isfile(dxf_file) and dxf_file.lower().endswith('.dxf'):
            labels, info = extract_labels(
                dxf_file, filter_non_parts, sort_order, debug,
                selected_layers, validate_ref_designators,
                extract_drawing_numbers_option, extract_title_option
            )
            results[dxf_file] = (labels, info)

    return results
