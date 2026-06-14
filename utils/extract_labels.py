import ezdxf
import re
import os
import sys
import gc
from typing import List, Tuple, Dict, Optional

# 抽出設定の読み込み（環境適応型）
# config.py が存在するプロジェクト（DXF-diff-manager 等）ではそこから読み込む
# 存在しないプロジェクトではフォールバックの内部定義を使用
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from config import extraction_config
except ImportError:
    class ExtractionConfig:
        """DXF抽出関連の設定（デフォルト値）"""
        # 両フォーマット対応: XX0000-000-00X（長）、XX0000-000X（短）
        DRAWING_NUMBER_PATTERN = r'[A-Z]{2}\d{4}-\d{3}(?:-\d{2})?[A-Z]'
        SOURCE_LABEL_PROXIMITY = 80        # 流用元図番ラベルからの検出距離
        DWG_NO_LABEL_PROXIMITY = 80        # DWG No. ラベルからの検出距離
        TITLE_PROXIMITY_X = 80             # TITLE ラベルからの横方向検出距離
        RIGHTMOST_DRAWING_TOLERANCE = 100.0  # 右端図面判定の許容範囲

    extraction_config = ExtractionConfig()

from .common_utils import process_circuit_symbol_labels, filter_non_circuit_symbols


def get_layers_from_dxf(dxf_file):
    """DXFファイルからレイヤー一覧を取得する"""
    try:
        doc = ezdxf.readfile(dxf_file)
        layer_names = [layer.dxf.name for layer in doc.layers]
        return sorted(layer_names)
    except Exception as e:
        print(f"レイヤー一覧の取得中にエラーが発生しました: {str(e)}")
        return []


def clean_mtext_format_codes(text: str) -> str:
    r"""MTEXTのフォーマットコードを除去してテキスト内容を返す"""
    if not text:
        return ""

    # 日本語環境の円マーク（¥）をバックスラッシュに正規化
    cleaned = text.replace('¥', '\\')

    # フォーマット制御コードを除去（\P は段落区切りとして後処理）
    cleaned = re.sub(r'\\f[^;]*;', '', cleaned)
    cleaned = re.sub(r'\\H[^;]*;', '', cleaned)
    cleaned = re.sub(r'\\W[^;]*;', '', cleaned)
    cleaned = re.sub(r'\\C[^;]*;', '', cleaned)
    cleaned = re.sub(r'\\A[^;]*;', '', cleaned)
    cleaned = re.sub(r'\\T[^;]*;', '', cleaned)
    cleaned = re.sub(r'\\(?!P)[^\\;]*;', '', cleaned)

    cleaned = cleaned.replace('\\~', ' ')
    cleaned = cleaned.replace('\\\\', '\\')
    cleaned = cleaned.replace('\\{', '{')
    cleaned = cleaned.replace('\\}', '}')

    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = cleaned.replace('\\P', ' ')  # 段落区切り → スペース
    return re.sub(r'\s+', ' ', cleaned).strip()


def extract_text_from_entity(entity) -> Tuple[str, str, Tuple[float, float]]:
    """TEXT / MTEXT エンティティからテキストと座標を抽出する"""
    try:
        x, y = 0.0, 0.0

        if entity.dxftype() == 'MTEXT':
            if hasattr(entity.dxf, 'insert'):
                x, y = entity.dxf.insert[0], entity.dxf.insert[1]
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'x') and hasattr(entity.dxf, 'y'):
                x, y = entity.dxf.x, entity.dxf.y
            else:
                try:
                    x = getattr(entity.dxf, 'x', 0.0)
                    y = getattr(entity.dxf, 'y', 0.0)
                except Exception:
                    x, y = 0.0, 0.0
        elif entity.dxftype() == 'TEXT':
            if hasattr(entity.dxf, 'insert'):
                x, y = entity.dxf.insert[0], entity.dxf.insert[1]
            elif hasattr(entity.dxf, 'location'):
                x, y = entity.dxf.location[0], entity.dxf.location[1]

        raw_text = ""

        if entity.dxftype() == 'TEXT':
            if hasattr(entity.dxf, 'text'):
                raw_text = entity.dxf.text
        elif entity.dxftype() == 'MTEXT':
            if hasattr(entity.dxf, 'text'):
                raw_text = entity.dxf.text
            if not raw_text and hasattr(entity, 'text'):
                try:
                    raw_text = entity.text
                except Exception:
                    pass
            if not raw_text and hasattr(entity, 'plain_text'):
                try:
                    raw_text = entity.plain_text()
                except Exception:
                    pass

        if raw_text:
            clean_text = clean_mtext_format_codes(raw_text) if entity.dxftype() == 'MTEXT' else raw_text.strip()
        else:
            clean_text = ""

        return raw_text, clean_text, (x, y)

    except Exception:
        return "", "", (0.0, 0.0)


def extract_drawing_numbers(text: str) -> List[str]:
    """テキストから図面番号フォーマットに一致する文字列を抽出する"""
    drawing_numbers = []
    for match in re.findall(extraction_config.DRAWING_NUMBER_PATTERN, text, re.IGNORECASE):
        if match.upper() not in [dn.upper() for dn in drawing_numbers]:
            drawing_numbers.append(match.upper())
    return drawing_numbers


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """2点間のユークリッド距離を計算する"""
    import math
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    return math.sqrt(dx * dx + dy * dy)


def is_single_uppercase_letter(text: str) -> bool:
    """英大文字1文字かどうかを判定（半角・全角両対応）"""
    if len(text) != 1:
        return False
    if 'A' <= text <= 'Z':
        return True
    if 'Ａ' <= text <= 'Ｚ':  # 全角英大文字
        return True
    return False


def extract_title_and_subtitle(
    all_labels: List[Tuple[str, Tuple[float, float]]],
    drawing_numbers: Optional[List[Tuple[str, Tuple[float, float]]]],
) -> Dict[str, Optional[str]]:
    """テキストラベルの位置関係からタイトルとサブタイトルを抽出する"""
    if not all_labels:
        return {'title': None, 'subtitle': None}

    title_text = None
    subtitle_text = None

    # TITLE と REVISION の位置を特定（複数ある場合は最も右側を採用）
    title_label_positions = []
    revision_label_positions = []

    for label, coords in all_labels:
        label_upper = label.upper().strip()
        if label_upper == 'TITLE':
            title_label_positions.append(coords)
        elif label_upper == 'REVISION':
            revision_label_positions.append(coords)

    if not title_label_positions:
        return {'title': None, 'subtitle': None}

    title_label_pos = max(title_label_positions, key=lambda pos: pos[0])
    revision_label_pos = max(revision_label_positions, key=lambda pos: pos[0]) if revision_label_positions else None

    # タイトル候補を収集（TITLE の右側かつ REVISION より下）
    title_proximity_x = extraction_config.TITLE_PROXIMITY_X
    title_candidates = []

    for label, coords in all_labels:
        label_upper = label.upper().strip()
        if label_upper in ['TITLE', 'REVISION']:
            continue
        if drawing_numbers and any(dn == label for dn, *_ in drawing_numbers):
            continue

        x_diff = coords[0] - title_label_pos[0]
        if 10 < x_diff < title_proximity_x:
            if revision_label_pos:
                if coords[1] < revision_label_pos[1]:
                    title_candidates.append((label, coords))
            else:
                title_candidates.append((label, coords))

    if not title_candidates:
        return {'title': None, 'subtitle': None}

    # 座標が近く同じラベルの重複を除去
    coord_tolerance = 1.0
    deduplicated = []
    for label, coords in title_candidates:
        is_dup = False
        for ex_label, ex_coords in deduplicated:
            if (label == ex_label
                    and abs(coords[0] - ex_coords[0]) <= coord_tolerance
                    and abs(coords[1] - ex_coords[1]) <= coord_tolerance):
                is_dup = True
                break
        if not is_dup:
            deduplicated.append((label, coords))

    title_candidates = deduplicated
    if not title_candidates:
        return {'title': None, 'subtitle': None}

    # Y座標でグルーピング（同じ行の単語を結合）
    y_tolerance = 5.0
    grouped_candidates = []
    sorted_candidates = sorted(title_candidates, key=lambda x: (-x[1][1], x[1][0]))

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

    if not grouped_candidates:
        return {'title': None, 'subtitle': None}

    # Y座標が最も高いグループ群の中で、X座標が最小のものをタイトルとする
    groups_with_min_x = [
        (group, min(c[0] for _, c in group), sum(c[1] for _, c in group) / len(group))
        for group in grouped_candidates
    ]

    max_y = max(avg_y for _, _, avg_y in groups_with_min_x)
    y_threshold = 10.0
    top_groups = [(g, mx, ay) for g, mx, ay in groups_with_min_x if ay >= max_y - y_threshold]
    title_group = min(top_groups, key=lambda x: x[1])[0]

    title_group_sorted = sorted(title_group, key=lambda x: x[1][0])
    title_text = ' '.join(label for label, _ in title_group_sorted)
    title_y_coord = title_group_sorted[0][1][1]

    # サブタイトルを探す（タイトルの直下かつ同じX座標範囲）
    title_min_x = min(c[0] for _, c in title_group)
    title_max_x = max(c[0] for _, c in title_group)
    x_tolerance = extraction_config.RIGHTMOST_DRAWING_TOLERANCE

    subtitle_candidates = [
        (g, mx, ay) for g, mx, ay in groups_with_min_x
        if ay < title_y_coord
        and mx <= title_max_x + x_tolerance
        and max(c[0] for _, c in g) >= title_min_x - x_tolerance
    ]

    if subtitle_candidates:
        subtitle_group = max(subtitle_candidates, key=lambda x: (x[2], -x[1]))[0]
        subtitle_group_sorted = sorted(subtitle_group, key=lambda x: x[1][0])
        subtitle_labels = [label for label, _ in subtitle_group_sorted]

        if len(subtitle_labels) > 1 and is_single_uppercase_letter(subtitle_labels[-1]):
            subtitle_labels = subtitle_labels[:-1]

        subtitle_text = ' '.join(subtitle_labels)

    return {'title': title_text, 'subtitle': subtitle_text}


def determine_drawing_number_types(
    drawing_numbers: List[Tuple],
    all_labels: Optional[List[Tuple[str, Tuple[float, float]]]] = None,
    filename: Optional[str] = None,
) -> Dict[str, str]:
    """ラベルと座標に基づいて図番と流用元図番を判別する。

    各候補は `(図番, 座標)` または `(図番, 座標, グループキー)`。グループキーは
    所属タイトルブロック（INSERT）の識別子で、旧・現行のタイトルブロックが同一
    座標に重なっているケースで「図番と流用元図番が同じブロックに属する」ことを
    判定するために使う。図番がファイル名等で確定したら、**同じグループ内**で
    流用元図番を判定し、別ブロック（旧版）の図番を誤って拾わないようにする。
    """
    if len(drawing_numbers) == 0:
        return {'main_drawing': None, 'source_drawing': None}

    # 候補を (図番, 座標, グループ) に正規化（グループ未指定は None）
    norm = [(item[0], item[1], item[2] if len(item) >= 3 else None) for item in drawing_numbers]

    if len(norm) == 1:
        return {'main_drawing': norm[0][0], 'source_drawing': None}

    main_drawing = None
    main_group = None
    source_drawing = None

    # 1. ファイル名と一致する図面番号を図番とする
    if filename:
        from pathlib import Path
        file_stem = Path(filename).stem
        for dn, coords, group in norm:
            if dn == file_stem or dn in file_stem or file_stem in dn:
                main_drawing = dn
                main_group = group
                break

    # 2. ラベルベースの判別
    if all_labels:
        source_label_positions = [
            coords for label, coords in all_labels
            if '流用元図番' in label or '流用元' in label
        ]
        dwg_label_positions = [
            coords for label, coords in all_labels
            if ('DWG' in label.upper().replace('\n', '').replace('\r', '').replace(' ', '')
                and 'NO' in label.upper().replace('\n', '').replace('\r', '').replace(' ', ''))
        ]

        # 流用元図番を「流用元図番」ラベルに最も近い図面番号から判別する。
        # 図番のグループが分かっている場合は、同一グループ内の候補に限定して
        # 判定し、重なった別ブロック（旧版）の図番を拾わないようにする。
        if source_label_positions:
            source_pool = norm
            if main_group is not None:
                same_group = [c for c in norm if c[2] == main_group and c[0] != main_drawing]
                if same_group:
                    source_pool = same_group

            min_distance = float('inf')
            closest_dn = None
            for dn, dn_coords, group in source_pool:
                if main_drawing and dn == main_drawing:
                    continue
                for label_coords in source_label_positions:
                    distance = calculate_distance(dn_coords, label_coords)
                    if distance < min_distance:
                        min_distance = distance
                        closest_dn = dn
            if closest_dn and min_distance < extraction_config.SOURCE_LABEL_PROXIMITY:
                if closest_dn != main_drawing:
                    source_drawing = closest_dn

        # 図番を「DWG No.」ラベルに最も近い図面番号から確認
        if dwg_label_positions and not main_drawing:
            min_distance = float('inf')
            closest_dn = None
            closest_group = None
            for dn, dn_coords, group in norm:
                for label_coords in dwg_label_positions:
                    distance = calculate_distance(dn_coords, label_coords)
                    if distance < min_distance:
                        min_distance = distance
                        closest_dn = dn
                        closest_group = group
            if closest_dn and min_distance < extraction_config.DWG_NO_LABEL_PROXIMITY:
                main_drawing = closest_dn
                main_group = closest_group

    # 3. フォールバック: 座標ベースの判別（最も右側の図面を対象）
    if not main_drawing or not source_drawing:
        max_x = max(coords[0] for _, coords, _ in norm)
        x_tolerance = extraction_config.RIGHTMOST_DRAWING_TOLERANCE
        rightmost_numbers = [
            (dn, coords, group) for dn, coords, group in norm
            if coords[0] >= max_x - x_tolerance
        ]
        sorted_numbers = sorted(rightmost_numbers, key=lambda x: (x[1][0] + x[1][1]), reverse=True)

        if not main_drawing:
            main_drawing = sorted_numbers[0][0]
            main_group = sorted_numbers[0][2]

        if not source_drawing and len(sorted_numbers) > 1:
            # 図番グループが分かっている場合は同一グループを優先する
            ordered = sorted_numbers[1:]
            if main_group is not None:
                ordered = [c for c in ordered if c[2] == main_group] + \
                          [c for c in ordered if c[2] != main_group]
            for dn, _coords, _group in ordered:
                if dn != main_drawing:
                    source_drawing = dn
                    break

    # 最終検証: 流用元図番が図番と同じ場合は None にする
    if source_drawing and source_drawing == main_drawing:
        source_drawing = None

    return {'main_drawing': main_drawing, 'source_drawing': source_drawing}


def extract_labels(dxf_file, filter_non_parts=False, sort_order="asc", debug=False,
                   selected_layers=None, validate_ref_designators=False,
                   extract_drawing_numbers_option=False, extract_title_option=False,
                   include_coordinates=False, original_filename=None):
    """DXFファイルからテキストラベルを抽出する"""
    info = {
        "total_extracted": 0,
        "filtered_count": 0,
        "final_count": 0,
        "processed_layers": 0,
        "total_layers": 0,
        "filename": os.path.basename(dxf_file),
        "invalid_ref_designators": [],
        "main_drawing_number": None,
        "source_drawing_number": None,
        "all_drawing_numbers": [],
        "title": None,
        "subtitle": None,
    }

    try:
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()

        all_layers = [layer.dxf.name for layer in doc.layers]
        info["total_layers"] = len(all_layers)

        if selected_layers is None:
            selected_layers = all_layers
        info["processed_layers"] = len(selected_layers)

        labels = []
        labels_with_coordinates = []
        drawing_number_candidates = []
        all_labels_with_coords = []

        # エンティティ収集
        # 各要素は (entity, group_key)。group_key は所属タイトルブロック（INSERT）の
        # 識別子。INSERT 由来は親 INSERT の handle、直接配置は自身の handle を使う。
        # 旧・現行のタイトルブロックが同一座標に重なっているケースで、図番と流用元
        # 図番が同じブロックに属することを判定するために用いる。
        all_entities_to_process = []

        def _entity_handle(entity):
            return getattr(entity.dxf, 'handle', None)

        # MODEL_SPACE
        for e in msp:
            if e.dxftype() in ['TEXT', 'MTEXT']:
                all_entities_to_process.append((e, _entity_handle(e)))

        # PAPER_SPACE（Model 以外のレイアウト）
        try:
            for layout in doc.layouts:
                if layout.name != 'Model':
                    for e in layout:
                        if e.dxftype() in ['TEXT', 'MTEXT']:
                            all_entities_to_process.append((e, _entity_handle(e)))
        except Exception:
            pass

        # INSERT エンティティを virtual_entities() で展開（座標変換を含む）
        # 展開後の仮想エンティティには親 INSERT の handle をグループキーとして付与する。
        try:
            for e in msp:
                if e.dxftype() == 'INSERT' and e.dxf.layer in selected_layers:
                    insert_group = _entity_handle(e)
                    try:
                        for virtual_entity in e.virtual_entities():
                            if virtual_entity.dxftype() in ['TEXT', 'MTEXT']:
                                all_entities_to_process.append((virtual_entity, insert_group))
                    except Exception:
                        pass

            for layout in doc.layouts:
                if layout.name != 'Model':
                    for e in layout:
                        if e.dxftype() == 'INSERT' and e.dxf.layer in selected_layers:
                            insert_group = _entity_handle(e)
                            try:
                                for virtual_entity in e.virtual_entities():
                                    if virtual_entity.dxftype() in ['TEXT', 'MTEXT']:
                                        all_entities_to_process.append((virtual_entity, insert_group))
                            except Exception:
                                pass
        except Exception:
            pass

        # 重複除去（同一種・同一レイヤー・同一座標）
        seen_entities = set()
        unique_entities = []
        for e, group_key in all_entities_to_process:
            try:
                entity_key = (
                    e.dxftype(),
                    e.dxf.layer if hasattr(e.dxf, 'layer') else '',
                    getattr(e.dxf, 'insert', (0, 0)) if hasattr(e.dxf, 'insert') else (0, 0),
                )
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    unique_entities.append((e, group_key))
            except Exception:
                unique_entities.append((e, group_key))

        del all_entities_to_process
        del seen_entities

        # テキスト抽出
        for e, group_key in unique_entities:
            if e.dxf.layer in selected_layers:
                raw_text, clean_text, coordinates = extract_text_from_entity(e)

                if clean_text:
                    if extract_drawing_numbers_option or extract_title_option:
                        all_labels_with_coords.append((clean_text, coordinates))

                    if extract_drawing_numbers_option:
                        for dn in extract_drawing_numbers(clean_text):
                            drawing_number_candidates.append((dn, coordinates, group_key))

                    labels.append(clean_text)
                    labels_with_coordinates.append((clean_text, coordinates[0], coordinates[1]))

        info["total_extracted"] = len(labels)

        # 図面番号の判別
        if extract_drawing_numbers_option and drawing_number_candidates:
            filename_for_matching = original_filename if original_filename else dxf_file
            drawing_info = determine_drawing_number_types(
                drawing_number_candidates,
                all_labels=all_labels_with_coords,
                filename=filename_for_matching,
            )
            info["main_drawing_number"] = drawing_info['main_drawing']
            info["source_drawing_number"] = drawing_info['source_drawing']
            info["all_drawing_numbers"] = [dn[0] for dn in drawing_number_candidates]

        # タイトル・サブタイトルの抽出
        if extract_title_option and all_labels_with_coords:
            title_info = extract_title_and_subtitle(
                all_labels_with_coords,
                drawing_numbers=drawing_number_candidates if extract_drawing_numbers_option else None,
            )
            info["title"] = title_info['title']
            info["subtitle"] = title_info['subtitle']

        # 機器符号フィルタリング・妥当性チェック
        symbol_result = process_circuit_symbol_labels(
            labels,
            filter_non_parts=filter_non_parts,
            validate_ref_designators=validate_ref_designators,
        )
        processed_labels = symbol_result['labels']
        info["filtered_count"] = symbol_result['filtered_count']
        info["invalid_ref_designators"] = symbol_result['invalid_ref_designators']

        # ソートと返却形式の選択
        if include_coordinates:
            filtered_set = set(processed_labels)
            processed_label_entries = [
                entry for entry in labels_with_coordinates if entry[0] in filtered_set
            ]
            if sort_order == "asc":
                processed_label_entries.sort(key=lambda x: (x[0], x[1], x[2]))
            elif sort_order == "desc":
                processed_label_entries.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            final_labels = processed_label_entries
        else:
            if sort_order == "asc":
                processed_labels.sort()
            elif sort_order == "desc":
                processed_labels.sort(reverse=True)
            final_labels = processed_labels

        info["final_count"] = len(final_labels)

        del doc
        del msp
        gc.collect()

        return final_labels, info

    except Exception as e:
        print(f"エラー: {str(e)}")
        info["error"] = str(e)
        gc.collect()
        return [], info


def process_multiple_dxf_files(dxf_files, filter_non_parts=False, sort_order="asc", debug=False,
                                selected_layers=None, validate_ref_designators=False,
                                extract_drawing_numbers_option=False, extract_title_option=False,
                                original_filenames=None):
    """複数のDXFファイルからラベルを抽出する"""
    results = {}

    for i, dxf_file in enumerate(dxf_files):
        original_filename = original_filenames[i] if original_filenames and i < len(original_filenames) else None

        if os.path.isdir(dxf_file):
            for root, _, files in os.walk(dxf_file):
                for file in files:
                    if file.lower().endswith('.dxf'):
                        file_path = os.path.join(root, file)
                        labels, info = extract_labels(
                            file_path, filter_non_parts, sort_order, debug,
                            selected_layers, validate_ref_designators,
                            extract_drawing_numbers_option, extract_title_option,
                            original_filename=file,
                        )
                        results[file_path] = (labels, info)
        elif os.path.isfile(dxf_file) and dxf_file.lower().endswith('.dxf'):
            labels, info = extract_labels(
                dxf_file, filter_non_parts, sort_order, debug,
                selected_layers, validate_ref_designators,
                extract_drawing_numbers_option, extract_title_option,
                original_filename=original_filename,
            )
            results[dxf_file] = (labels, info)

    return results


# ===========================================================================
# 矩形領域抽出（オプション機能）
# ---------------------------------------------------------------------------
# 電気回路 DXF 内の閉領域（直交ポリゴン、四角形とは限らない）を検出し、領域内
# ラベルに領域名を付与する。仕様（検証済み）:
#   - 図面枠      : lineweight=100 の線分で囲まれた枠。枠内のみ処理対象。
#   - 領域境界線  : lineweight=25 かつ color=2(ACI黄)。レイヤーは不定で使えない。
#   - 接点マージン: ±2。同一X(縦)/同一Y(横)の線分は隙間を無視して1本に結合
#                   （境界線上に部品があり線分が途切れるため）。
#   - 閉領域      : 結合線の交点で平面グラフ化 → 面探索 → 面積>=枠面積×20%。
#   - 領域名      : 境界近傍の英字3文字以上ラベル（複数候補→UIで確定）。
#   - ラベル所属  : 点-多角形内包判定（1ラベルが複数領域に所属可）。
# ===========================================================================

DEFAULT_REGION_CONFIG = {
    'frame_lineweight': 100,    # 図面枠の線の太さ
    'region_lineweight': 25,    # 領域境界線の太さ
    'region_color': 2,          # 領域境界線の色(ACI)
    'snap': 2.0,                # 軸平行判定・レベルクラスタの許容誤差
    'face_snap': 0.1,           # 矩形を構成する線分同士の接続点(交点)の座標マージン
                                # ※小さく（違う矩形を取り込むリスクを抑える）
    'merge_level_tol': 0.5,     # 共線セグメント結合時のレベル座標(縦=x/横=y)一致許容
                                # ※小さくする（別レベルの線=別矩形を結合しない）
    # ギャップ（隙間）の橋渡し方針：部品ラベルは縦線分だけを途切れさせるため、
    # 縦線分のギャップのみ橋渡しし、横線分のギャップは橋渡ししない（別矩形の取り込み防止）。
    'bridge_vertical_gaps': True,    # 縦線分(同一X)のギャップを橋渡しする
    'bridge_horizontal_gaps': False, # 横線分(同一Y)のギャップは橋渡ししない
    'corner_tol': 0.5,               # 縦線端点と横線端点が一致（コーナー）とみなす許容。
                                     # ギャップ両端にコーナー相手がいれば橋渡ししない。
    'area_ratio': 0.20,         # 単独の領域の最小面積（枠面積比）
    'group_area_ratio': 0.10,   # 同名複数ピースを合算した場合の最小合計面積（枠面積比）
    'min_face_ratio': 0.005,    # 個々の閉領域として残す最小面積（枠面積比、ノイズ除去）
    'name_max_dist': 10.0,      # 名称ラベルの境界からの最大距離
    'name_min_dist': 1.0,       # 名称ラベルの境界からの最小距離（線分上=0 を除外）
    'name_min_letters': 3,      # 名称候補に必要な英字数
    'name_exclude_terms': ('NOTE', '☆'),  # 候補から除外する語（含む場合）
    'name_exclude_lowercase': True,  # 英小文字を含むラベルを名称候補から除外
    'exclude_titleblock': True, # 図番枠（タイトルブロック）を領域から除外
    'exclude_circuit_symbols': True,   # 機器符号(候補)を名称候補から除外
    'circuit_symbol_keep_terms': ('RACK',),  # この語を含むラベルは機器符号扱いしない（例 RACK1）
    'exclude_connection_point_regions': True,  # 境界に接続点(円)を持つ領域(配線ループ)を除外
    'connection_point_threshold': 1,    # 境界上の接続点がこの数(個数)以上なら除外
    'connection_point_margin': 0.1,    # 接続点が境界線上とみなす座標距離マージン
}


def _collect_region_geometry(msp, cfg):
    """msp を1回走査し、INSERT も展開して、図面枠線・領域境界線・テキスト・
    接続点（CIRCLE を含むブロックの INSERT 位置）を収集する。"""
    frame_lines = []
    region_lines = []
    label_entities = []
    connection_points = []
    flw = cfg['frame_lineweight']
    rlw = cfg['region_lineweight']
    rcol = cfg['region_color']

    doc = getattr(msp, 'doc', None)
    _circle_block = {}

    def block_has_circle(name):
        if name not in _circle_block:
            has = False
            try:
                blk = doc.blocks.get(name) if doc else None
                if blk is not None:
                    has = any(x.dxftype() == 'CIRCLE' for x in blk)
            except Exception:
                has = False
            _circle_block[name] = has
        return _circle_block[name]

    def handle_line(e):
        lw = getattr(e.dxf, 'lineweight', None)
        if lw == flw:
            frame_lines.append((e.dxf.start, e.dxf.end))
        elif lw == rlw and getattr(e.dxf, 'color', None) == rcol:
            region_lines.append((e.dxf.start, e.dxf.end))

    for e in msp:
        t = e.dxftype()
        if t == 'LINE':
            handle_line(e)
        elif t in ('TEXT', 'MTEXT'):
            label_entities.append(e)
        elif t == 'INSERT':
            if block_has_circle(e.dxf.name):
                ins = e.dxf.insert
                connection_points.append((ins[0], ins[1]))
            try:
                for v in e.virtual_entities():
                    vt = v.dxftype()
                    if vt == 'LINE':
                        handle_line(v)
                    elif vt in ('TEXT', 'MTEXT'):
                        label_entities.append(v)
            except Exception:
                pass
    return frame_lines, region_lines, label_entities, connection_points


def _count_connection_points_on_boundary(polygon, points, margin):
    """ポリゴン境界から margin 以内にある接続点の数を返す（bbox で事前絞り込み）。"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = min(xs) - margin - 1, max(xs) + margin + 1
    y0, y1 = min(ys) - margin - 1, max(ys) + margin + 1
    n = 0
    for (px, py) in points:
        if x0 <= px <= x1 and y0 <= py <= y1:
            if _dist_point_to_polygon((px, py), polygon) <= margin:
                n += 1
    return n


def _split_axis_aligned(pairs, eps):
    """線分(start,end)を水平 H[(y,x0,x1)] と垂直 V[(x,y0,y1)] に分類する。"""
    H = []
    V = []
    for s, en in pairs:
        x1, y1, x2, y2 = s[0], s[1], en[0], en[1]
        if abs(y1 - y2) <= eps and abs(x1 - x2) > eps:
            H.append(((y1 + y2) / 2.0, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) <= eps and abs(y1 - y2) > eps:
            V.append(((x1 + x2) / 2.0, min(y1, y2), max(y1, y2)))
    return H, V


def _cluster_1d(vals, tol):
    vals = sorted(vals)
    out = []
    cur = [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            out.append(sum(cur) / len(cur))
            cur = [v]
    out.append(sum(cur) / len(cur))
    return out


def _gap_has_circle(level, a, b, circles, band):
    """縦線分(level=x)のギャップ [a,b]（y方向）に接続点(円)が乗っているか判定する。"""
    if not circles:
        return False
    for (cx, cy) in circles:
        if abs(cx - level) <= band and a - band <= cy <= b + band:
            return True
    return False


def _has_corner_partner(level, y, h_endpoints, tol):
    """縦線端点 (level, y) に、横線分の端点が一致しているか（＝コーナー相手がいるか）。
    コーナー相手がいる端点は境界がそこで折れるので、ギャップ橋渡ししない。"""
    for (hx, hy) in (h_endpoints or ()):
        if abs(hx - level) <= tol and abs(hy - y) <= tol:
            return True
    return False


def _merge_collinear(items, level_tol, bridge=True, circles=None, circle_band=2.0,
                     h_endpoints=None, corner_tol=0.5):
    """同一レベル(±level_tol)の共線セグメントを結合する。

    bridge=True のとき隙間（ギャップ）も橋渡しして1本にする（部品で途切れた縦線分の
    復元用）。bridge=False のときは重なり/接触するセグメントのみ結合し、隙間は別スパン
    として残す（横線分。別矩形の取り込みを防ぐ）。

    縦線のギャップ橋渡しは、**ギャップ両端のどちらにも横線分の端点が一致しない**場合
    のみ行う（端点が一致する＝コーナーで境界が折れるステップなので橋渡ししない。これに
    より、別境界片や段差を誤って繋がない）。circles がギャップ上にある場合も橋渡ししない。
    """
    if not items:
        return []
    items = sorted(items, key=lambda t: t[0])
    groups = []
    cur = [items[0]]
    for it in items[1:]:
        if it[0] - cur[-1][0] <= level_tol:
            cur.append(it)
        else:
            groups.append(cur)
            cur = [it]
    groups.append(cur)

    out = []
    for g in groups:
        level = sum(t[0] for t in g) / len(g)
        spans = sorted((t[1], t[2]) for t in g)
        merged = [list(spans[0])]
        for lo, hi in spans[1:]:
            phi = merged[-1][1]
            if lo <= phi + 1e-6:  # 重なり/接触 → 結合
                merged[-1][1] = max(phi, hi)
            elif (bridge
                  and not _has_corner_partner(level, phi, h_endpoints, corner_tol)
                  and not _has_corner_partner(level, lo, h_endpoints, corner_tol)
                  and not _gap_has_circle(level, phi, lo, circles, circle_band)):
                merged[-1][1] = max(phi, hi)  # 橋渡し（両端コーナー無し・円無し）
            else:
                merged.append([lo, hi])       # 隙間 → 別スパンとして分離
        for lo, hi in merged:
            out.append((level, lo, hi))
    return out


def detect_drawing_frames(frame_lines, eps=2.0, min_side=400.0):
    """lineweight=100 の線分から図面枠（複数可）を検出する。
    枠の縦長辺が左右ペアで横並びになる前提。戻り値: [(xl,xr,y0,y1), ...]"""
    _, V = _split_axis_aligned(frame_lines, eps)
    tall = [v for v in V if (v[2] - v[1]) >= min_side]
    if len(tall) < 2:
        return []
    xedges = _cluster_1d([v[0] for v in tall], eps)
    ys = [v[1] for v in tall] + [v[2] for v in tall]
    y0, y1 = min(ys), max(ys)
    frames = []
    for i in range(0, len(xedges) - 1, 2):
        frames.append((xedges[i], xedges[i + 1], y0, y1))
    return frames


def _find_rectilinear_faces(Hm, Vm, eps):
    """結合済み水平線 Hm[(y,x0,x1)]・垂直線 Vm[(x,y0,y1)] から閉領域(面)を列挙する。

    接続は **線分の端点が相手の線分に乗っている箇所（角・T字）のみ** で作る。
    中ほど同士の交差（どちらの端点でもない交差）では接続しない。これにより、
    コネクタ横線が矩形右辺の途中を横切るだけの箇所で誤って繋がるのを防ぐ。
    """
    import math as _m

    # 座標を許容誤差クラスタリングで正規化する（round の境界で一致点が分裂するのを防ぐ）。
    # 手描きの微小ズレ（例 y=231.91 と 231.96）を同一ノードに寄せる。
    ctol = max(eps, 0.2)

    def _canon_map(values):
        sv = sorted(set(values))
        m = {}
        if not sv:
            return m
        cluster = [sv[0]]
        for v in sv[1:]:
            if v - cluster[-1] <= ctol:
                cluster.append(v)
            else:
                c = sum(cluster) / len(cluster)
                for u in cluster:
                    m[u] = c
                cluster = [v]
        c = sum(cluster) / len(cluster)
        for u in cluster:
            m[u] = c
        return m

    all_x = set()
    all_y = set()
    for (y, x0, x1) in Hm:
        all_y.add(y); all_x.add(x0); all_x.add(x1)
    for (x, y0, y1) in Vm:
        all_x.add(x); all_y.add(y0); all_y.add(y1)
    cx = _canon_map(all_x)
    cy = _canon_map(all_y)

    def cluster_key(x, y):
        return (round(cx[x], 3), round(cy[y], 3))

    v_endpoints = []
    for (x, y0, y1) in Vm:
        v_endpoints.append((x, y0))
        v_endpoints.append((x, y1))
    h_endpoints = []
    for (y, x0, x1) in Hm:
        h_endpoints.append((x0, y))
        h_endpoints.append((x1, y))

    node_xy = {}
    line_pts = {}
    # 横線上のノード = 自身の端点 ＋ そこに端点で接する縦線の位置
    for hi, (y, x0, x1) in enumerate(Hm):
        xs = [x0, x1]
        for (vx, vy) in v_endpoints:
            if x0 - eps <= vx <= x1 + eps and abs(vy - y) <= eps:
                xs.append(vx)
        for x in xs:
            k = cluster_key(x, y)
            node_xy[k] = (x, y)
            line_pts.setdefault(('H', hi), []).append((x, k))
    # 縦線上のノード = 自身の端点 ＋ そこに端点で接する横線の位置
    for vi, (x, y0, y1) in enumerate(Vm):
        ys = [y0, y1]
        for (hx, hy) in h_endpoints:
            if y0 - eps <= hy <= y1 + eps and abs(hx - x) <= eps:
                ys.append(hy)
        for yy in ys:
            k = cluster_key(x, yy)
            node_xy[k] = (x, yy)
            line_pts.setdefault(('V', vi), []).append((yy, k))
    adj = {}
    for pts in line_pts.values():
        pts = sorted(set(pts))
        for a in range(len(pts) - 1):
            ka, kb = pts[a][1], pts[a + 1][1]
            if ka != kb:
                adj.setdefault(ka, set()).add(kb)
                adj.setdefault(kb, set()).add(ka)
    if not adj:
        return []

    def ang(a, b):
        ax, ay = node_xy[a]
        bx, by = node_xy[b]
        return _m.atan2(by - ay, bx - ax)

    order = {n: sorted(nb, key=lambda mm: ang(n, mm)) for n, nb in adj.items()}
    visited = set()
    faces = []
    for u in list(adj.keys()):
        for v in adj[u]:
            if (u, v) in visited:
                continue
            face = []
            cu, cv = u, v
            ok = True
            while True:
                visited.add((cu, cv))
                face.append(node_xy[cu])
                nb = order[cv]
                w = nb[(nb.index(cu) - 1) % len(nb)]
                cu, cv = cv, w
                if (cu, cv) == (u, v):
                    break
                if len(face) > 200000:
                    ok = False
                    break
            if ok and len(face) >= 4:
                faces.append(face)
    return faces


def _polygon_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _polygon_corners(poly, tol=0.5):
    """ポリゴンの角（直角に折れる頂点）だけを抽出し、左下から順に並べて返す。

    面探索由来の共線中間点を除去し、最も左下（最小y→最小x）の角を先頭にする。
    """
    n = len(poly)
    out = []
    for i in range(n):
        p0 = poly[(i - 1) % n]
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) > tol:   # 折れ点（共線でない）→ 角
            out.append((round(p1[0], 2), round(p1[1], 2)))
    if not out:
        out = [(round(x, 2), round(y, 2)) for (x, y) in poly]
    start = min(range(len(out)), key=lambda i: (out[i][1], out[i][0]))
    return out[start:] + out[:start]


def _point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_to_polygon(pt, poly):
    import math as _m
    x, y = pt
    best = float('inf')
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / denom))
        px, py = x1 + t * dx, y1 + t * dy
        best = min(best, _m.hypot(x - px, y - py))
    return best


def _detect_regions(RH, RV, frame, frame_area, cfg, labels=None, circles=None):
    """1つの図面枠内で、面積>=枠面積×area_ratio の閉領域を検出する。"""
    xl, xr, y0, y1 = frame
    Hf = [h for h in RH if y0 - 5 <= h[0] <= y1 + 5 and h[2] >= xl - 5 and h[1] <= xr + 5]
    Vf = [v for v in RV if xl - 5 <= v[0] <= xr + 5 and v[2] >= y0 - 5 and v[1] <= y1 + 5]
    if not Hf or not Vf:
        return []
    # 共線セグメントの結合はレベル座標を厳密一致(merge_level_tol)で行い、別レベルの
    # 線（=別矩形）を誤って繋がない。ギャップ橋渡しは縦線分のみ（部品ラベルは縦線分を
    # 途切れさせる）。横線分のギャップは橋渡ししない。接続点(交点)判定は face_snap。
    # 縦線分のギャップが CIRCLE で繋がっている場合は橋渡ししない（配線ループ除外）。
    mtol = cfg.get('merge_level_tol', 0.5)
    fsnap = cfg.get('face_snap', 0.1)
    bridge_v = cfg.get('bridge_vertical_gaps', True)
    bridge_h = cfg.get('bridge_horizontal_gaps', False)
    cband = cfg.get('connection_point_margin', 2.0)
    ctol = cfg.get('corner_tol', 0.5)
    fcircles = [c for c in (circles or []) if xl - 5 <= c[0] <= xr + 5 and y0 - 5 <= c[1] <= y1 + 5]
    # 横線分の端点（縦ギャップのコーナー相手判定用）
    h_endpoints = []
    for (hy, hx0, hx1) in Hf:
        h_endpoints.append((hx0, hy))
        h_endpoints.append((hx1, hy))
    Hm = _merge_collinear(Hf, mtol, bridge=bridge_h)
    Vm = _merge_collinear(Vf, mtol, bridge=bridge_v, circles=fcircles, circle_band=cband,
                          h_endpoints=h_endpoints, corner_tol=ctol)
    # 端点接続ベースの面探索（中ほど交差では繋がない）ため、部品矩形の縦線は領域辺の
    # 途中を横切るだけで接続せず、回り込みは発生しない。
    faces = _find_rectilinear_faces(Hm, Vm, fsnap)
    thr = frame_area * cfg.get('min_face_ratio', 0.005)
    regions = []
    seen = set()
    for f in sorted(faces, key=_polygon_area, reverse=True):
        a = _polygon_area(f)
        if a < thr:
            continue
        xs = [p[0] for p in f]
        ys = [p[1] for p in f]
        bb = (round(min(xs)), round(max(xs)), round(min(ys)), round(max(ys)))
        if bb in seen:
            continue
        seen.add(bb)
        regions.append({'polygon': f, 'area': a})
    return regions


def _count_letters(s):
    return sum(1 for ch in s if ch.isascii() and ch.isalpha())


def _bottom_edges(polygon, level_tol=2.0):
    """ポリゴンの下端（最小y）にある横エッジ群 [(x0,x1,y), ...] を返す。"""
    min_y = min(p[1] for p in polygon)
    segs = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if abs(y1 - y2) < 0.5 and abs(y1 - min_y) <= level_tol:
            segs.append((min(x1, x2), max(x1, x2), y1))
    return segs


def _all_horizontal_edges(polygon):
    """ポリゴンの全横エッジ [(x0,x1,y), ...] を返す（上端・中段含む）。"""
    segs = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if abs(y1 - y2) < 0.5:
            segs.append((min(x1, x2), max(x1, x2), y1))
    return segs


def _dist_to_bottom_edge(pt, bottom_segs):
    """点から下端横エッジ群までの最短距離。"""
    import math as _m
    x, y = pt
    best = float('inf')
    for (x0, x1, ey) in bottom_segs:
        if x0 <= x <= x1:
            d = abs(y - ey)
        else:
            d = _m.hypot(x - (x0 if x < x0 else x1), y - ey)
        best = min(best, d)
    return best


def region_name_candidates(polygon, labels, max_dist=10.0, min_dist=1.0, min_letters=3,
                           limit=8, exclude_circuit_symbols=True, exclude_terms=('NOTE', '☆'),
                           exclude_lowercase=True, circuit_keep_terms=('RACK',)):
    """領域名候補ラベルを横エッジへの距離順に返す（テキスト重複除去）。

    通常は下端エッジからの距離 [min_dist, max_dist] で評価する。
    候補がゼロの場合は全横エッジ（上端・中段含む）+ min_dist=0 でフォールバック再探索する。
    これにより上端内側に名称が置かれたボックスにも対応する。
    条件:
      - 英字 min_letters 字以上
      - exclude_terms のいずれかを含むラベル（例 NOTE, ☆）は除外
      - exclude_lowercase=True のとき英小文字を含むラベルは除外（領域名は大文字）
      - exclude_circuit_symbols=True のとき機器符号（候補）パターン一致は除外
    """
    terms = [s for s in (exclude_terms or ())]
    bottom = _bottom_edges(polygon)
    if not bottom:
        return []

    def _scan(edge_segs, md_lo):
        cand = []
        for (t, x, y) in labels:
            if _count_letters(t) < min_letters:
                continue
            if exclude_lowercase and any('a' <= ch <= 'z' for ch in t):
                continue
            up = t.upper()
            if any(term.upper() in up for term in terms):
                continue
            if exclude_circuit_symbols and not any(k.upper() in up for k in (circuit_keep_terms or ())):
                matched, _ = filter_non_circuit_symbols([t])
                if matched:
                    continue
            d = _dist_to_bottom_edge((x, y), edge_segs)
            if md_lo <= d <= max_dist:
                cand.append((d, t))
        return cand

    cand = _scan(bottom, min_dist)

    # 候補なし → 全横エッジ（上端含む）+ min_dist=0 でフォールバック
    if not cand:
        all_edges = _all_horizontal_edges(polygon)
        cand = _scan(all_edges, 0.0)

    cand.sort(key=lambda c: c[0])
    seen = set()
    out = []
    for d, t in cand:
        if t in seen:
            continue
        seen.add(t)
        out.append((round(d, 1), t))
        if len(out) >= limit:
            break
    return out


def _is_titleblock_region(polygon, labels):
    """領域内に図番パターンとタイトル系語が同居していれば図番枠とみなす。"""
    has_dn = False
    has_term = False
    terms = ('TITLE', 'REVISION', 'DWG', '流用元', '図番')
    for (t, x, y) in labels:
        if not _point_in_polygon((x, y), polygon):
            continue
        if not has_dn and extract_drawing_numbers(t):
            has_dn = True
        if not has_term:
            up = t.upper()
            if any(k in up or k in t for k in terms):
                has_term = True
        if has_dn and has_term:
            return True
    return False


def analyze_dxf_regions(dxf_file, config=None):
    """DXFファイルを解析し、図面枠・閉領域（名称候補つき）・図面枠内ラベルを返す。

    戻り値 dict:
      frames: [(xl,xr,y0,y1), ...]
      frame_area: float
      labels: [(text, x, y), ...]  （図面枠内のみ）
      regions: [{id, frame, polygon, area, area_pct, name_candidates, default_name}]
      error: str | None
    """
    cfg = dict(DEFAULT_REGION_CONFIG)
    if config:
        cfg.update(config)
    result = {'frames': [], 'frame_area': 0.0, 'labels': [], 'regions': [], 'error': None}
    try:
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()
        frame_lines, region_lines, label_entities, connection_points = \
            _collect_region_geometry(msp, cfg)

        frames = detect_drawing_frames(frame_lines, cfg['snap'])
        result['frames'] = frames
        if not frames:
            result['error'] = ('図面枠（太さ %d の線で囲まれた枠）が見つかりませんでした。'
                               % cfg['frame_lineweight'])
            return result
        frame_area = (frames[0][1] - frames[0][0]) * (frames[0][3] - frames[0][2])
        result['frame_area'] = frame_area

        # 図面枠内ラベル（重複除去）
        seen = set()
        frame_labels = []
        for it in label_entities:
            _, clean_text, (x, y) = extract_text_from_entity(it)
            if not clean_text:
                continue
            in_frame = any(xl - 1 <= x <= xr + 1 and y0 - 1 <= y <= y1 + 1
                           for (xl, xr, y0, y1) in frames)
            if not in_frame:
                continue
            key = (clean_text, round(x, 1), round(y, 1))
            if key in seen:
                continue
            seen.add(key)
            frame_labels.append((clean_text, x, y))
        result['labels'] = frame_labels

        from collections import defaultdict as _dd
        RH, RV = _split_axis_aligned(region_lines, cfg['snap'])
        single_thr = frame_area * cfg['area_ratio']            # 単独領域の閾値(20%)
        group_thr = frame_area * cfg.get('group_area_ratio', 0.10)  # 同名複数ピース合算の閾値(10%)

        # 1) 各図面（フレーム）の候補面を検出（除外適用・名称候補付与）
        frame_cands = []
        for fi, frame in enumerate(frames):
            cands_list = []
            for reg in _detect_regions(RH, RV, frame, frame_area, cfg, frame_labels,
                                       connection_points):
                if cfg['exclude_titleblock'] and _is_titleblock_region(reg['polygon'], frame_labels):
                    continue
                if cfg['exclude_connection_point_regions']:
                    cp = _count_connection_points_on_boundary(
                        reg['polygon'], connection_points, cfg['connection_point_margin'])
                    if cp >= cfg['connection_point_threshold']:
                        continue
                ncands = region_name_candidates(
                    reg['polygon'], frame_labels,
                    max_dist=cfg['name_max_dist'], min_dist=cfg['name_min_dist'],
                    min_letters=cfg['name_min_letters'],
                    exclude_circuit_symbols=cfg['exclude_circuit_symbols'],
                    exclude_terms=cfg['name_exclude_terms'],
                    exclude_lowercase=cfg['name_exclude_lowercase'],
                    circuit_keep_terms=cfg.get('circuit_symbol_keep_terms', ('RACK',)))
                cands_list.append({
                    'polygon': reg['polygon'], 'area': reg['area'],
                    'name_candidates': ncands,
                    'default_name': ncands[0][1] if ncands else '',
                })
            frame_cands.append(cands_list)

        # 2) 第1図面（最左フレーム）で「同名複数ピース合算>=group_thr」となる名称を
        #    ターゲットとする（MPD RACK2 のように2矩形で合算が閾値超のケース）。
        #    他図面では、このターゲット名称の矩形を面積に関係なく抽出する。
        target_names = set()
        if frame_cands:
            by_name = _dd(list)
            for cf in frame_cands[0]:
                if cf['default_name']:
                    by_name[cf['default_name']].append(cf['area'])
            for nm, areas in by_name.items():
                if len(areas) >= 2 and sum(areas) >= group_thr:
                    target_names.add(nm)

        # 3) 採用条件: 個別面積>=単独閾値(20%)、または 名称がターゲット（複数ピース合算で
        #    第1図面が閾値超）。ターゲット名称は他図面でも面積に関係なく採用。
        regions = []
        rid = 0
        for fi, cands_list in enumerate(frame_cands):
            for cf in cands_list:
                if not (cf['area'] >= single_thr
                        or (cf['default_name'] and cf['default_name'] in target_names)):
                    continue
                regions.append({
                    'id': rid,
                    'frame': fi,
                    'polygon': cf['polygon'],
                    'corners': _polygon_corners(cf['polygon']),
                    'area': cf['area'],
                    'area_pct': 100.0 * cf['area'] / frame_area,
                    'name_candidates': cf['name_candidates'],
                    'default_name': cf['default_name'],
                })
                rid += 1
        result['regions'] = regions

        del doc, msp
        gc.collect()
    except Exception as e:
        result['error'] = str(e)
        gc.collect()
    return result


def assign_region_labels(labels, named_regions):
    """各ラベル(text,x,y)が内包される領域名のリストを返す。

    named_regions: [{'polygon': [...], 'name': str}]（名称確定済み）。
    戻り値: [(text, x, y, [region_name, ...]), ...]
    """
    out = []
    for (t, x, y) in labels:
        names = []
        for reg in named_regions:
            nm = reg.get('name')
            if nm and _point_in_polygon((x, y), reg['polygon']) and nm not in names:
                names.append(nm)
        out.append((t, x, y, names))
    return out
