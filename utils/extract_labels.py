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


def determine_drawing_number_types(drawing_numbers: List[Tuple[str, Tuple[float, float]]]) -> Dict[str, str]:
    """
    座標に基づいて図番と流用元図番を判別する

    Args:
        drawing_numbers: (図面番号, (X座標, Y座標))のリスト

    Returns:
        dict: {'main_drawing': '図番', 'source_drawing': '流用元図番'}
    """
    if len(drawing_numbers) < 2:
        # 図面番号が1つまたは0個の場合
        if len(drawing_numbers) == 1:
            return {'main_drawing': drawing_numbers[0][0], 'source_drawing': None}
        else:
            return {'main_drawing': None, 'source_drawing': None}

    # 座標でソート（右下が図番、それ以外が流用元図番）
    # 一般的に図番は図面の右下に配置される
    sorted_numbers = sorted(drawing_numbers, key=lambda x: (x[1][0] + x[1][1]), reverse=True)

    # 最も右下にあるものを図番とする
    main_drawing = sorted_numbers[0][0]

    # 残りの中で最も座標値が大きいものを流用元図番とする
    if len(sorted_numbers) > 1:
        source_drawing = sorted_numbers[1][0]
    else:
        source_drawing = None

    return {'main_drawing': main_drawing, 'source_drawing': source_drawing}


def extract_labels(dxf_file, filter_non_parts=False, sort_order="asc", debug=False,
                  selected_layers=None, validate_ref_designators=False,
                  extract_drawing_numbers_option=False):
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
        "all_drawing_numbers": []        # 抽出されたすべての図面番号
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

        # 図面番号の判別
        if extract_drawing_numbers_option and drawing_number_candidates:
            drawing_info = determine_drawing_number_types(drawing_number_candidates)
            info["main_drawing_number"] = drawing_info['main_drawing']
            info["source_drawing_number"] = drawing_info['source_drawing']
            info["all_drawing_numbers"] = [dn[0] for dn in drawing_number_candidates]



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
                              extract_drawing_numbers_option=False):
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
                            extract_drawing_numbers_option
                        )
                        results[file_path] = (labels, info)
        # 単一のDXFファイルの場合
        elif os.path.isfile(dxf_file) and dxf_file.lower().endswith('.dxf'):
            labels, info = extract_labels(
                dxf_file, filter_non_parts, sort_order, debug,
                selected_layers, validate_ref_designators,
                extract_drawing_numbers_option
            )
            results[dxf_file] = (labels, info)

    return results
