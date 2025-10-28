import os
import tempfile
import traceback
import re

def save_uploadedfile(uploadedfile):
    """アップロードされたファイルを一時ディレクトリに保存する"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploadedfile.name)[1]) as f:
        f.write(uploadedfile.getbuffer())
        return f.name

def handle_error(e, show_traceback=True):
    """エラーを適切に処理して表示する"""
    import streamlit as st
    st.error(f"エラーが発生しました: {str(e)}")
    if show_traceback:
        st.error(traceback.format_exc())

def filter_non_circuit_symbols(labels, debug=False):
    """
    機器符号フォーマットに一致しないラベルをフィルタリングする

    新しい機器符号フォーマット:
    - AA+ (例: CNCNT, FB)
    - A+N+ (例: R10, CN3, PSW1)
    - A+N+A+ (例: X14A, RMSS2A)
    - AA+([内容]) (例: FB(), MSS(MOTOR))
    - A+N+([内容]) (例: R10(2.2K), MSSA(+))
    - A+N+A+([内容]) (例: U23B(DAC))

    Args:
        labels: フィルタリング対象のラベルリスト
        debug: デバッグ情報を出力するかどうか

    Returns:
        tuple: (フィルタリング後のラベルリスト, 除外されたラベル数)
    """

    patterns = [
        # 英文字のみ（2文字以上）
        r'^[A-Za-z]{2,}$',

        # 英文字+数字
        r'^[A-Za-z]+\d+$',

        # 英文字+数字+英文字
        r'^[A-Za-z]+\d+[A-Za-z]+$',

        # 英文字のみ+括弧（オプション）
        r'^[A-Za-z]{2,}\([^)]*\)$',

        # 英文字+数字+括弧（オプション）
        r'^[A-Za-z]+\d+\([^)]*\)$',

        # 英文字+数字+英文字+括弧（オプション）
        r'^[A-Za-z]+\d+[A-Za-z]+\([^)]*\)$',
    ]

    filtered_labels = []
    excluded_count = 0

    for label in labels:
        is_match = False
        for pattern in patterns:
            if re.match(pattern, label):
                is_match = True
                break

        if is_match:
            filtered_labels.append(label)
            if debug:
                print(f"✓ 機器符号として認識: {label}")
        else:
            excluded_count += 1
            if debug:
                print(f"✗ 機器符号として除外: {label}")

    return filtered_labels, excluded_count

def validate_circuit_symbols(labels):
    """
    機器符号の妥当性をチェックし、適合しないものを返す

    Args:
        labels: チェック対象のラベルリスト

    Returns:
        list: 適合しない機器符号のリスト
    """
    # 標準的な機器符号パターンの定義
    standard_patterns = [
        # CB系（遮断器）
        r'^CB\d+$',                 # CB001, CB999
        r'^ELB\(CB\)\d+$',         # ELB(CB)001
        r'^MCCB\d+$',              # MCCB001
        r'^NFB\d+$',               # NFB001

        # 抵抗器
        r'^R\d*$',                 # R, R1, R10

        # コンデンサ
        r'^C\d*$',                 # C, C1, C10

        # インダクタ
        r'^L\d*$',                 # L, L1, L10

        # トランジスタ
        r'^Q\d*$',                 # Q, Q1, Q10

        # IC・集積回路
        r'^U\d*[A-Z]*$',           # U, U1, U10A

        # 電源関連
        r'^PSW?\d*$',              # PS, PSW, PS1, PSW1
        r'^DC\d*$',                # DC, DC1
        r'^AC\d*$',                # AC, AC1

        # モータ・機械系
        r'^M\d*[A-Z]*$',           # M, M1, M1A
        r'^MOT\d*$',               # MOT, MOT1

        # リレー・接触器
        r'^K\d*[A-Z]*$',           # K, K1, K1A
        r'^MC\d*$',                # MC, MC1

        # スイッチ・ボタン
        r'^S\d*[A-Z]*$',           # S, S1, S1A
        r'^SW\d*$',                # SW, SW1
        r'^PB\d*$',                # PB, PB1

        # 表示・ランプ
        r'^H\d*[A-Z]*$',           # H, H1, H1A
        r'^HL\d*$',                # HL, HL1
        r'^PL\d*$',                # PL, PL1

        # 端子・コネクタ
        r'^X\d*[A-Z]*$',           # X, X1, X14A
        r'^CN\d*$',                # CN, CN1
        r'^TB\d*$',                # TB, TB1

        # その他
        r'^F\d*$',                 # F, F1 (ヒューズ)
        r'^T\d*$',                 # T, T1 (変圧器)
        r'^A\d*$',                 # A, A1
    ]

    invalid_symbols = []

    for label in labels:
        is_valid = False
        for pattern in standard_patterns:
            if re.match(pattern, label):
                is_valid = True
                break

        if not is_valid:
            invalid_symbols.append(label)

    return invalid_symbols

def process_circuit_symbol_labels(labels, filter_non_parts=False, validate_ref_designators=False, debug=False):
    """
    ラベルに対して機器符号処理を統合的に実行する

    Args:
        labels: 処理対象のラベルリスト
        filter_non_parts: 機器符号以外のラベルをフィルタリングするかどうか
        validate_ref_designators: 機器符号の妥当性をチェックするかどうか
        debug: デバッグ情報を表示するかどうか

    Returns:
        dict: 処理結果を含む辞書
            - 'labels': 処理後のラベルリスト
            - 'filtered_count': フィルタリングで除外されたラベル数
            - 'invalid_ref_designators': 適合しない機器符号のリスト（妥当性チェック有効時のみ）
    """
    result = {
        'labels': labels.copy(),
        'filtered_count': 0,
        'invalid_ref_designators': []
    }

    # フィルタリング処理
    if filter_non_parts:
        filtered_labels, filtered_count = filter_non_circuit_symbols(labels, debug)
        result['labels'] = filtered_labels
        result['filtered_count'] = filtered_count

    # 機器符号妥当性チェック（フィルタリング後のラベルに対して実行）
    if validate_ref_designators and filter_non_parts:
        invalid_designators = validate_circuit_symbols(result['labels'])
        result['invalid_ref_designators'] = invalid_designators

    return result
