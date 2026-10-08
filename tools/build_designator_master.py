#!/usr/bin/env python3
"""機器符号一覧（designator master）を原本 Excel から生成する CLI。

使い方:
    python tools/build_designator_master.py <原本.xls> [--out-dir data] [--digest-out PATH]

原本ファイルのパスは引数で必ず指定する（ハードコードしない。原本は
ULVAC の設計情報であり、このリポジトリ外のパスを指す想定のため）。

実行結果として `<out-dir>/designator_master.csv` と
`<out-dir>/designator_master.xlsx`（いずれも `.gitignore` 対象、平文）を
生成し、実測の不変条件（件数）を標準出力に表示する
（`docs/REF_DESIGNATOR.md`「機器符号一覧」節の実測値と一致するかどうかの確認用）。

**鍵（`DESIGNATOR_MASTER_KEY`）が解決できた場合のみ**、HMACダイジェストファイル
（既定 `assets/designator_digests.txt`、git管理対象）も生成する。鍵は
環境変数 → `.streamlit/secrets.toml` の順に解決する（`model.designator_master.
resolve_secret()`）。鍵が解決できない場合はダイジェスト生成をスキップし、
その旨を表示する（平文CSV/Excelの生成は従来どおり行う）。

**鍵の値そのものは標準出力に一切表示しない**（表示するのは鍵フィンガープリント
——鍵自体ではない検査用の短い値のみ）。鍵はこのツールが生成することもない
（`python3 -c "import secrets; print(secrets.token_hex(32))"` 等で
利用者自身が1回だけ生成し、`.streamlit/secrets.toml` に配置する）。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.designator_master import (  # noqa: E402
    build_designator_master, key_fingerprint, resolve_secret, write_csv, write_digest_file,
    write_excel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xls_path', help='原本 Excel ファイル（.xls）のパス')
    parser.add_argument('--out-dir', default=os.path.join(PROJECT_ROOT, 'data'),
                         help='平文CSV/Excelの出力先ディレクトリ（既定: data/、.gitignore対象）')
    parser.add_argument('--digest-out',
                         default=os.path.join(PROJECT_ROOT, 'assets', 'designator_digests.txt'),
                         help='HMACダイジェストファイルの出力先（既定: assets/designator_digests.txt、'
                              'git管理対象）')
    args = parser.parse_args()

    if not os.path.isfile(args.xls_path):
        print(f'エラー: 原本ファイルが見つかりません: {args.xls_path}', file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    result = build_designator_master(args.xls_path)

    csv_path = os.path.join(args.out_dir, 'designator_master.csv')
    xlsx_path = os.path.join(args.out_dir, 'designator_master.xlsx')
    write_csv(result, csv_path)
    write_excel(result, xlsx_path)

    stats = result.stats
    print(f'原本: {args.xls_path}')
    print(f'分類（シート）数: {stats["sheet_count"]}')
    print(f'取り消し線で除外: 全体{stats["struck_whole_excluded"]}件 / 部分{stats["struck_partial_excluded"]}件')
    print(f'展開後の総行数（S系除外前）: {stats["raw_expanded_count"]}')
    print(f'S系除外: 用途空{stats["spare_excluded_empty"]}件 + 枠取り{stats["spare_excluded_frame"]}件 '
          f'= {stats["spare_excluded_total"]}件')
    print(f'最終行数: {stats["final_count"]}')
    print(f'未処理（警告）件数: {stats["warning_count"]}')
    if result.warnings:
        print('--- 警告の内容 ---')
        for w in result.warnings:
            print(f'  [{w.category}] 行{w.sheet_row}: {w.message}')
    print(f'出力: {csv_path}')
    print(f'出力: {xlsx_path}')

    secret = resolve_secret()
    if not secret:
        print()
        print('DESIGNATOR_MASTER_KEY が未設定のため、ダイジェストファイルの生成を'
              'スキップしました（環境変数、または .streamlit/secrets.toml に設定してください）。')
        return 0

    os.makedirs(os.path.dirname(args.digest_out) or '.', exist_ok=True)
    write_digest_file(result, args.digest_out, secret)
    unique_codes = len({row.code for row in result.rows})
    print()
    print(f'ダイジェスト件数: 最大{unique_codes}件（正規化による重複統合後はこれ以下）')
    print(f'鍵フィンガープリント: {key_fingerprint(secret)}')
    print(f'出力: {args.digest_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
