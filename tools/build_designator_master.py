#!/usr/bin/env python3
"""機器符号一覧（designator master）を原本 Excel から生成する CLI。

使い方:
    python tools/build_designator_master.py <原本.xls> [--out-dir data]

原本ファイルのパスは引数で必ず指定する（ハードコードしない。原本は
ULVAC の設計情報であり、このリポジトリ外のパスを指す想定のため）。

実行結果として `<out-dir>/designator_master.csv` と
`<out-dir>/designator_master.xlsx` を生成し、実測の不変条件（件数）を
標準出力に表示する（`docs/REF_DESIGNATOR.md`「機器符号一覧」節の実測値と
一致するかどうかの確認用）。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.designator_master import build_designator_master, write_csv, write_excel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xls_path', help='原本 Excel ファイル（.xls）のパス')
    parser.add_argument('--out-dir', default=os.path.join(PROJECT_ROOT, 'data'),
                         help='出力先ディレクトリ（既定: data/）')
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
