"""判断ログ（未確定ラベルの採用/非採用）の記録層（v1.7.0）。

「未確定ラベル」UI でユーザーが行った手動判断（採用=adopted / 非採用=rejected）を
追記型 CSV として蓄積する。蓄積したログは tools/reference_designator_analyzer.py が
読み込み、確定パターン・除外パターンの候補提案に使う。

バックエンド（record() が github_conf の有無で選択する）:
- GitHubBackend: Streamlit Community Cloud 用。呼び出し元（View 層）が
  st.secrets['github'] を読み取って渡す PAT で、ログ専用リポジトリ
  （既定 rdpishibashi/dxf-label-decisions）の decision_log.csv に
  Contents API で追記する。アプリ本体のリポジトリに置くとコミットごとに
  Streamlit Cloud が再デプロイされるため、必ず別リポジトリを使うこと。
- FileBackend: ローカル / Windows アプリ用。既定 ~/Documents/DXF-extract-labels/
  decision_log.csv に追記する。環境変数 DXF_DECISION_LOG_PATH で変更可能
  （Dropbox 配下を指定すれば複数 PC のログを自動的に一元化できる）。

このモジュールは streamlit に依存しない（モデル層）。st.secrets へのアクセスは
呼び出し元の責務。record() は例外を外に出さない（記録の失敗で抽出本体を止めない
ため）。失敗時は (False, メッセージ) を返し、呼び出し元がフォールバック
（entries_to_csv_bytes() によるダウンロード提供）を行う。
"""

import base64
import csv
import io
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CSV_HEADER = [
    'timestamp', 'source', 'file_name', 'drawing_number',
    'label', 'decision', 'count', 'app_version', 'patterns_version',
]

DEFAULT_LOG_FILENAME = 'decision_log.csv'
DEFAULT_GITHUB_PATH = 'decision_log.csv'
DEFAULT_GITHUB_BRANCH = 'main'
_GITHUB_API = 'https://api.github.com'


def build_entries(
    file_name: str,
    drawing_number: Optional[str],
    review_labels: Iterable[Tuple[str, float, float]],
    approved: set,
    app_version: str,
    patterns_version: str,
    source: str = '',
    timestamp: Optional[str] = None,
) -> List[Dict[str, str]]:
    """1ファイル分の未確定ラベル判断をログエントリ（dict のリスト）にする。

    review_labels は (正規化済みラベル, x, y) のリスト（重複あり＝図面内の個数）。
    ユニークなラベルごとに1行とし、approved に含まれれば adopted、
    含まれなければ rejected。source は空なら record() 時にバックエンドが埋める。
    """
    counts: Dict[str, int] = {}
    for t, _x, _y in review_labels:
        counts[t] = counts.get(t, 0) + 1
    ts = timestamp or datetime.now().astimezone().isoformat(timespec='seconds')
    entries = []
    for label in sorted(counts):
        entries.append({
            'timestamp': ts,
            'source': source,
            'file_name': file_name,
            'drawing_number': drawing_number or '',
            'label': label,
            'decision': 'adopted' if label in approved else 'rejected',
            'count': str(counts[label]),
            'app_version': app_version,
            'patterns_version': patterns_version,
        })
    return entries


def entries_to_csv_bytes(entries: List[Dict[str, str]], with_header: bool = True) -> bytes:
    """エントリを CSV バイト列にする（記録失敗時のフォールバックダウンロード用）。

    utf-8-sig で返す（Windows の Excel でそのまま文字化けせずに開けるため）。
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER, lineterminator='\n')
    if with_header:
        writer.writeheader()
    for e in entries:
        writer.writerow(e)
    return buf.getvalue().encode('utf-8-sig')


def _entries_to_csv_lines(entries: List[Dict[str, str]]) -> str:
    """ヘッダーなしの CSV 行文字列（既存ファイルへの追記用）。"""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER, lineterminator='\n')
    for e in entries:
        writer.writerow(e)
    return buf.getvalue()


class FileBackend:
    """ローカル CSV への追記。ファイルがなければヘッダー付きで新規作成する。"""

    name = 'ローカルファイル'

    def __init__(self, path: Optional[str] = None):
        self.path = Path(
            path
            or os.environ.get('DXF_DECISION_LOG_PATH')
            or Path.home() / 'Documents' / 'DXF-extract-labels' / DEFAULT_LOG_FILENAME
        )

    def default_source(self) -> str:
        return socket.gethostname()

    def append(self, entries: List[Dict[str, str]]) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        # utf-8-sig は新規作成時のみ（追記で BOM を混入させない）
        with open(self.path, 'a', encoding='utf-8-sig' if is_new else 'utf-8',
                  newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, lineterminator='\n')
            if is_new:
                writer.writeheader()
            for e in entries:
                writer.writerow(e)
        return f"{self.name}（{self.path}）に {len(entries)} 件を記録しました"


class GitHubBackend:
    """GitHub Contents API によるログ専用リポジトリの CSV への追記。

    競合（他セッションが先にコミット。HTTP 409/422）時は sha を取り直して
    1回だけリトライする。ファイルが存在しない場合はヘッダー付きで新規作成する。
    """

    name = 'GitHub'

    def __init__(self, token: str, repo: str,
                 path: str = DEFAULT_GITHUB_PATH, branch: str = DEFAULT_GITHUB_BRANCH):
        self.token = token
        self.repo = repo
        self.path = path
        self.branch = branch

    def default_source(self) -> str:
        return 'cloud'

    def _headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        }

    def _get_current(self, requests) -> Tuple[Optional[str], str]:
        """(sha, 既存テキスト) を返す。ファイルが無ければ (None, '')。"""
        url = f'{_GITHUB_API}/repos/{self.repo}/contents/{self.path}'
        r = requests.get(url, headers=self._headers(),
                         params={'ref': self.branch}, timeout=15)
        if r.status_code == 404:
            return None, ''
        r.raise_for_status()
        data = r.json()
        text = base64.b64decode(data['content']).decode('utf-8')
        return data['sha'], text

    def _put(self, requests, sha: Optional[str], text: str, n: int):
        url = f'{_GITHUB_API}/repos/{self.repo}/contents/{self.path}'
        payload = {
            'message': f'log: 判断ログ {n} 件を追記',
            'content': base64.b64encode(text.encode('utf-8')).decode('ascii'),
            'branch': self.branch,
        }
        if sha:
            payload['sha'] = sha
        r = requests.put(url, headers=self._headers(), json=payload, timeout=15)
        r.raise_for_status()

    def append(self, entries: List[Dict[str, str]]) -> str:
        import requests
        new_lines = _entries_to_csv_lines(entries)
        last_error = None
        for _attempt in range(2):
            sha, text = self._get_current(requests)
            if not text:
                text = ','.join(CSV_HEADER) + '\n'
            elif not text.endswith('\n'):
                text += '\n'
            try:
                self._put(requests, sha, text + new_lines, len(entries))
                return f"{self.name}（{self.repo}/{self.path}）に {len(entries)} 件を記録しました"
            except requests.HTTPError as e:
                # 409/422: 他のコミットと競合 → sha を取り直して1回だけリトライ
                status = e.response.status_code if e.response is not None else None
                if status in (409, 422):
                    last_error = e
                    continue
                raise
        raise last_error


def fetch_log_text(token: str, repo: str,
                    path: str = DEFAULT_GITHUB_PATH,
                    branch: str = DEFAULT_GITHUB_BRANCH) -> str:
    """ログ専用リポジトリから decision_log.csv の内容をテキストで取得する（分析用の読み取り専用）。

    ファイルが存在しない場合は空文字列を返す。`tools/reference_designator_analyzer.py`
    のログ集計モードが使う（`GitHubBackend` は追記=書き込み専用のため、読み取り専用の
    この関数を別に用意する）。
    """
    import requests
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    url = f'{_GITHUB_API}/repos/{repo}/contents/{path}'
    r = requests.get(url, headers=headers, params={'ref': branch}, timeout=15)
    if r.status_code == 404:
        return ''
    r.raise_for_status()
    data = r.json()
    return base64.b64decode(data['content']).decode('utf-8')


def _github_backend_from_conf(conf: Optional[dict]) -> Optional[GitHubBackend]:
    """github 設定 dict（token/repo/[path]/[branch]）があれば GitHubBackend を作る。

    conf は呼び出し元（View 層）が `st.secrets.get('github')` 等で読み取って渡す。
    st.secrets へのアクセス自体（secrets.toml が無い環境での例外）は呼び出し元の
    責務であり、ここでは扱わない（モデル層を streamlit 非依存に保つため）。
    """
    if not conf:
        return None
    token = conf.get('token')
    repo = conf.get('repo')
    if not token or not repo:
        return None
    return GitHubBackend(
        token=token, repo=repo,
        path=conf.get('path', DEFAULT_GITHUB_PATH),
        branch=conf.get('branch', DEFAULT_GITHUB_BRANCH),
    )


def pick_backend(github_conf: Optional[dict] = None):
    """利用可能なバックエンドを返す（github_conf があれば GitHub、無ければローカルファイル）。"""
    return _github_backend_from_conf(github_conf) or FileBackend()


def record(entries: List[Dict[str, str]], github_conf: Optional[dict] = None) -> Tuple[bool, str]:
    """エントリを記録する。例外を外に出さず (成功?, メッセージ) を返す。

    github_conf は st.secrets['github'] 相当の dict（呼び出し元が読み取って渡す）。
    """
    if not entries:
        return True, '記録対象の判断はありませんでした（未確定ラベルなし）'
    try:
        backend = pick_backend(github_conf)
        src = backend.default_source()
        for e in entries:
            if not e.get('source'):
                e['source'] = src
        return True, backend.append(entries)
    except Exception as e:
        return False, f'判断ログの記録に失敗しました: {e}'
