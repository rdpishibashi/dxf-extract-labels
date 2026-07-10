"""utils/decision_log.py の単体テスト（モデル層のみ、Streamlit UI 非依存）。

組み合わせ表:
  - build_entries: 採用/非採用混在、同一ラベル複数出現（個数集計）
  - FileBackend: 新規作成時（ヘッダー付与）/ 既存ファイルへの追記（ヘッダーなし）
  - GitHubBackend: 新規ファイル（404）/ 既存ファイル（sha あり）/ 競合リトライ（409→成功）
  - pick_backend: st.secrets['github'] あり→GitHubBackend / なし→FileBackend
  - record(): 空エントリ→記録スキップ
"""
import base64
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import decision_log


class TestBuildEntries(unittest.TestCase):
    def test_counts_and_decision_split(self):
        review_labels = [
            ('R10', 0, 0), ('R10', 1, 1),  # 同一ラベルが2回出現→個数2
            ('CN3', 2, 2),
        ]
        entries = decision_log.build_entries(
            file_name='a.dxf', drawing_number='EE1234-500-01A',
            review_labels=review_labels, approved={'R10'},
            app_version='1.7.0', patterns_version='1.6.4',
        )
        by_label = {e['label']: e for e in entries}
        self.assertEqual(by_label['R10']['decision'], 'adopted')
        self.assertEqual(by_label['R10']['count'], '2')
        self.assertEqual(by_label['CN3']['decision'], 'rejected')
        self.assertEqual(by_label['CN3']['count'], '1')
        self.assertEqual(by_label['R10']['app_version'], '1.7.0')
        self.assertEqual(by_label['R10']['patterns_version'], '1.6.4')
        self.assertEqual(by_label['R10']['file_name'], 'a.dxf')
        self.assertEqual(by_label['R10']['drawing_number'], 'EE1234-500-01A')

    def test_empty_review_labels_yields_no_entries(self):
        entries = decision_log.build_entries(
            file_name='a.dxf', drawing_number=None,
            review_labels=[], approved=set(),
            app_version='1.7.0', patterns_version='1.6.4',
        )
        self.assertEqual(entries, [])

    def test_missing_drawing_number_becomes_empty_string(self):
        entries = decision_log.build_entries(
            file_name='a.dxf', drawing_number=None,
            review_labels=[('X1', 0, 0)], approved=set(),
            app_version='1.7.0', patterns_version='1.6.4',
        )
        self.assertEqual(entries[0]['drawing_number'], '')


class TestEntriesToCsvBytes(unittest.TestCase):
    def test_header_and_bom(self):
        entries = decision_log.build_entries(
            file_name='a.dxf', drawing_number=None,
            review_labels=[('X1', 0, 0)], approved={'X1'},
            app_version='1.7.0', patterns_version='1.6.4',
        )
        data = decision_log.entries_to_csv_bytes(entries)
        text = data.decode('utf-8-sig')
        lines = text.strip().split('\n')
        self.assertEqual(lines[0], ','.join(decision_log.CSV_HEADER))
        self.assertIn('X1', lines[1])
        self.assertIn('adopted', lines[1])


class TestFileBackend(unittest.TestCase):
    def test_creates_new_file_with_header_then_appends_without_header(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'nested', 'decision_log.csv')
            backend = decision_log.FileBackend(path=path)

            entries1 = decision_log.build_entries(
                file_name='a.dxf', drawing_number=None,
                review_labels=[('X1', 0, 0)], approved={'X1'},
                app_version='1.7.0', patterns_version='1.6.4', source='hostA',
            )
            msg1 = backend.append(entries1)
            self.assertIn('1 件', msg1)

            entries2 = decision_log.build_entries(
                file_name='b.dxf', drawing_number=None,
                review_labels=[('X2', 0, 0)], approved=set(),
                app_version='1.7.0', patterns_version='1.6.4', source='hostA',
            )
            backend.append(entries2)

            with open(path, encoding='utf-8-sig') as f:
                content = f.read()
            lines = content.strip().split('\n')
            self.assertEqual(lines[0], ','.join(decision_log.CSV_HEADER))
            # ヘッダーは1回だけ（2回目の追記でヘッダーが重複しない）
            self.assertEqual(sum(1 for l in lines if l == ','.join(decision_log.CSV_HEADER)), 1)
            self.assertEqual(len(lines), 3)  # header + 2 data rows

    def test_default_source_is_hostname(self):
        backend = decision_log.FileBackend(path='/tmp/unused.csv')
        import socket
        self.assertEqual(backend.default_source(), socket.gethostname())


class TestGitHubBackend(unittest.TestCase):
    def _entries(self):
        return decision_log.build_entries(
            file_name='a.dxf', drawing_number=None,
            review_labels=[('X1', 0, 0)], approved={'X1'},
            app_version='1.7.0', patterns_version='1.6.4', source='cloud',
        )

    def test_append_creates_new_file_when_404(self):
        backend = decision_log.GitHubBackend(token='t', repo='org/repo')
        get_resp = MagicMock(status_code=404)
        put_resp = MagicMock(status_code=201)
        put_resp.raise_for_status = MagicMock()

        with patch('requests.get', return_value=get_resp) as mock_get, \
             patch('requests.put', return_value=put_resp) as mock_put:
            msg = backend.append(self._entries())

        self.assertIn('1 件', msg)
        mock_get.assert_called_once()
        put_kwargs = mock_put.call_args.kwargs
        self.assertNotIn('sha', put_kwargs['json'])
        decoded = base64.b64decode(put_kwargs['json']['content']).decode('utf-8')
        self.assertTrue(decoded.startswith(','.join(decision_log.CSV_HEADER)))
        self.assertIn('X1', decoded)

    def test_append_to_existing_file_includes_sha_and_preserves_content(self):
        backend = decision_log.GitHubBackend(token='t', repo='org/repo')
        existing_text = ','.join(decision_log.CSV_HEADER) + '\n2026-01-01T00:00:00,cloud,old.dxf,,OLD,adopted,1,1.0,1.0\n'
        get_resp = MagicMock(status_code=200)
        get_resp.json.return_value = {
            'sha': 'abc123',
            'content': base64.b64encode(existing_text.encode('utf-8')).decode('ascii'),
        }
        get_resp.raise_for_status = MagicMock()
        put_resp = MagicMock(status_code=200)
        put_resp.raise_for_status = MagicMock()

        with patch('requests.get', return_value=get_resp), \
             patch('requests.put', return_value=put_resp) as mock_put:
            backend.append(self._entries())

        put_kwargs = mock_put.call_args.kwargs
        self.assertEqual(put_kwargs['json']['sha'], 'abc123')
        decoded = base64.b64decode(put_kwargs['json']['content']).decode('utf-8')
        self.assertIn('OLD', decoded)  # 既存行が保持される
        self.assertIn('X1', decoded)  # 新規行が追記される

    def test_append_retries_once_on_conflict(self):
        import requests as real_requests
        backend = decision_log.GitHubBackend(token='t', repo='org/repo')
        get_resp = MagicMock(status_code=404)
        conflict_resp = MagicMock(status_code=409)
        conflict_error = real_requests.HTTPError(response=conflict_resp)
        put_resp_fail = MagicMock()
        put_resp_fail.raise_for_status.side_effect = conflict_error
        put_resp_ok = MagicMock(status_code=201)
        put_resp_ok.raise_for_status = MagicMock()

        with patch('requests.get', return_value=get_resp) as mock_get, \
             patch('requests.put', side_effect=[put_resp_fail, put_resp_ok]) as mock_put:
            msg = backend.append(self._entries())

        self.assertIn('1 件', msg)
        self.assertEqual(mock_get.call_count, 2)  # 1回目失敗後、sha取り直しで2回目
        self.assertEqual(mock_put.call_count, 2)

    def test_default_source_is_cloud(self):
        backend = decision_log.GitHubBackend(token='t', repo='org/repo')
        self.assertEqual(backend.default_source(), 'cloud')


class TestFetchLogText(unittest.TestCase):
    def test_returns_empty_string_when_not_found(self):
        resp = MagicMock(status_code=404)
        with patch('requests.get', return_value=resp):
            text = decision_log.fetch_log_text(token='t', repo='org/repo')
        self.assertEqual(text, '')

    def test_returns_decoded_content_when_found(self):
        content = ','.join(decision_log.CSV_HEADER) + '\n'
        resp = MagicMock(status_code=200)
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            'content': base64.b64encode(content.encode('utf-8')).decode('ascii'),
        }
        with patch('requests.get', return_value=resp) as mock_get:
            text = decision_log.fetch_log_text(token='t', repo='org/repo', branch='main')
        self.assertEqual(text, content)
        self.assertEqual(mock_get.call_args.kwargs['params'], {'ref': 'main'})


class TestPickBackendAndRecord(unittest.TestCase):
    def test_pick_backend_prefers_github_when_secrets_present(self):
        fake_secrets = {'github': {'token': 'tok', 'repo': 'org/repo'}}
        with patch('streamlit.secrets', fake_secrets):
            backend = decision_log.pick_backend()
        self.assertIsInstance(backend, decision_log.GitHubBackend)
        self.assertEqual(backend.repo, 'org/repo')

    def test_pick_backend_falls_back_to_file_when_no_secrets(self):
        with patch('streamlit.secrets', {}):
            backend = decision_log.pick_backend()
        self.assertIsInstance(backend, decision_log.FileBackend)

    def test_pick_backend_falls_back_when_secrets_access_raises(self):
        class RaisingSecrets:
            def get(self, *a, **kw):
                raise RuntimeError('no secrets.toml')
        with patch('streamlit.secrets', RaisingSecrets()):
            backend = decision_log.pick_backend()
        self.assertIsInstance(backend, decision_log.FileBackend)

    def test_record_empty_entries_skips_backend(self):
        ok, msg = decision_log.record([])
        self.assertTrue(ok)
        self.assertIn('未確定ラベルなし', msg)

    def test_record_success_fills_source_when_missing(self):
        entries = [{
            'timestamp': 't', 'source': '', 'file_name': 'a.dxf',
            'drawing_number': '', 'label': 'X1', 'decision': 'adopted',
            'count': '1', 'app_version': '1.7.0', 'patterns_version': '1.6.4',
        }]
        fake_backend = MagicMock()
        fake_backend.default_source.return_value = 'hostZ'
        fake_backend.append.return_value = 'ok'
        with patch.object(decision_log, 'pick_backend', return_value=fake_backend):
            ok, msg = decision_log.record(entries)
        self.assertTrue(ok)
        self.assertEqual(entries[0]['source'], 'hostZ')

    def test_record_failure_returns_false_with_message(self):
        fake_backend = MagicMock()
        fake_backend.default_source.return_value = 'hostZ'
        fake_backend.append.side_effect = RuntimeError('network down')
        entries = [{
            'timestamp': 't', 'source': '', 'file_name': 'a.dxf',
            'drawing_number': '', 'label': 'X1', 'decision': 'adopted',
            'count': '1', 'app_version': '1.7.0', 'patterns_version': '1.6.4',
        }]
        with patch.object(decision_log, 'pick_backend', return_value=fake_backend):
            ok, msg = decision_log.record(entries)
        self.assertFalse(ok)
        self.assertIn('network down', msg)


if __name__ == '__main__':
    unittest.main()
