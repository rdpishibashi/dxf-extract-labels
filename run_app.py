import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


APP_NAME = 'DXF-Extract-Labels'
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8501


def detect_network_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(('8.8.8.8', 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith('127.'):
                return ip
        except OSError:
            pass

    for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ip = sockaddr[0]
        if ip and not ip.startswith('127.'):
            return ip

    return '127.0.0.1'


def get_downloads_dir() -> Path:
    user_profile = os.environ.get('USERPROFILE')
    if user_profile:
        return Path(user_profile) / 'Downloads'
    return Path.home() / 'Downloads'


def write_launch_info(runtime_dir: Path, local_url: str, network_url: str) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    launch_info = runtime_dir / f'{APP_NAME}-urls.txt'
    launch_info.write_text(
        f'Local URL: {local_url}\nNetwork URL: {network_url}\n',
        encoding='utf-8',
    )


def open_browser_when_ready(url: str, host: str = '127.0.0.1', port: int = SERVER_PORT, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                webbrowser.open(url, new=0, autoraise=False)
                return
        time.sleep(0.5)


def configure_environment(bundle_dir: Path) -> None:
    secrets_path = bundle_dir / '.streamlit' / 'secrets.toml'
    if secrets_path.exists():
        os.environ['STREAMLIT_SECRETS_FILES'] = str(secrets_path)


def main() -> None:
    frozen = getattr(sys, 'frozen', False)
    if frozen:
        bundle_dir = Path(sys._MEIPASS)
        runtime_dir = get_downloads_dir()
    else:
        bundle_dir = Path(__file__).resolve().parent
        runtime_dir = bundle_dir

    os.chdir(bundle_dir)
    configure_environment(bundle_dir)

    local_url = f'http://127.0.0.1:{SERVER_PORT}'
    network_ip = detect_network_ip()
    network_url = f'http://{network_ip}:{SERVER_PORT}'
    write_launch_info(runtime_dir, local_url, network_url)
    threading.Thread(target=open_browser_when_ready, args=(local_url,), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        'streamlit', 'run',
        str(bundle_dir / 'app.py'),
        '--server.address', SERVER_HOST,
        '--server.port', str(SERVER_PORT),
        '--server.headless', 'true',
        '--global.developmentMode=false',
        '--browser.gatherUsageStats=false',
    ]
    stcli.main()


if __name__ == '__main__':
    main()
