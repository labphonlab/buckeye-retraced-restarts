#!/usr/bin/env python3
"""
14_boundary_check_serve.py — 語境界の目視確認ツールを手元のブラウザで開く

tools/boundary_check.html と boundary_check/ の items.csv・clips/*.wav だけを
127.0.0.1 で配信し、ブラウザでツールを開く。ツールはフォルダを選ばずに自動で読み込む。
key.csv（条件の対応表）など、それ以外のファイルは配信しない（判定のブラインドを保つため）。
判定はページから送られるたびに boundary_check/results.csv に書き出す（自動保存）。

使い方:  python3 scripts/14_boundary_check_serve.py   （終了は Ctrl+C）
"""
import argparse
import http.server
import os
import re
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ALLOWED = {
    "/tools/boundary_check.html": os.path.join(ROOT, "tools", "boundary_check.html"),
    "/boundary_check/items.csv": os.path.join(ROOT, "boundary_check", "items.csv"),
    "/boundary_check/results.csv": os.path.join(ROOT, "boundary_check", "results.csv"),  # 再開用（条件は含まない）
}
CLIP = re.compile(r"^/boundary_check/clips/(bc\d+\.wav)$")
RESULTS = os.path.join(ROOT, "boundary_check", "results.csv")
MAX_POST = 1_000_000
WRITE_LOCK = threading.Lock()  # 同時に届いた保存で一時ファイルが混ざらないようにする
TYPES = {".html": "text/html; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".wav": "audio/wav"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/tools/boundary_check.html")
            self.end_headers()
            return
        fp = ALLOWED.get(path)
        m = CLIP.match(path)
        if m:
            fp = os.path.join(ROOT, "boundary_check", "clips", m.group(1))
        if not fp or not os.path.isfile(fp):
            self.send_error(404)
            return
        with open(fp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", TYPES[os.path.splitext(fp)[1]])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        # 結果の自動保存。同じサーバーから開いたページ以外からの書き込みは受けない
        origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path != "/boundary_check/results.csv" or self.headers.get("Origin") != origin:
            self.send_error(403)
            return
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_POST:
            self.send_error(413)
            return
        body = self.rfile.read(n)
        if not body.lstrip(b"\xef\xbb\xbf").startswith(b"item_id,"):
            self.send_error(400)
            return
        tmp = RESULTS + ".tmp"
        with WRITE_LOCK:
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, RESULTS)  # 書きかけのファイルが残らないように置き換える
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):  # 1件ごとのアクセス記録は出さない
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを開かない")
    args = ap.parse_args()
    if not os.path.isfile(ALLOWED["/boundary_check/items.csv"]):
        raise SystemExit("boundary_check/items.csv がない。先に scripts/14_boundary_check_sample.py を実行する。")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/tools/boundary_check.html"
    print(f"{url} で配信中（終了は Ctrl+C）")
    print(f"判定は {os.path.relpath(RESULTS, ROOT)} に自動保存する")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
