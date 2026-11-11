#!/usr/bin/env python3
"""
16_onset_serve.py — 語頭の付け直しツールを手元のブラウザで開く（DECISIONS.md D23）

tools/onset_mark.html と onset_check/ の items.csv・clips/*.wav だけを 127.0.0.1 で配信する。
key.csv（条件とコーパス時刻の対応表）は配信しない（ブラインドを保つため）。
付けた位置はページから送られるたびに onset_check/results.csv に書き出す（自動保存）。
配信と保存の仕組みは 14_boundary_check_serve.py と同じ（同じ Handler を、配信先だけ替えて使う）。

使い方:  python3 scripts/16_onset_serve.py   （終了は Ctrl+C）
"""
import argparse
import errno
import http.server
import importlib.util
import os
import re
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("bcs", os.path.join(HERE, "14_boundary_check_serve.py"))
bcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcs)

KIT = os.path.join(ROOT, "onset_check")
bcs.ALLOWED = {
    "/tools/onset_mark.html": os.path.join(ROOT, "tools", "onset_mark.html"),
    "/onset_check/items.csv": os.path.join(KIT, "items.csv"),
    "/onset_check/results.csv": os.path.join(KIT, "results.csv"),  # 再開用（条件・コーパス時刻は含まない）
}
bcs.CLIP = re.compile(r"^/onset_check/clips/(on\d+\.wav)$")
bcs.RESULTS = os.path.join(KIT, "results.csv")


class Handler(bcs.Handler):
    def do_GET(self):
        if self.path.split("?", 1)[0] == "/":
            self.send_response(302)
            self.send_header("Location", "/tools/onset_mark.html")
            self.end_headers()
            return
        m = bcs.CLIP.match(self.path.split("?", 1)[0])
        if m:  # 14 の Handler はクリップを boundary_check/clips から引くので、ここで差し替える
            fp = os.path.join(KIT, "clips", m.group(1))
            if not os.path.isfile(fp):
                self.send_error(404)
                return
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/onset_check/results.csv":
            self.send_error(403)
            return
        self.path = "/boundary_check/results.csv"  # 14 の Handler は保存先の確認にこのパスを使う（書き先は bcs.RESULTS）
        super().do_POST()


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを開かない")
    args = ap.parse_args()
    if not os.path.isfile(bcs.ALLOWED["/onset_check/items.csv"]):
        raise SystemExit("onset_check/items.csv がない。先に scripts/16_onset_kit.py を実行する。")
    url = f"http://127.0.0.1:{args.port}/tools/onset_mark.html"
    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        raise SystemExit(f"ポート {args.port} は使用中。ツールは既に起動しているはずなので、{url} をブラウザで開く。")
    print(f"{url} で配信中（終了は Ctrl+C）")
    print(f"付けた位置は {os.path.relpath(bcs.RESULTS, ROOT)} に自動保存する")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
