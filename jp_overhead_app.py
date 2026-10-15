#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jp_overhead_app.py - 日本株「上値の重さ」検索画面(ローカルWebアプリ)

  python jp_overhead_app.py            → ブラウザが開きます (http://localhost:8765)
  python jp_overhead_app.py --demo     → 合成データで動作確認(ネット不要)

jp_overhead.py と同じフォルダに置いてください。追加インストールは不要です。
画面で銘柄コード(6981 285A)や社名の一部(村田 キオクシア)を入力して検索すると、
添付画像と同じ3パネル(株価1年 / 残存ロング分布 / 上値の重さ)+判定コメントを表示します。
"""
import argparse, html, os, re, sys, threading, time, unicodedata, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jp_overhead as jo

C = jo.C
ST = dict(db=None, db_err=None, loaded=False, demo=False, pdf=None, out=None)
DB_LOCK = threading.Lock()
PLOT_LOCK = threading.Lock()          # matplotlib はスレッドセーフではないため直列化
PRICE_CACHE = {}                       # code -> (時刻, df, info)
PRICE_TTL = 1800
CODE_RE = re.compile(r"\d{3}[0-9A-Z]")


# ------------------------------------------------------------------ データ
def ensure_db():
    with DB_LOCK:
        if ST["loaded"] or ST["demo"]:
            return
        try:
            print("信用残PDFを取得・解析中(初回のみ1〜2分かかります)...", flush=True)
            pdf = ST["pdf"] or jo.fetch_latest_pdf(os.path.join(ST["out"], "cache"))
            ST["db"] = jo.parse_margin_pdf(pdf)
            ST["db_err"] = None
            print(f"信用残: {pdf} ({len(ST['db'])}銘柄)", flush=True)
        except Exception as e:
            ST["db"], ST["db_err"] = None, str(e)
            print(f"[警告] 信用残を取得できません: {e}", file=sys.stderr, flush=True)
        ST["loaded"] = True


def resolve(q):
    """入力文字列 → (銘柄コード一覧, 注記一覧)"""
    codes, notes = [], []
    for tok in re.split(r"[\s,、，;；]+", q.strip()):
        if not tok:
            continue
        norm = unicodedata.normalize("NFKC", tok).upper()
        if CODE_RE.fullmatch(norm):
            codes.append(norm)
            continue
        db = ST["db"]
        if db is None:
            notes.append(f"「{tok}」: 社名検索には信用残データが必要です(コードで入力してください)")
            continue
        hits = jo.search_names(db, tok)
        if not hits:
            notes.append(f"「{tok}」に一致する銘柄が見つかりません")
            continue
        names = {c[:-1] if c.endswith("0") else c: n for c, n in zip(db["code"], db["name"])}
        shown = "、".join(f"{names.get(h, '')}({h})" for h in hits[:5])
        more = f" ほか{len(hits) - 5}件" if len(hits) > 5 else ""
        notes.append(f"「{tok}」→ {shown}{more}")
        codes += hits[:5]
    return list(dict.fromkeys(codes)), notes


def get_prices(code):
    now = time.time()
    hit = PRICE_CACHE.get(code)
    if hit and now - hit[0] < PRICE_TTL:
        return hit[1], hit[2]
    df, info = jo.load_prices(code)
    PRICE_CACHE[code] = (now, df, info)
    return df, info


def run_search(codes, scope, floats):
    results, errors = [], []
    for c in codes:
        try:
            if ST["demo"]:
                df, info, mg = jo.demo_data(c)
            else:
                df, info = get_prices(c)
                mg = jo.margin_info(ST["db"], c, scope)
            with PLOT_LOCK:
                r = jo.analyze(c, df, info, mg, floats.get(c))
                jo.draw(r, os.path.join(ST["out"], f"{c}.png"), comments=False)
            results.append(r)
        except Exception as e:
            errors.append(f"{c}: {e}")
    return results, errors


def parse_floats(s):
    out = {}
    for part in re.split(r"[\s,、]+", s.strip()):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[unicodedata.normalize("NFKC", k).upper()] = float(v.replace(",", ""))
            except ValueError:
                pass
    return out


# ------------------------------------------------------------------ HTML
CSS = f"""
*{{box-sizing:border-box}}
body{{margin:0;background:{C['bg']};color:{C['text']};font-family:'Yu Gothic','Meiryo','Noto Sans JP',sans-serif}}
header{{padding:18px 28px;border-bottom:1px solid {C['axis']};background:{C['panel']};position:sticky;top:0;z-index:5}}
h1{{font-size:18px;margin:0 0 10px}}
form{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
input[type=text]{{background:{C['bg']};border:1px solid {C['axis']};color:{C['text']};padding:9px 12px;border-radius:6px;font-size:15px}}
#q{{flex:1;min-width:280px}} #fl{{width:230px}}
select{{background:{C['bg']};border:1px solid {C['axis']};color:{C['text']};padding:9px;border-radius:6px}}
button{{background:{C['cur']};color:#111;border:0;padding:10px 26px;border-radius:6px;font-weight:bold;font-size:15px;cursor:pointer}}
main{{padding:20px 28px;max-width:1500px;margin:auto}}
.hint,.note{{color:{C['sub']};font-size:13px;margin:6px 0}}
.err{{color:{C['bad']};font-size:13px;margin:6px 0}}
table{{border-collapse:collapse;width:100%;margin:14px 0 30px;font-size:13px}}
th,td{{border-bottom:1px solid {C['axis']};padding:7px 10px;text-align:right;white-space:nowrap}}
th{{color:{C['sub']};font-weight:normal}} td.l,th.l{{text-align:left}}
tr:hover td{{background:{C['panel']}}}
a{{color:{C['price']}}}
.card{{background:{C['panel']};border:1px solid {C['axis']};border-radius:8px;padding:14px;margin-bottom:26px}}
.card img{{width:100%;height:auto;display:block;border-radius:4px}}
.card h2{{font-size:16px;margin:0 0 10px;display:flex;justify-content:space-between}}
.badge{{font-weight:bold}} ul{{margin:10px 0 0;padding-left:20px;font-size:13px;line-height:1.7}}
.range{{color:{C['cur']}}}
#busy{{display:none;position:fixed;inset:0;background:rgba(11,14,20,.85);z-index:20;align-items:center;justify-content:center;font-size:18px;text-align:center;line-height:1.8}}
"""


def fmt(v, f="{:,.0f}", none="-"):
    return none if v is None else f.format(v)


def page(q="", scope="total", fl="", results=None, errors=None, notes=None, db_err=None):
    e = html.escape
    sel = lambda v: "selected" if scope == v else ""
    h = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>上値の重さ 検索</title><style>{CSS}</style></head><body>",
         "<div id='busy'><div>検索中です…<br><small>株価を取得して計算しています。<br>初回は信用残PDFの取得で1〜2分かかります。</small></div></div>",
         "<header><h1>日本株 上値の重さ 検索"
         + (" <small style='color:%s'>(デモデータ)</small>" % C['warn'] if ST["demo"] else "") + "</h1>",
         "<form method='get' onsubmit=\"document.getElementById('busy').style.display='flex'\">",
         f"<input type='text' id='q' name='q' value='{e(q)}' placeholder='銘柄コード or 社名の一部(例: 6981 6976 285A / 村田 キオクシア)' autofocus>",
         f"<select name='scope'><option value='total' {sel('total')}>信用: 一般+制度</option>"
         f"<option value='std' {sel('std')}>信用: 制度のみ</option></select>",
         f"<input type='text' id='fl' name='fl' value='{e(fl)}' placeholder='浮動株数 任意 例 6981=23000000'>",
         "<button type='submit'>検索</button></form></header><main>"]
    if db_err:
        h.append(f"<p class='err'>信用残を取得できませんでした(平均取得単価・信用の判定なしで表示): {e(db_err)}</p>")
    for n in notes or []:
        h.append(f"<p class='note'>{e(n)}</p>")
    for n in errors or []:
        h.append(f"<p class='err'>[エラー] {e(n)}</p>")

    if results is None:
        h.append("<p class='hint'>銘柄コード(4桁。285Aのような英字入りも可)または社名の一部を入力して検索してください。"
                 "複数はスペース区切りです。<br>例: <a href='?q=6981+6976+285A'>6981 6976 285A</a> ／ "
                 "<a href='?q=村田+太陽誘電'>村田 太陽誘電</a></p>")
    elif results:
        rows = sorted(results, key=lambda r: -r["above"])
        h.append("<table><tr><th class='l'>コード</th><th class='l'>銘柄</th><th>現在値</th><th>平均取得単価</th>"
                 "<th>乖離</th><th>信用/浮動株</th><th>この上の玉</th><th>含み益の玉</th><th>半減期</th><th class='l'>判定</th></tr>")
        for r in rows:
            gap = (r["avg"] / r["cur"] - 1) * 100 if r["avg"] else None
            h.append(
                f"<tr><td class='l'><a href='#c{e(r['code'])}'>{e(r['code'])}</a></td><td class='l'>{e(r['name'])}</td>"
                f"<td>{r['cur']:,.0f}</td><td>{fmt(r['avg'])}</td><td>{fmt(gap, '{:+.1f}%')}</td>"
                f"<td>{fmt(r['margin_pct'], '{:.1f}%')}</td><td>{r['above']:.0f}%</td><td>{r['in_profit']:.0f}%</td>"
                f"<td>{r['half']:.0f}日</td><td class='l'><b style='color:{r['rating_color']}'>{e(r['rating'])}</b></td></tr>")
        h.append("</table>")
        ts = int(time.time())
        for r in rows:
            h.append(f"<div class='card' id='c{e(r['code'])}'><h2><span>{e(r['code'])} {e(r['name'])}</span>"
                     f"<span class='badge' style='color:{r['rating_color']}'>判定: {e(r['rating'])} (スコア {r['score']})</span></h2>"
                     f"<img src='/img/{e(r['code'])}.png?t={ts}' alt='{e(r['code'])}'><ul>")
            for n in r["notes"]:
                cls = " class='range'" if "【直近" in n else ""
                h.append(f"<li{cls}>{e(n)}</li>")
            h.append("</ul></div>")
    h.append("</main></body></html>")
    return "\n".join(h)


# ------------------------------------------------------------------ サーバー
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="text/html; charset=utf-8", code=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        m = re.fullmatch(r"/img/([0-9A-Z]{4})\.png", u.path)
        if m:
            p = os.path.join(ST["out"], m.group(1) + ".png")
            if os.path.exists(p):
                return self.send(open(p, "rb").read(), "image/png")
            return self.send("not found", "text/plain", 404)
        if u.path != "/":
            return self.send("not found", "text/plain", 404)
        qs = parse_qs(u.query)
        q = (qs.get("q", [""])[0]).strip()
        scope = qs.get("scope", ["total"])[0]
        scope = scope if scope in ("total", "std") else "total"
        fl = qs.get("fl", [""])[0]
        if not q:
            return self.send(page(scope=scope))
        try:
            ensure_db()
            codes, notes = resolve(q)
            if not codes:
                return self.send(page(q, scope, fl, [], [], notes, ST["db_err"]))
            results, errors = run_search(codes, scope, parse_floats(fl))
            self.send(page(q, scope, fl, results, errors, notes, ST["db_err"]))
        except Exception as e:
            self.send(page(q, scope, fl, [], [f"{type(e).__name__}: {e}"]), code=500)


def main():
    ap = argparse.ArgumentParser(description="上値の重さ 検索画面")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pdf", help="JPX信用残PDFを手動指定")
    ap.add_argument("--out", default="result_app")
    ap.add_argument("--demo", action="store_true", help="合成データで動作確認")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    ST.update(demo=a.demo, pdf=a.pdf, out=a.out)
    os.makedirs(a.out, exist_ok=True)
    jo.setup_font()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/"
    print(f"検索画面: {url}   (終了は Ctrl+C)", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("終了します")


if __name__ == "__main__":
    main()
