#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jp_overhead.py  -  日本株 上値の重さスクリーナー
=================================================
銘柄コード(または社名の一部)を渡すと、銘柄ごとに
  ・株価1年チャート(+平均取得単価 / 現在値 / 上値抵抗帯 1〜3)
  ・残存ロング分布(現物 + 信用買い)
  ・上値の重さ(この価格より上に残る玉の%)
  ・判定(上値重い / やや重い / 軽め + 理由コメント)
を1枚の画像にし、一覧(index.html / results.csv)にまとめます。

信用買い残の平均取得単価 = JPX「銘柄別信用取引残高」の 買残金額 ÷ 買残株数
(ヤーマンさん動画の手法。JPXのPDFを自動取得して解析します)

使い方
  pip install yfinance pdfplumber matplotlib numpy pandas scipy
  python jp_overhead.py 6981 6976 285A
  python jp_overhead.py --search 村田 --search キオクシア
  python jp_overhead.py --list tickers.txt --out result
  python jp_overhead.py 285A --scope std      # 制度信用のみで平均取得単価を計算
  python jp_overhead.py 6981 --float 6981=23000000   # 浮動株数を手動指定
  python jp_overhead.py 6981 6976 --demo      # ネット無しの動作確認(合成データ)
"""
import argparse, base64, datetime as dt, html, math, os, re, sys, unicodedata
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ---------------------------------------------------------------- 配色(添付画像に合わせる)
C = dict(
    bg="#0b0e14", panel="#0e121a", grid="#1b2130", axis="#2a3140",
    text="#c9d1d9", sub="#8b949e",
    price="#8ab4f8", spot="#4a9eed", margin="#e85a5a",
    resist="#a33a3a", cur="#f5c542", avg="#5fd068", curve="#ffa940",
    good="#5fd068", warn="#f5c542", bad="#e85a5a",
)

# ---------------------------------------------------------------- 判定しきい値(自由に調整してください)
TH_ABOVE_HEAVY = 40      # この価格より上の玉が40%以上 → +2点
TH_ABOVE_MID = 25        # 25%以上 → +1点
TH_MARGIN_HEAVY = 30     # 信用買い/浮動株 30%以上 → +2点
TH_MARGIN_MID = 15       # 15%以上 → +1点
RANGE_DAYS = 60          # 「直近レンジ」の日数(営業日)
NEAR_TOP = 0.95          # 平均取得単価が直近レンジ高値の95%以上 → 上限付近

VERSION = 4   # 画面アプリ側が「jp_overhead.pyが古い」を検知するための番号
JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/margin/01.html"
UA = {"User-Agent": "Mozilla/5.0 (jp_overhead)"}


def setup_font():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    cands = ["Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic", "Hiragino Sans", "IPAexGothic",
             "IPAGothic", "TakaoGothic", "Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK SC",
             "Noto Sans CJK KR"]
    have = {f.name for f in fm.fontManager.ttflist}
    use = [c for c in cands if c in have]
    plt.rcParams["font.family"] = use + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ================================================================= JPX 信用残
def fetch_latest_pdf(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    page = urlopen(Request(JPX_PAGE, headers=UA), timeout=30).read().decode("utf-8", "ignore")
    links = re.findall(r'href="([^"]*?(\d{8})_mtall\.pdf)"', page)
    if not links:
        raise RuntimeError("JPXページからPDFリンクを取得できませんでした")
    href, d = max(links, key=lambda x: x[1])
    path = os.path.join(cache_dir, f"{d}_mtall.pdf")
    if not os.path.exists(path):
        data = urlopen(Request(urljoin(JPX_PAGE, href), headers=UA), timeout=120).read()
        open(path, "wb").write(data)
    return path


_ROW = re.compile(r"([0-9A-Z]{5})\s*([A-Z]{2}[A-Z0-9]{10})\s*(株数|金額)\s*(?:Shs\.|Val\.)?(.*)")
_NAME_SPLIT = re.compile(r"普通株式|受益証券|投資証券|投信等|優先出資証券")


def _pdf_lines(pdf_path, engine):
    if engine == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                yield from (page.extract_text() or "").splitlines()
    else:
        from pypdf import PdfReader
        for page in PdfReader(pdf_path).pages:
            yield from (page.extract_text() or "").splitlines()


def _parse_lines(lines):
    rec, date, sample = {}, None, []
    for line in lines:
        if len(sample) < 80:
            sample.append(line)
        if date is None:
            m = re.search(r"(\d{4})/(\d{2})/(\d{2})\s*申込み現在", line)
            if m:
                date = "-".join(m.groups())
        m = _ROW.search(line)
        if not m:
            continue
        code, _isin, kind, rest = m.groups()
        rest = re.sub(r"\d+(?:\.\d+)?%", " ", rest)          # 上場比(0.1%)を除く
        nums = [int(t.replace(",", "")) for t in re.findall(r"\d[\d,]*", rest)]
        if len(nums) < 12:
            continue
        r = rec.setdefault(code, {"code": code})
        if kind == "株数":
            head = re.sub(r"^[A-Z]\s+", "", line[:m.start()])
            r["name"] = _NAME_SPLIT.split(head)[0].strip(" 　")
            r.update(sell_sh=nums[0], buy_sh=nums[2], neg_buy_sh=nums[8], std_buy_sh=nums[10])
        else:
            r.update(buy_amt=nums[2], neg_buy_amt=nums[8], std_buy_amt=nums[10])
    return rec, date, sample


def parse_margin_pdf(pdf_path):
    """JPX銘柄別信用取引残高PDF → DataFrame(1銘柄1行)"""
    cache_csv = pdf_path.replace(".pdf", ".csv")
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv, dtype={"code": str})
    rec, date, sample = {}, None, []
    for engine in ("pdfplumber", "pypdf"):
        try:
            rec, date, sample = _parse_lines(_pdf_lines(pdf_path, engine))
        except ImportError:
            continue
        if any("buy_amt" in r and "buy_sh" in r for r in rec.values()):
            break
    df = pd.DataFrame(list(rec.values()))
    if df.empty or "buy_sh" not in df or "buy_amt" not in df:
        dbg = os.path.join(os.path.dirname(pdf_path), "debug_pdf_text.txt")
        open(dbg, "w", encoding="utf-8").write("\n".join(sample))
        raise RuntimeError(f"PDFから銘柄行を読み取れませんでした。{dbg} を開発者(Claude)に貼ってください")
    df["date"] = date
    df = df.dropna(subset=["buy_sh", "buy_amt"]).reset_index(drop=True)
    df.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    return df


def to_jpx_code(c):
    c = unicodedata.normalize("NFKC", str(c)).upper().strip()
    return c + "0" if len(c) == 4 else c


def margin_info(db, code, scope):
    if db is None:
        return None
    row = db[db["code"] == to_jpx_code(code)]
    if row.empty:
        return None
    r = row.iloc[0]
    sh, amt = (r.std_buy_sh, r.std_buy_amt) if scope == "std" else (r.buy_sh, r.buy_amt)
    if not sh or sh <= 0:
        return None
    return dict(shares=float(sh), amount=float(amt), avg=float(amt) / float(sh),
                name_jp=r["name"], date=r["date"])


def search_names(db, word):
    w = unicodedata.normalize("NFKC", word)
    hit = db[db["name"].map(lambda s: w in unicodedata.normalize("NFKC", str(s)))]
    return [c[:-1] if c.endswith("0") else c for c in hit["code"]]


# ================================================================= 株価
def load_prices(code):
    import yfinance as yf
    t = yf.Ticker(f"{unicodedata.normalize('NFKC', code).upper()}.T")
    df = t.history(period="2y", auto_adjust=False)   # Closeは分割調整済み
    if df.empty:
        raise RuntimeError(f"{code}: 株価を取得できませんでした")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    info = {}
    try:
        gi = t.get_info()
        info = dict(name=gi.get("shortName") or gi.get("longName"),
                    issued=gi.get("sharesOutstanding"), float=gi.get("floatShares"))
    except Exception:
        pass
    splits = []
    try:
        s = t.splits
        s.index = pd.to_datetime(s.index).tz_localize(None)
        splits = [(d, r) for d, r in s.items() if d >= df.index[-1] - pd.Timedelta(days=370)]
    except Exception:
        pass
    info["splits"] = splits
    return df, info


def demo_data(code):
    rng = np.random.default_rng(int(re.sub(r"\D", "", code) or 1))
    n = 420
    idx = pd.bdate_range(end=pd.Timestamp("2026-10-15"), periods=n)
    base = 5000 + rng.random() * 3000
    x = np.cumsum(rng.normal(0, 0.022, n)) + np.sin(np.linspace(0, 5, n)) * 0.6
    close = base * np.exp(x - x[0]) * np.linspace(1, 1.3, n)
    high = close * (1 + np.abs(rng.normal(0, 0.012, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.012, n)))
    vol = rng.lognormal(13.5, 0.5, n)
    df = pd.DataFrame(dict(Open=close, High=high, Low=low, Close=close, Volume=vol), index=idx)
    issued = 125e6
    info = dict(name=f"Demo Corp {code}", issued=issued, float=issued * 0.8, splits=[(idx[-150], 4.0)])
    p = close[-1]
    m = dict(shares=issued * 0.8 * 0.18, avg=p * (1.02 + rng.random() * 0.1), name_jp="デモ", date="2026-10-13")
    m["amount"] = m["shares"] * m["avg"]
    return df, info, m


# ================================================================= モデル
def spread(grid, lo, hi, c):
    if hi <= lo:
        w = np.zeros_like(grid)
        w[np.argmin(np.abs(grid - c))] = 1
        return w
    w = np.where(grid <= c, (grid - lo) / max(c - lo, 1e-9), (hi - grid) / max(hi - c, 1e-9))
    w = np.clip(w, 0, None)
    w[(grid < lo) | (grid > hi)] = 0
    if w.sum() == 0:
        w[np.argmin(np.abs(grid - c))] = 1
    return w / w.sum()


def remaining_profile(df, grid, F, slow=1.0):
    """日々の出来高(約定)のうち、その後の売買で入れ替わらずに残っている分を価格帯別に積み上げる。
    残存率 = Π(1 - 日次回転率)  (回転率 = 出来高 / 浮動株)。slow<1で回転が遅い(=長く持つ)玉。"""
    v = df["Volume"].values.astype(float)
    turn = np.clip(v / F * slow, 0, 0.95)
    surv = np.ones(len(v))
    for s in range(len(v) - 2, -1, -1):
        surv[s] = surv[s + 1] * (1 - turn[s + 1])
    w = v * surv
    prof = np.zeros_like(grid)
    for i, (lo, hi, c) in enumerate(zip(df["Low"].values, df["High"].values, df["Close"].values)):
        prof += w[i] * spread(grid, lo, hi, c)
    return prof, w.sum() / F


def tilt_to_mean(grid, q, target):
    z = (grid - target) / target

    def mean(l):
        a = l * z
        a = a - a.max()
        w = q * np.exp(a)
        return (grid * w).sum() / w.sum(), w

    lo, hi = -300.0, 300.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mean(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    m, w = mean((lo + hi) / 2)
    return w, abs(m - target) / target < 0.01


def analyze(code, df, info, mg, float_override=None):
    df = df.tail(255).copy()                       # 約1年
    cur = float(df["Close"].iloc[-1])
    issued = info.get("issued") or None
    F = float_override or info.get("float") or issued
    if not F:
        F = float(df["Volume"].tail(250).sum()) / 5
    issued = issued or F
    grid = np.linspace(df["Low"].min() * 0.97, df["High"].max() * 1.03, 180)
    bw = grid[1] - grid[0]

    tot, cover = remaining_profile(df, grid, F, 1.0)
    tot = tot / tot.sum() * min(F, tot.sum())
    marg = np.zeros_like(grid)
    tilt_ok = None
    if mg:
        mp, _ = remaining_profile(df, grid, F, 0.25)
        if mg["avg"] > grid.min() and mg["avg"] < grid.max():
            mp, tilt_ok = tilt_to_mean(grid, mp, mg["avg"])
        mp = mp / mp.sum() * mg["shares"]
        marg = np.minimum(mp, tot)
    spot = np.clip(tot - marg, 0, None)
    mass = spot + marg
    cs = mass.sum()

    def pct_above(p):
        return 100 * mass[grid > p].sum() / cs

    above = pct_above(cur)
    pcts = np.array([pct_above(g) for g in grid])

    # 上値抵抗帯 1〜3(現在値より上の山を質量順に)
    sm = gaussian_filter1d(mass, 2)
    pk, _ = find_peaks(sm)
    pk = [i for i in pk if grid[i] > cur * 1.01]
    pk = sorted(pk, key=lambda i: -sm[i])
    res = []
    for i in pk:
        if all(abs(grid[i] - grid[j]) / grid[j] > 0.05 for j in res):
            res.append(i)
        if len(res) == 3:
            break
    res_prices = [float(grid[i]) for i in res]

    turn = (df["Volume"] / F).tail(20).median()
    half = math.log(2) / -math.log(1 - min(turn, 0.9)) if turn > 0 else float("inf")

    close = df["Close"]
    pk_i = int(np.argmax(close.values))
    peak = float(close.iloc[pk_i])
    trough = float(close.iloc[pk_i:].min())

    res_d = dict(code=code, name=info.get("name") or (mg or {}).get("name_jp") or "", cur=cur,
                 avg=mg["avg"] if mg else None, margin_sh=mg["shares"] if mg else None,
                 margin_date=mg["date"] if mg else None,
                 float=F, issued=issued, df=df, grid=grid, bw=bw, spot=spot, marg=marg,
                 pcts=pcts, above=above, in_profit=100 - above, half=half, cover=cover,
                 margin_pct=(100 * mg["shares"] / F) if mg else None,
                 res=res_prices, peak=peak, trough=trough, tilt_ok=tilt_ok,
                 splits=info.get("splits", []))
    res_d.update(judge(res_d))
    return res_d


def judge(r):
    df, cur, avg = r["df"], r["cur"], r["avg"]
    score, notes = 0, []
    a = r["above"]
    if a >= TH_ABOVE_HEAVY:
        score += 2
        notes.append(f"現在値より上に残る玉が{a:.0f}%と多い(戻り売り圧力)")
    elif a >= TH_ABOVE_MID:
        score += 1
        notes.append(f"現在値より上に残る玉は{a:.0f}%(やや多い)")
    else:
        notes.append(f"現在値より上の玉は{a:.0f}%と少なく、上値は軽め")
    mp = r["margin_pct"]
    if mp is not None:
        if mp >= TH_MARGIN_HEAVY:
            score += 2
            notes.append(f"信用買い残が浮動株の{mp:.1f}%と大きい")
        elif mp >= TH_MARGIN_MID:
            score += 1
            notes.append(f"信用買い残は浮動株の{mp:.1f}%")
    if avg:
        gap = (avg / cur - 1) * 100
        if gap > 3:
            score += 2
            notes.append(f"信用買いの平均取得単価{avg:,.0f}円は現在値を{gap:.1f}%上回る(含み損・やれやれ売り待ち)")
        elif gap > 0:
            score += 1
            notes.append(f"平均取得単価{avg:,.0f}円は現在値の直上({gap:+.1f}%)")
        else:
            notes.append(f"平均取得単価{avg:,.0f}円は現在値を下回る({gap:+.1f}%・含み益)")
        # ① 直近レンジとの位置関係
        rg = df.tail(RANGE_DAYS)
        hi, lo = float(rg["High"].max()), float(rg["Low"].min())
        pos = (avg - lo) / (hi - lo) if hi > lo else 0.5
        if avg >= hi * NEAR_TOP:
            score += 1
            where = "上回って" if avg > hi else "付近に"
            notes.append(f"【直近{RANGE_DAYS}日レンジ】平均取得単価は上限{hi:,.0f}円{where}あり、レンジ上限が戻りの壁になりやすい")
        elif pos >= 0.66:
            notes.append(f"【直近{RANGE_DAYS}日レンジ】平均取得単価はレンジ上寄り({pos*100:.0f}%位置)")
        elif pos <= 0.33:
            notes.append(f"【直近{RANGE_DAYS}日レンジ】平均取得単価はレンジ下寄り({pos*100:.0f}%位置)")
        else:
            notes.append(f"【直近{RANGE_DAYS}日レンジ】平均取得単価はレンジ中位({pos*100:.0f}%位置)")
        if r["splits"] and avg > cur * 1.05:
            d, ratio = r["splits"][-1]
            notes.append(f"{d:%Y-%m-%d}に分割(×{ratio:g})あり。分割前の高値約定が平均を押し上げている可能性")
    if not avg:
        notes.append("信用残データなし: 平均取得単価・信用の判定は含まれません(参考値)")
    rating = "上値重い" if score >= 5 else "やや重い" if score >= 3 else "上値軽め"
    color = C["bad"] if score >= 5 else C["warn"] if score >= 3 else C["good"]
    return dict(score=score, rating=rating, rating_color=color, notes=notes)


# ================================================================= 描画
def fmt_m(x):
    return f"{x/1e6:,.0f}"


def draw(r, path, comments=True):
    df, grid, bw, cur, avg = r["df"], r["grid"], r["bw"], r["cur"], r["avg"]
    fig = plt.figure(figsize=(15, 6.6 if comments else 5.3), facecolor=C["bg"])
    gs = GridSpec(1, 3, width_ratios=[3.6, 1.25, 1.25], wspace=0.03,
                  left=0.06, right=0.985, top=0.80 if comments else 0.78, bottom=0.27 if comments else 0.11)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharey=a1)
    a3 = fig.add_subplot(gs[2], sharey=a1)
    for a in (a1, a2, a3):
        a.set_facecolor(C["panel"])
        a.grid(color=C["grid"], lw=0.6)
        a.tick_params(colors=C["sub"], labelsize=8)
        for s in a.spines.values():
            s.set_color(C["axis"])
    ylo, yhi = grid.min(), grid.max()

    # --- 株価
    a1.plot(df.index, df["Close"], color=C["price"], lw=1.3)
    for i, p in enumerate(r["res"], 1):
        a1.axhline(p, color=C["resist"], ls="--", lw=0.9, alpha=0.9)
        a1.text(1.004, p, f"{i}.", transform=a1.get_yaxis_transform(), color=C["bad"],
                fontsize=8, va="center", clip_on=False)
    a1.axhline(cur, color=C["cur"], ls="--", lw=1.1)
    labs = [(cur, f"現在値 {cur:,.0f}", C["cur"])]
    if avg:
        a1.axhline(avg, color=C["avg"], ls=":", lw=1.1)
        labs.append((avg, f"平均取得単価 {avg:,.0f}", C["avg"]))
    labs.sort(key=lambda x: x[0], reverse=True)
    span = yhi - ylo
    for k, (p, t, col) in enumerate(labs):
        up = (k == 0) if len(labs) > 1 else True
        a1.text(0.012, p + (span * 0.008 if up else -span * 0.008), t, transform=a1.get_yaxis_transform(),
                color=col, fontsize=8.5, va="bottom" if up else "top")
    a1.set_ylim(ylo, yhi)
    a1.set_xlim(df.index[0], df.index[-1])
    a1.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
    a1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- 残存ロング分布
    a2.barh(grid, r["spot"] / 1e6, height=bw, color=C["spot"], label="現物", lw=0)
    a2.barh(grid, r["marg"] / 1e6, height=bw, left=r["spot"] / 1e6, color=C["margin"], label="信用買い", lw=0)
    a2.axhline(cur, color=C["cur"], ls="--", lw=1.1)
    if avg:
        a2.axhline(avg, color=C["avg"], ls=":", lw=1.0)
    a2.set_title("残存ロング分布", color=C["text"], fontsize=11, loc="left", pad=8)
    a2.set_xlabel("百万株", color=C["sub"], fontsize=8)
    plt.setp(a2.get_yticklabels(), visible=False)
    leg = a2.legend(loc="upper right", fontsize=8, facecolor=C["panel"], edgecolor=C["axis"])
    for t in leg.get_texts():
        t.set_color(C["text"])

    # --- 上値の重さ
    a3.plot(r["pcts"], grid, color=C["curve"], lw=2)
    a3.axhline(cur, color=C["cur"], ls="--", lw=1.1)
    a3.scatter([r["above"]], [cur], color=C["cur"], s=50, zorder=5)
    right = r["above"] > 60
    low_edge = (cur - ylo) < span * 0.12
    a3.annotate(f"{r['above']:.0f}%が\nこの上", (r["above"], cur),
                xytext=(r["above"] + (-4 if right else 4), cur + (span * 0.05 if low_edge else -span * 0.05)),
                color=C["cur"], fontsize=9, ha="right" if right else "left",
                va="bottom" if low_edge else "top")
    a3.set_xlim(0, 100)
    a3.set_title("上値の重さ", color=C["text"], fontsize=11, loc="left", pad=8)
    a3.set_xlabel("この価格より上の玉(%)", color=C["sub"], fontsize=8)
    plt.setp(a3.get_yticklabels(), visible=False)
    lines = [f"含み益の玉 {r['in_profit']:.0f}%", f"半減期 {r['half']:.0f}日"]
    if r["margin_pct"] is not None:
        lines.append(f"信用は浮動株の{r['margin_pct']:.1f}%")
    lines.append(f"モデル被覆率 {min(r['cover'],9.99)*100:.0f}%")
    for i, t in enumerate(lines):
        a3.text(0.97, 0.97 - i * 0.052, t, transform=a3.transAxes, ha="right", va="top",
                color=C["sub"], fontsize=8.5)

    # --- ヘッダー
    fig.text(0.06, 0.945, f"{r['code']} {r['name']}", color=C["text"], fontsize=13, fontweight="bold")
    fl_pct = r["float"] / r["issued"] * 100 if r["issued"] else 100
    fig.text(0.06, 0.895,
             f"ピーク {r['peak']:,.0f} → 谷 {r['trough']:,.0f} ({(r['trough']/r['peak']-1)*100:.0f}%)   "
             f"推定浮動株 {fmt_m(r['float'])}百万株(発行済{fmt_m(r['issued'])}百万株の{fl_pct:.0f}%)",
             color=C["sub"], fontsize=9)
    fig.text(0.985, 0.945, f"判定: {r['rating']}  (スコア {r['score']})", color=r["rating_color"],
             fontsize=13, fontweight="bold", ha="right")
    if r["margin_date"]:
        fig.text(0.985, 0.895, f"信用残 {r['margin_date']} 時点", color=C["sub"], fontsize=8.5, ha="right")
    # --- コメント
    for i, t in enumerate(r["notes"][:6] if comments else []):
        fig.text(0.06, 0.155 - i * 0.026, "・" + t, color=C["text"] if "【直近" not in t else C["cur"], fontsize=9)
    fig.savefig(path, dpi=130, facecolor=C["bg"])
    plt.close(fig)


# ================================================================= 一覧
def write_index(results, outdir, margin_date):
    rows = sorted(results, key=lambda r: -r["above"])
    cols = ["code", "name", "cur", "avg", "gap_pct", "margin_pct", "above", "in_profit", "half", "score", "rating"]
    recs = []
    for r in rows:
        recs.append(dict(code=r["code"], name=r["name"], cur=round(r["cur"]),
                         avg=round(r["avg"]) if r["avg"] else None,
                         gap_pct=round((r["avg"] / r["cur"] - 1) * 100, 1) if r["avg"] else None,
                         margin_pct=round(r["margin_pct"], 1) if r["margin_pct"] is not None else None,
                         above=round(r["above"], 1), in_profit=round(r["in_profit"], 1),
                         half=round(r["half"], 1), score=r["score"], rating=r["rating"]))
    pd.DataFrame(recs)[cols].to_csv(os.path.join(outdir, "results.csv"), index=False, encoding="utf-8-sig")

    th = {"code": "コード", "name": "銘柄", "cur": "現在値", "avg": "平均取得単価", "gap_pct": "取得単価乖離%",
          "margin_pct": "信用買い/浮動株%", "above": "この上の玉%", "in_profit": "含み益の玉%",
          "half": "半減期(日)", "score": "スコア", "rating": "判定"}
    h = ["<html><head><meta charset='utf-8'><title>上値の重さ 一覧</title><style>",
         f"body{{background:{C['bg']};color:{C['text']};font-family:sans-serif;margin:24px}}",
         f"table{{border-collapse:collapse;margin-bottom:28px}}td,th{{border-bottom:1px solid {C['axis']};padding:6px 12px;text-align:right;font-size:13px}}",
         f"th{{color:{C['sub']}}}td:nth-child(2),th:nth-child(2){{text-align:left}}img{{max-width:100%;margin-bottom:18px}}a{{color:{C['price']}}}",
         "</style></head><body>", f"<h2>上値の重さ 一覧 <small style='color:{C['sub']}'>(信用残: {margin_date or 'なし'})</small></h2>",
         "<table><tr>" + "".join(f"<th>{th[c]}</th>" for c in cols) + "</tr>"]
    for rec, r in zip(recs, rows):
        cells = []
        for c in cols:
            v = rec[c]
            v = "-" if v is None else (f"{v:,}" if isinstance(v, (int, float)) and c in ("cur", "avg") else v)
            if c == "rating":
                v = f"<b style='color:{r['rating_color']}'>{v}</b>"
            if c == "code":
                v = f"<a href='#c{v}'>{v}</a>"
            cells.append(f"<td>{v}</td>")
        h.append("<tr>" + "".join(cells) + "</tr>")
    h.append("</table>")
    for r in rows:
        h.append(f"<a id='c{r['code']}'></a><img src='{r['code']}.png'>")
    h.append("</body></html>")
    open(os.path.join(outdir, "index.html"), "w", encoding="utf-8").write("\n".join(h))


# ================================================================= main
def main():
    ap = argparse.ArgumentParser(description="日本株 上値の重さスクリーナー")
    ap.add_argument("codes", nargs="*", help="銘柄コード (例: 6981 285A)")
    ap.add_argument("--search", action="append", default=[], help="社名の一部で検索(JPX信用残の銘柄名)")
    ap.add_argument("--list", help="銘柄コードを1行1つ書いたテキスト")
    ap.add_argument("--out", default="result")
    ap.add_argument("--pdf", help="JPX信用残PDFを手動指定(省略時は最新を自動取得)")
    ap.add_argument("--scope", choices=["total", "std"], default="total",
                    help="total=一般+制度(動画と同じ計算) / std=制度信用のみ")
    ap.add_argument("--float", action="append", default=[], help="浮動株数の手動指定 CODE=株数")
    ap.add_argument("--demo", action="store_true", help="合成データで動作確認")
    a = ap.parse_args()

    setup_font()
    os.makedirs(a.out, exist_ok=True)
    codes = list(a.codes)
    if a.list:
        codes += [l.strip() for l in open(a.list, encoding="utf-8") if l.strip() and not l.startswith("#")]
    fo = {unicodedata.normalize("NFKC", k).upper(): float(v.replace(",", ""))
          for k, v in (x.split("=") for x in a.float)}

    db = None
    if not a.demo:
        try:
            pdf = a.pdf or fetch_latest_pdf(os.path.join(a.out, "cache"))
            db = parse_margin_pdf(pdf)
            print(f"信用残: {pdf} ({len(db)}銘柄)")
        except Exception as e:
            print(f"[警告] 信用残を取得できません: {e}\n  → 信用買い/平均取得単価なしで続行します", file=sys.stderr)
    for w in a.search:
        if db is None:
            print("[警告] 名前検索には信用残PDFが必要です", file=sys.stderr)
            continue
        hit = search_names(db, w)
        print(f"検索「{w}」: {hit[:10]}")
        codes += hit[:5]
    codes = list(dict.fromkeys(unicodedata.normalize("NFKC", c).upper() for c in codes))
    bad = [c for c in codes if not re.fullmatch(r"\d{3}[0-9A-Z]", c)]
    if bad:
        print(f"[警告] 銘柄コードとして不正なので除外します: {bad}\n"
              f"  (4桁の数字、または285Aのような形式。pipコマンドは別の行で実行してください)", file=sys.stderr)
        codes = [c for c in codes if c not in bad]
    if not codes:
        ap.error("有効な銘柄コードがありません")

    results, mdate = [], None
    for c in codes:
        try:
            if a.demo:
                df, info, mg = demo_data(c)
            else:
                df, info = load_prices(c)
                mg = margin_info(db, c, a.scope)
            mdate = mdate or (mg or {}).get("date")
            r = analyze(c, df, info, mg, fo.get(c))
            draw(r, os.path.join(a.out, f"{c}.png"))
            results.append(r)
            print(f"{c} {r['name'][:20]:20s} 現在値{r['cur']:>8,.0f} 平均取得{(r['avg'] or 0):>8,.0f} "
                  f"この上{r['above']:4.0f}%  {r['rating']}")
        except Exception as e:
            print(f"[エラー] {c}: {e}", file=sys.stderr)
    if results:
        write_index(results, a.out, mdate)
        print(f"\n出力: {os.path.abspath(a.out)}/index.html , results.csv , <コード>.png")


if __name__ == "__main__":
    main()
