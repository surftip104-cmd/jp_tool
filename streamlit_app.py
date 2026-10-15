# -*- coding: utf-8 -*-
"""
streamlit_app.py - 日本株「上値の重さ」検索 (Streamlit版)

Streamlit Community Cloud の "Main file path" には、このファイル(streamlit_app.py)を指定します。
同じリポジトリに jp_overhead.py / requirements.txt / packages.txt を置いてください。
"""
import os
import re
import tempfile
import threading
import unicodedata

import pandas as pd
import streamlit as st

import jp_overhead as jo

if getattr(jo, "VERSION", 0) < 4:
    st.error("jp_overhead.py が古いバージョンです。最新の jp_overhead.py に置き換えてください(同じフォルダ/リポジトリにある jp_overhead.py を上書き)。")
    st.stop()

st.set_page_config(page_title="上値の重さ 検索", page_icon="📉", layout="wide")

OUT = os.path.join(tempfile.gettempdir(), "jp_overhead_app")
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
CODE_RE = re.compile(r"\d{3}[0-9A-Z]")
RATING_MARK = {"上値重い": "🔴", "やや重い": "🟡", "上値軽め": "🟢"}


# ------------------------------------------------------------------ 補助(画面APIの版差を吸収)
def show_image(img):
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ データ
@st.cache_resource(show_spinner=False)
def plot_lock():
    return threading.Lock()          # matplotlib はスレッドセーフではないので直列化


@st.cache_resource(show_spinner="信用残PDFを取得・解析中です(初回のみ1〜2分かかります)…")
def load_db_auto():
    # 失敗時は例外 → キャッシュされず、次回また試行される
    pdf = jo.fetch_latest_pdf(os.path.join(OUT, "cache"))
    return jo.parse_margin_pdf(pdf)


@st.cache_resource(show_spinner="アップロードされたPDFを解析中です…")
def load_db_uploaded(data: bytes):
    import hashlib
    name = hashlib.md5(data).hexdigest() + "_mtall.pdf"
    path = os.path.join(OUT, "cache", name)
    with open(path, "wb") as f:
        f.write(data)
    return jo.parse_margin_pdf(path)


@st.cache_data(ttl=1800, show_spinner=False)
def get_prices(code):
    return jo.load_prices(code)


def get_db(uploaded):
    try:
        if uploaded is not None:
            return load_db_uploaded(uploaded.getvalue()), None
        return load_db_auto(), None
    except Exception as e:
        return None, str(e)


def resolve(q, db):
    codes, notes = [], []
    for tok in re.split(r"[\s,、，;；]+", q.strip()):
        if not tok:
            continue
        norm = unicodedata.normalize("NFKC", tok).upper()
        if CODE_RE.fullmatch(norm):
            codes.append(norm)
            continue
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


def run_search(codes, db, scope, floats, demo, progress):
    results, errors = [], []
    for i, c in enumerate(codes):
        progress.progress(i / max(len(codes), 1), text=f"{c} を計算中…")
        try:
            if demo:
                df, info, mg = jo.demo_data(c)
            else:
                df, info = get_prices(c)
                mg = jo.margin_info(db, c, scope)
            path = os.path.join(OUT, "img", f"{c}.png")
            with plot_lock():
                r = jo.analyze(c, df, info, mg, floats.get(c))
                jo.draw(r, path, comments=False)
            with open(path, "rb") as f:
                r["png"] = f.read()
            results.append(r)
        except Exception as e:
            errors.append(f"{c}: {e}")
    progress.progress(1.0, text="完了")
    return results, errors


# ------------------------------------------------------------------ 画面
jo.setup_font()
st.title("日本株 上値の重さ 検索")

with st.sidebar:
    st.header("設定")
    scope_label = st.selectbox("信用買いの範囲", ["一般+制度(動画と同じ計算)", "制度信用のみ"])
    scope = "total" if scope_label.startswith("一般") else "std"
    float_txt = st.text_input("浮動株数(任意)", value="", placeholder="6981=23000000")
    uploaded = st.file_uploader("JPX信用残PDF(自動取得できないとき)", type="pdf")
    demo = st.checkbox("デモデータで動作確認(ネット不要)", value=False)
    st.caption("信用残: JPX「銘柄別信用取引残高」/ 株価: Yahoo Finance(yfinance)")

with st.form("search"):
    q = st.text_input("銘柄コード または 社名の一部(スペース区切りで複数)", value="",
                      placeholder="例: 6981 6976 285A / 村田 キオクシア")
    submitted = st.form_submit_button("検索")

if submitted and q.strip():
    db, db_err = (None, None) if demo else get_db(uploaded)
    notes = []
    if db_err:
        notes.append("信用残を取得できませんでした(平均取得単価・信用の判定なしで表示します)。"
                     "サイドバーからJPXのPDFをアップロードすると反映できます。 詳細: " + db_err)
    codes, n2 = resolve(q, db)
    notes += n2
    results, errors = [], []
    if codes:
        results, errors = run_search(codes, db, scope, parse_floats(float_txt), demo, st.progress(0.0))
    st.session_state["res"] = dict(q=q, results=results, errors=errors, notes=notes)

res = st.session_state.get("res")
if not res:
    st.info("銘柄コード(4桁。285Aのような英字入りも可)または社名の一部を入力して「検索」を押してください。")
else:
    for n in res["notes"]:
        st.caption(n)
    for e in res["errors"]:
        st.error(e)
    rows = sorted(res["results"], key=lambda r: -r["above"])
    if rows:
        table = pd.DataFrame([{
            "コード": r["code"], "銘柄": r["name"],
            "現在値": round(r["cur"]),
            "平均取得単価": round(r["avg"]) if r["avg"] else None,
            "乖離%": round((r["avg"] / r["cur"] - 1) * 100, 1) if r["avg"] else None,
            "信用/浮動株%": round(r["margin_pct"], 1) if r["margin_pct"] is not None else None,
            "この上の玉%": round(r["above"]),
            "含み益の玉%": round(r["in_profit"]),
            "半減期(日)": round(r["half"]),
            "判定": f"{RATING_MARK.get(r['rating'], '')} {r['rating']}",
        } for r in rows])
        show_table(table)
        for r in rows:
            st.subheader(f"{r['code']} {r['name']}  —  判定: {RATING_MARK.get(r['rating'], '')} {r['rating']}(スコア {r['score']})")
            show_image(r["png"])
            st.markdown("\n".join("- " + n for n in r["notes"]))
