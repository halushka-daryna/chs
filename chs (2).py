"""
Лабораторна робота "Автоматичне укладання частотних словників (ЧС)".

Пайплайн:
  1. Токенізація (tokenize_uk) -> слововживання (тільки слова, кирилиця).
  2. Лематизація + частиномовні теги (pymorphy2, словник uk).
  3. Вибірка = перші N слововживань (за замовчуванням 20 000),
     поділ на 20 підвибірок по 1000 слововживань.
  4. Базова таблиця tokens (кожне слововживання - окремий рядок, повтори НЕ ігноруються).
  5. Три БД, що будуються агрегуючими SQL-запитами (SELECT ... COUNT/SUM ... GROUP BY):
       freq_wordform  - БД_ЧС_словоформ
       freq_lemma     - БД_ЧС_лем
       freq_pos       - БД_ЧС_ЧМ
  6. Додаткова статистика: TF-IDF по лемах, TTR, гапакси, середня довжина слова,
     таблиця відносної похибки, однорідність підвибірок.

Запуск:
    python chs.py --texts text1.txt text2.txt --names khvylovyi andrukhovych
"""
import argparse
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# ---------- необов'язкові залежності ----------
try:
    import tokenize_uk
except ImportError:
    tokenize_uk = None
try:
    import pymorphy3 as pymorphy2
except ImportError:
    try:
        import pymorphy2
    except ImportError:
        pymorphy2 = None

N_PARTS = 20
PART_SIZE = 1000

CYR_WORD = re.compile(r"[А-Яа-яІіЇїЄєҐґ]+(?:['’ʼ\-][А-Яа-яІіЇїЄєҐґ]+)*")
HAS_CYR = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")

# теги pymorphy2 (uk) -> позначення з прикладу БД_ЧС_ЧМ
POS_MAP = {
    "NOUN": "noun", "VERB": "verb", "INFN": "verb", "ADJF": "adj", "ADJS": "adj",
    "ADVB": "adv", "CONJ": "conj", "PREP": "prep", "PRCL": "part", "INTJ": "intj",
    "NUMR": "numr", "GRND": "advp", "PRTF": "adj", "PRTS": "adj",
    "NPRO": "npro", "PRED": "pred", "COMP": "comp",
}
POS_UA = {
    "noun": "Іменник", "verb": "Дієслово", "adj": "Прикметник", "adv": "Прислівник",
    "conj": "Сполучник", "prep": "Прийменник", "part": "Частка", "intj": "Вигук",
    "numr": "Числівник", "advp": "Дієприслівник", "npro": "Займенник",
    "pred": "Предикатив", "comp": "Компаратив", "unkn": "Невідоме",
}


# ---------- 1. Токенізація ----------
def tokenize(text: str):
    """Повертає список слововживань (рядки в нижньому регістрі)."""
    if tokenize_uk is not None:
        raw = tokenize_uk.tokenize_words(text)
        words = [t for t in raw if HAS_CYR.search(t) and t[0].isalpha()]
    else:  # запасний варіант, якщо tokenize_uk не встановлено
        words = CYR_WORD.findall(text)
    return [w.lower() for w in words]


def clean_text(text: str) -> str:
    """Прибирає службові виноски в кінці тексту (рядки на кшталт '1 Причина війни')."""
    lines = []
    for ln in text.splitlines():
        if re.match(r"^\s*\d{1,2}\s+\S", ln) and len(ln) < 120 and "(" in ln and ")" in ln:
            continue
        lines.append(ln)
    return "\n".join(lines)


# ---------- 2. Лематизація ----------
class Morph:
    def __init__(self):
        self.available = pymorphy2 is not None
        self._an = pymorphy2.MorphAnalyzer(lang="uk") if self.available else None
        self._cache = {}

    def analyze(self, w: str):
        if w in self._cache:
            return self._cache[w]
        if self._an is None:
            res = (w, "unkn")
        else:
            p = self._an.parse(w)[0]
            pos = POS_MAP.get(p.tag.POS, "unkn") if p.tag.POS else "unkn"
            res = (p.normal_form, pos)
        self._cache[w] = res
        return res


# ---------- 3-5. База даних ----------
class DB:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.cur = self.conn.cursor()

    def init_schema(self):
        part_cols = ",\n".join(f"part_{i} INTEGER NOT NULL DEFAULT 0" for i in range(1, N_PARTS + 1))
        self.cur.executescript(f"""
        DROP TABLE IF EXISTS tokens;
        DROP TABLE IF EXISTS samples;
        DROP TABLE IF EXISTS freq_wordform;
        DROP TABLE IF EXISTS freq_lemma;
        DROP TABLE IF EXISTS freq_pos;
        CREATE TABLE samples (
            sample_id INTEGER PRIMARY KEY, name TEXT, source TEXT, genre TEXT, n_tokens INTEGER);
        CREATE TABLE tokens (
            token_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL REFERENCES samples(sample_id),
            position INTEGER NOT NULL,          -- номер слововживання у вибірці
            part_no INTEGER NOT NULL,           -- номер підвибірки 1..20
            wordform TEXT NOT NULL, lemma TEXT NOT NULL, pos TEXT NOT NULL);
        CREATE INDEX ix_tokens ON tokens(sample_id, wordform);
        CREATE TABLE freq_wordform (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER, wordform TEXT,
            lemma TEXT, pos TEXT, abs_freq INTEGER, {part_cols});
        CREATE TABLE freq_lemma (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER, lemma TEXT,
            abs_freq INTEGER, {part_cols});
        CREATE TABLE freq_pos (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER, pos TEXT, pos_ua TEXT,
            abs_freq INTEGER, {part_cols});
        """)
        self.conn.commit()

    def add_sample(self, sid, name, source, genre, tokens, morph):
        self.cur.execute("INSERT INTO samples VALUES (?,?,?,?,?)", (sid, name, source, genre, len(tokens)))
        rows = []
        for i, w in enumerate(tokens):
            lemma, pos = morph.analyze(w)
            rows.append((sid, i + 1, i // PART_SIZE + 1, w, lemma, pos))
        self.cur.executemany(
            "INSERT INTO tokens(sample_id,position,part_no,wordform,lemma,pos) VALUES (?,?,?,?,?,?)", rows)
        self.conn.commit()

    @staticmethod
    def _part_sums():
        return ",\n".join(f"SUM(CASE WHEN part_no={i} THEN 1 ELSE 0 END)" for i in range(1, N_PARTS + 1))

    def build_frequency_dbs(self):
        """Усі три ЧС отримуємо агрегуючими запитами INSERT ... SELECT ... GROUP BY."""
        cols = ",".join(f"part_{i}" for i in range(1, N_PARTS + 1))
        sums = self._part_sums()
        self.cur.executescript(f"""
        INSERT INTO freq_wordform(sample_id,wordform,lemma,pos,abs_freq,{cols})
        SELECT sample_id, wordform, MIN(lemma), MIN(pos), COUNT(*), {sums}
        FROM tokens GROUP BY sample_id, wordform ORDER BY sample_id, COUNT(*) DESC, wordform;

        INSERT INTO freq_lemma(sample_id,lemma,abs_freq,{cols})
        SELECT sample_id, lemma, COUNT(*), {sums}
        FROM tokens GROUP BY sample_id, lemma ORDER BY sample_id, COUNT(*) DESC, lemma;

        INSERT INTO freq_pos(sample_id,pos,abs_freq,{cols})
        SELECT sample_id, pos, COUNT(*), {sums}
        FROM tokens GROUP BY sample_id, pos ORDER BY sample_id, COUNT(*) DESC;
        """)
        for k, v in POS_UA.items():
            self.cur.execute("UPDATE freq_pos SET pos_ua=? WHERE pos=?", (v, k))
        self.conn.commit()

    # ---------- аналітичні запити ----------
    def q(self, sql, args=()):
        return self.cur.execute(sql, args).fetchall()


# ---------- 6. Додаткова статистика ----------
def basic_stats(db: DB, sid: int):
    n, = db.q("SELECT COUNT(*) FROM tokens WHERE sample_id=?", (sid,))[0]
    types_form, = db.q("SELECT COUNT(DISTINCT wordform) FROM tokens WHERE sample_id=?", (sid,))[0]
    types_lem, = db.q("SELECT COUNT(DISTINCT lemma) FROM tokens WHERE sample_id=?", (sid,))[0]
    hapax_form, = db.q("SELECT COUNT(*) FROM freq_wordform WHERE sample_id=? AND abs_freq=1", (sid,))[0]
    hapax_lem, = db.q("SELECT COUNT(*) FROM freq_lemma WHERE sample_id=? AND abs_freq=1", (sid,))[0]
    avg_len, = db.q("SELECT AVG(LENGTH(wordform)) FROM tokens WHERE sample_id=?", (sid,))[0]
    return {
        "Слововживань (N)": n,
        "Різних словоформ": types_form,
        "Різних лем": types_lem,
        "TTR словоформ": round(types_form / n, 4),
        "TTR лем": round(types_lem / n, 4),
        "Гапаксів (словоформи)": hapax_form,
        "Гапаксів (леми)": hapax_lem,
        "Середня довжина слова": round(avg_len, 2),
    }


def relative_error_table(db: DB, sid: int, z=1.96):
    """Відносна похибка частоти (довірча ймовірність 95%): delta = z*sqrt((1-p)/(N*p))."""
    n, = db.q("SELECT COUNT(*) FROM tokens WHERE sample_id=?", (sid,))[0]
    out = []
    for pos, ua, f in db.q("SELECT pos,pos_ua,abs_freq FROM freq_pos WHERE sample_id=? ORDER BY abs_freq DESC", (sid,)):
        p = f / n
        out.append((pos, ua, f, round(p, 4), round(100 * z * math.sqrt((1 - p) / (n * p)), 2)))
    return out


def homogeneity(db: DB, sid: int):
    """Коефіцієнт варіації (V) частоти кожної частини мови по 20 підвибірках."""
    res = []
    for row in db.q("SELECT pos, " + ",".join(f"part_{i}" for i in range(1, N_PARTS + 1)) +
                    " FROM freq_pos WHERE sample_id=? ORDER BY abs_freq DESC", (sid,)):
        vals = row[1:]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        res.append((row[0], round(m, 1), round(sd, 2), round(100 * sd / m, 1) if m else 0))
    return res


def tfidf_top(db: DB, top=15):
    """TF-IDF по лемах. Документ = підвибірка (всього 2 x 20 документів)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    docs, owner = [], []
    for sid, part in db.q("SELECT DISTINCT sample_id, part_no FROM tokens ORDER BY 1,2"):
        lemmas = [r[0] for r in db.q(
            "SELECT lemma FROM tokens WHERE sample_id=? AND part_no=? ORDER BY position", (sid, part))]
        docs.append(" ".join(lemmas))
        owner.append(sid)
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
    mat = vec.fit_transform(docs)
    names = np.array(vec.get_feature_names_out())
    owner = np.array(owner)
    result = {}
    for sid in sorted(set(owner)):
        mean = np.asarray(mat[owner == sid].mean(axis=0)).ravel()
        idx = mean.argsort()[::-1][:top]
        result[sid] = [(names[i], round(float(mean[i]), 4)) for i in idx]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", nargs="+", default=[
        "texts3/second.txt", "texts3/vynnychenko.txt"])
    ap.add_argument("--names", nargs="+", default=["dziuba", "vynnychenko"])
    ap.add_argument("--genres", nargs="+", default=["публіцистика", "художній"])
    ap.add_argument("--db", default="data/freq3.db")
    ap.add_argument("--report", default="report3.md")
    ap.add_argument("--size", type=int, default=N_PARTS * PART_SIZE)
    a = ap.parse_args()
    genres = a.genres or [""] * len(a.texts)

    if tokenize_uk is None:
        print("УВАГА: tokenize_uk не знайдено, використано regex-токенізатор.", file=sys.stderr)
    morph = Morph()
    if not morph.available:
        print("УВАГА: pymorphy2 не знайдено - леми = словоформи, pos = 'unkn'. "
              "Встановіть: pip install pymorphy2 pymorphy2-dicts-uk tokenize-uk", file=sys.stderr)

    db = DB(a.db)
    db.init_schema()
    for sid, (path, name, genre) in enumerate(zip(a.texts, a.names, genres), start=1):
        text = clean_text(Path(path).read_text(encoding="utf-8"))
        toks = tokenize(text)
        total = len(toks)
        if total < a.size:
            print(f"УВАГА: у {name} лише {total} слововживань (< {a.size}).", file=sys.stderr)
        toks = toks[:a.size]
        db.add_sample(sid, name, path, genre, toks, morph)
        print(f"{name}: всього слів у тексті {total}, у вибірці {len(toks)}")
    db.build_frequency_dbs()

    # ---- звіт ----
    out = ["# Звіт: автоматичне укладання ЧС\n"]
    if not morph.available:
        out.append("> **Лематизація не виконувалась (pymorphy2 не встановлено): "
                   "частиномовні таблиці недостовірні.**\n")
    names = {r[0]: r[1] for r in db.q("SELECT sample_id,name FROM samples")}
    tfidf = tfidf_top(db)
    for sid, name in names.items():
        out.append(f"\n## Вибірка {sid}: {name}\n")
        out.append("### Загальна статистика\n\n| Показник | Значення |\n|---|---|")
        for k, v in basic_stats(db, sid).items():
            out.append(f"| {k} | {v} |")
        out.append("\n### БД_ЧС_ЧМ (частини мови)\n\n| pos | Частина мови | абс. частота | p | відн. похибка, % | V (по підвибірках), % |\n|---|---|---|---|---|---|")
        hom = {h[0]: h for h in homogeneity(db, sid)}
        for pos, ua, f, p, err in relative_error_table(db, sid):
            out.append(f"| {pos} | {ua} | {f} | {p} | {err} | {hom[pos][3]} |")
        out.append("\n### Топ-20 лем (БД_ЧС_лем)\n\n| лема | частота |\n|---|---|")
        for lem, f in db.q("SELECT lemma,abs_freq FROM freq_lemma WHERE sample_id=? ORDER BY abs_freq DESC LIMIT 20", (sid,)):
            out.append(f"| {lem} | {f} |")
        out.append("\n### Топ TF-IDF (лема, середнє по підвибірках)\n\n| лема | TF-IDF |\n|---|---|")
        for lem, v in tfidf[sid]:
            out.append(f"| {lem} | {v} |")
    Path(a.report).write_text("\n".join(out), encoding="utf-8")
    print(f"БД: {a.db}; звіт: {a.report}")


if __name__ == "__main__":
    main()
