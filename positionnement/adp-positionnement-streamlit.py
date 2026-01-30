import io
import re
import csv
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================
# CONFIG
# =========================
BAREME_FILENAME = "bareme_powerbi_classmarker.csv"
N_SCORED = 14  # Barème logique Q1..Q14
DEFAULT_SEUIL_CURSUS_2 = 28  # ajustable dans l'UI

# =========================
# HELPERS
# =========================
def norm_text(s: str) -> str:
    """Normalisation pour rendre le matching barème <-> libellés plus tolérant."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")          # BOM
    s = s.replace("\u00A0", " ")         # nbsp
    s = s.replace("’", "'")              # apostrophe typographique
    s = s.replace("…", "...")            # ellipsis
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def read_uploaded_lines(uploaded_file) -> list[str]:
    raw = uploaded_file.getvalue()
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    raise ValueError("Impossible de décoder le CSV (encodage non supporté).")

def split_letters(cell: str) -> list[str]:
    """
    Extrait A-E. Supporte multi-réponses: "A,C" / "A | C" / "A C" etc.
    """
    cell = norm_text(cell)
    if not cell or cell.lower() in {"no answer", "n/a"}:
        return []
    return re.findall(r"[A-E]", cell.upper())

# =========================
# PARSE QUESTION BANK (Options) FROM CLASSMARKER FILE
# =========================
def extract_question_bank(lines: list[str]) -> dict:
    """
    Retour:
      bank["Q3"] = {"question_text": "...", "options": {"A": "...", "B": "..."}}
    """
    bank = {}
    q_re = re.compile(r"^Q(\d+)\s*$")
    i = 0

    while i < len(lines):
        m = q_re.match(lines[i].strip())
        if not m:
            i += 1
            continue

        qid = f"Q{m.group(1)}"
        bank[qid] = {"question_text": "", "options": {}}
        i += 1

        while i < len(lines) and not q_re.match(lines[i].strip()):
            line = lines[i].strip()

            # Question,"..."
            if line.startswith("Question,") or line.startswith('"Question",') or line.startswith('Question,"'):
                parts = next(csv.reader([line]))
                if len(parts) >= 2:
                    bank[qid]["question_text"] = norm_text(parts[1])

            # Options,"A) ..." + lignes suivantes
            if line.startswith("Options,") or line.startswith('"Options",') or line.startswith('Options,"'):
                parts = next(csv.reader([line]))
                if len(parts) >= 2:
                    cell = norm_text(parts[1])
                    mm = re.match(r"^([A-E])\)\s*(.*)$", cell)
                    if mm:
                        bank[qid]["options"][mm.group(1)] = norm_text(mm.group(2))

                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt:
                        j += 1
                        continue
                    if q_re.match(nxt):
                        break
                    if nxt.startswith(",") or nxt.startswith(',"'):
                        parts2 = next(csv.reader([nxt]))
                        for c in parts2:
                            c = norm_text(c)
                            mm2 = re.match(r"^([A-E])\)\s*(.*)$", c)
                            if mm2:
                                bank[qid]["options"][mm2.group(1)] = norm_text(mm2.group(2))
                    j += 1
                i = j - 1

            i += 1

    return bank

def sorted_qids(bank: dict) -> list[str]:
    """Q1..Qn triées numériquement."""
    return sorted(bank.keys(), key=lambda x: int(re.findall(r"\d+", x)[0]))

def get_choice_qids(bank: dict) -> list[str]:
    """Questions à choix = celles qui ont des options."""
    qids = [qid for qid, meta in bank.items() if meta.get("options")]
    return sorted(qids, key=lambda x: int(re.findall(r"\d+", x)[0]))

def get_text_qids(bank: dict) -> list[str]:
    """Questions texte libre = celles sans options (souvent nom/email)."""
    qids = [qid for qid, meta in bank.items() if not meta.get("options")]
    return sorted(qids, key=lambda x: int(re.findall(r"\d+", x)[0]))

# =========================
# EXTRACT PARTICIPANTS (Answered:)
# =========================
def extract_participant_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if "Answered:" in line:
            rows.append(next(csv.reader([line])))
    return rows

# =========================
# LOAD BAREME
# =========================
HERE = Path(__file__).resolve().parent
BAREME_PATH = HERE / BAREME_FILENAME

def load_bareme() -> pd.DataFrame:
    df = pd.read_csv(BAREME_PATH, encoding="utf-8-sig")
    df.columns = [norm_text(c) for c in df.columns]
    for col in ["question", "reponse", "points"]:
        if col not in df.columns:
            raise ValueError(f"Le barème doit contenir la colonne '{col}'.")
    df["question"] = df["question"].map(norm_text)
    df["reponse_norm"] = df["reponse"].map(norm_text)
    df["key"] = df["question"] + "||" + df["reponse_norm"]
    return df

# =========================
# CORE: DECODE + SCORE (robuste à l'ordre)
# =========================
def decode_and_score(rows: list[list[str]], bank: dict, bareme: pd.DataFrame, seuil: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Associe chaque réponse à son Qid dans l'ordre Q1..Qn (peu importe où sont nom/email).
    - Détecte les 14 questions à choix (options A/B/C..), les mappe au barème Q1..Q14 par ordre.
    - Nom / email récupérés depuis les questions texte libre (si présentes), sinon vides.
    """
    bareme_map = dict(zip(bareme["key"], bareme["points"]))

    qids_all = sorted_qids(bank)         # ex: Q1..Q16
    qids_choice = get_choice_qids(bank)  # ex: Q3..Q16 ou Q1..Q14
    qids_text = get_text_qids(bank)      # ex: Q1..Q2 ou Q15..Q16

    if len(qids_choice) < N_SCORED:
        raise ValueError(f"Questions à choix détectées: {len(qids_choice)}. Attendu au moins {N_SCORED}.")

    # On ne garde que les 14 premières à choix (au cas où)
    qids_choice = qids_choice[:N_SCORED]

    # mapping contrôle qualité : barème Q1..Q14 -> Qid ClassMarker + texte question
    mapping_rows = []
    for i in range(1, N_SCORED + 1):
        cm_qid = qids_choice[i-1]
        mapping_rows.append({
            "Barème": f"Q{i}",
            "Question ClassMarker": cm_qid,
            "Texte question (ClassMarker)": bank.get(cm_qid, {}).get("question_text", "")
        })
    mapping_df = pd.DataFrame(mapping_rows)

    # heuristique pour trouver nom/email dans les questions texte (si présentes)
    def pick_text_answer(text_qids, answers_by_qid, keyword):
        for qid in text_qids:
            qt = bank.get(qid, {}).get("question_text", "").lower()
            if keyword in qt:
                return norm_text(answers_by_qid.get(qid, ""))
        return ""

    out_rows = []
    for r in rows:
        if "Answered:" not in r:
            continue
        idx = r.index("Answered:")
        tail = r[idx + 1 :]

        # Tail doit contenir une réponse par question Q1..Qn
        if len(tail) < len(qids_all):
            # certains exports peuvent tronquer, on ignore la ligne si incomplète
            continue

        # associer réponses -> qid dans l'ordre
        answers_by_qid = {qid: tail[i] for i, qid in enumerate(qids_all)}

        # nom/email si questions texte présentes
        nom = pick_text_answer(qids_text, answers_by_qid, "nom") or norm_text(answers_by_qid.get(qids_text[0], "")) if qids_text else ""
        email = pick_text_answer(qids_text, answers_by_qid, "email") or (norm_text(answers_by_qid.get(qids_text[1], "")) if len(qids_text) > 1 else "") if qids_text else ""

        record = {
            "nom_prenom": nom,
            "email": email,
        }

        total = 0
        zero_count = 0

        # Scoring: barème Q1..Q14 appliqué aux 14 questions à choix détectées (ordre)
        for i in range(1, N_SCORED + 1):
            bareme_q = f"Q{i}"
            cm_qid = qids_choice[i-1]

            # réponse brute = lettres (A/B/...) ou multi
            raw_ans = answers_by_qid.get(cm_qid, "")
            letters = split_letters(raw_ans)

            # lettres -> libellés exacts (options)
            opts = bank.get(cm_qid, {}).get("options", {})
            selected_texts = [norm_text(opts.get(L, L)) for L in letters]
            selected_texts = [t for t in selected_texts if t]

            # multi réponse -> max points
            pts_list = []
            for ans_text in selected_texts:
                key = f"{bareme_q}||{ans_text}"
                if key in bareme_map:
                    pts_list.append(int(bareme_map[key]))

            pts = max(pts_list) if pts_list else 0
            if pts == 0:
                zero_count += 1

            record[f"{bareme_q}_reponse"] = " | ".join(selected_texts)
            record[f"{bareme_q}_points"] = pts
            total += pts

        record["score_total"] = total
        record["cursus"] = "Cursus 2" if total >= seuil else "Cursus 1"
        record["nb_questions_a_0"] = zero_count

        out_rows.append(record)

    return pd.DataFrame(out_rows), mapping_df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultats")
    return output.getvalue()

# =========================
# STREAMLIT UI
# =========================
st.title("Positionnement Power BI — Résultats & Cursus (ClassMarker)")

st.write(
    "Importe **n'importe quel export CSV ClassMarker** (*results_with_answers*). "
    "Le script détecte automatiquement quelles questions sont à choix (scorées) et applique le barème **Q1→Q14** "
    "dans le bon ordre."
)

uploaded_results = st.file_uploader("CSV ClassMarker (results_with_answers)", type=["csv"])
seuil = st.number_input("Seuil pour Cursus 2", min_value=0, max_value=300, value=DEFAULT_SEUIL_CURSUS_2, step=1)

st.caption(f"Barème utilisé automatiquement : `{BAREME_FILENAME}` (dans le même dossier que `app.py`).")

if uploaded_results:
    try:
        lines = read_uploaded_lines(uploaded_results)
        bank = extract_question_bank(lines)
        rows = extract_participant_rows(lines)
        if not rows:
            st.error("Aucune ligne participant trouvée (pas de 'Answered:'). Vérifie l’export ClassMarker.")
            st.stop()

        bareme = load_bareme()
        df_scored, mapping_df = decode_and_score(rows, bank, bareme, int(seuil))

        # ---- Contrôle mapping (ce que tu veux absolument) ----
        with st.expander("Vérification : correspondance barème ↔ questions ClassMarker"):
            st.dataframe(mapping_df, use_container_width=True)

        if df_scored.empty:
            st.error("Aucun participant décodé. Vérifie le CSV exporté (ordre / présence des questions).")
            st.stop()

        # ---- Tableau résultats ----
        st.success(f"Résultats calculés ✅ ({len(df_scored)} participant(s))")
        st.dataframe(df_scored, use_container_width=True)

        suspicious = (df_scored["nb_questions_a_0"] > 0).sum()
        if suspicious > 0:
            st.warning(
                f"{suspicious} participant(s) ont au moins une question à 0 point. "
                "Cela arrive si un libellé de réponse ne matche pas exactement le barème (ponctuation/espaces)."
            )

        # ---- Export Excel ----
        xlsx = to_excel_bytes(df_scored)
        st.download_button(
            "Télécharger le fichier Excel (scores + cursus)",
            data=xlsx,
            file_name="resultats_positionnement_powerbi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except FileNotFoundError:
        st.error(
            f"Barème introuvable : `{BAREME_FILENAME}`.\n"
            "Place le fichier barème dans le même dossier que `app.py`."
        )
    except Exception as e:
        st.error(f"Erreur : {e}")
