import os
import re
import math
import pandas as pd
import requests
from scipy.stats import poisson
from dotenv import load_dotenv
from typing import Any, Optional, Tuple, List, Dict
from datetime import datetime
import pytz

# =========================
# CONFIGURAÇÕES
# =========================
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
OLLAMA_URL = "http://localhost:11434/api/chat"

BATCH_SIZE = 5
DECAY_FACTOR = 0.90

USER_TZ = "America/Sao_Paulo"

# Ligas -> arquivo
LEAGUE_FILES = {
    "1": ("Ligue 1", "ligue-1-2025-UTC.csv"),
    "2": ("Premier League (EPL)", "epl-2025-GMTStandardTime.csv"),
    "3": ("La Liga", "la-liga-2025-UTC.csv"),
    "4": ("Bundesliga", "bundesliga-2025-UTC.csv"),
}

# Vantagem casa (padrão)
HOME_ADV_GOALS_DEFAULT = 0.35

# Se quiser por liga, habilite aqui (opcional)
HOME_ADV_BY_LEAGUE = {
    "Ligue 1": 0.30,
    "Premier League (EPL)": 0.33,
    "La Liga": 0.32,
    "Bundesliga": 0.34,
}

# Linhas permitidas para Over/Under (no modo auto ele escolhe entre essas)
TOTAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]

# Até quantos gols somar na aproximação de Poisson (quanto maior, mais preciso)
MAX_GOALS = 10

# =========================
# UTILITÁRIOS
# =========================
def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)

def _parse_result(result: Any) -> Tuple[Optional[int], Optional[int]]:
    if result is None:
        return None, None
    s = str(result).strip().replace(" ", "")
    if s in ("", "-", "N/A", "nan", "None"):
        return None, None
    m = re.match(r"^(\d+)-(\d+)$", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def _parse_user_date(date_str: str) -> datetime.date:
    ds = date_str.strip().lower()
    now_local = datetime.now(pytz.timezone(USER_TZ)).date()

    if ds in ("hoje", "today"):
        return now_local

    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", ds)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d).date()

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", ds)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d).date()

    raise ValueError("Data inválida. Use 'hoje', 'DD/MM/AAAA' ou 'AAAA-MM-DD'.")

def _parse_over_mode(s: str) -> Tuple[str, Optional[float]]:
    """
    Aceita:
      - 'auto'
      - '2.5' / '3.5' etc.
    """
    s = s.strip().lower()
    if s in ("auto", ""):
        return "auto", None
    try:
        line = float(s)
        return "fixed", line
    except:
        raise ValueError("Over dinâmico inválido. Use 'auto' ou um número (ex: 2.5).")

def load_league_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não achado: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]

    goals = df["RESULT"].apply(_parse_result)
    df["HOME_GOALS"] = goals.apply(lambda x: x[0])
    df["AWAY_GOALS"] = goals.apply(lambda x: x[1])

    if "DATE" in df.columns:
        df["DATE_DT"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)

        tz = pytz.timezone(USER_TZ)
        df["DATE_LOCAL"] = df["DATE_DT"].dt.tz_convert(tz)
        df["DATE_LOCAL_DAY"] = df["DATE_LOCAL"].dt.date
    else:
        df["DATE_DT"] = pd.NaT
        df["DATE_LOCAL"] = pd.NaT
        df["DATE_LOCAL_DAY"] = None

    return df

def get_form_from_league_df(team_name: str, league_df: pd.DataFrame, last_n: int = 5) -> str:
    played = league_df.dropna(subset=["HOME_GOALS", "AWAY_GOALS"]).copy()
    if played.empty:
        return "N/A"

    mask_team = (
        played["HOME TEAM"].astype(str).str.lower().eq(team_name.lower())
        | played["AWAY TEAM"].astype(str).str.lower().eq(team_name.lower())
    )
    team_games = played[mask_team].copy()

    if "DATE_DT" in team_games.columns and team_games["DATE_DT"].notna().any():
        team_games = team_games.sort_values("DATE_DT")
    team_games = team_games.tail(last_n)

    res = []
    for _, r in team_games.iterrows():
        h = str(r["HOME TEAM"])
        a = str(r["AWAY TEAM"])
        hg, ag = int(r["HOME_GOALS"]), int(r["AWAY_GOALS"])
        is_home = (h.lower() == team_name.lower())

        if hg > ag:
            res.append("V" if is_home else "D")
        elif ag > hg:
            res.append("V" if not is_home else "D")
        else:
            res.append("E")

    return "-".join(res) if res else "N/A"

# =========================
# WEIGHTED METRICS + POISSON
# =========================
def compute_weighted_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    team_history: Dict[str, List[Dict[str, int]]] = {}

    played = df.dropna(subset=["HOME_GOALS", "AWAY_GOALS"]).copy()
    if played.empty:
        return {}

    if "DATE_DT" in played.columns and played["DATE_DT"].notna().any():
        played = played.sort_values("DATE_DT")

    for _, r in played.iterrows():
        h, a = str(r["HOME TEAM"]), str(r["AWAY TEAM"])
        hg, ag = int(r["HOME_GOALS"]), int(r["AWAY_GOALS"])
        team_history.setdefault(h, []).append({"gf": hg, "ga": ag})
        team_history.setdefault(a, []).append({"gf": ag, "ga": hg})

    final_metrics: Dict[str, Dict[str, float]] = {}
    for team, games in team_history.items():
        w_gf = w_ga = total_w = 0.0
        for i, g in enumerate(reversed(games)):
            w = DECAY_FACTOR ** i
            w_gf += g["gf"] * w
            w_ga += g["ga"] * w
            total_w += w
            if i >= 10:
                break
        final_metrics[team] = {"att": w_gf / total_w, "def": w_ga / total_w}

    return final_metrics

def poisson_arrays(lh: float, la: float, max_goals: int = MAX_GOALS) -> Tuple[List[float], List[float]]:
    ph = [poisson.pmf(i, lh) for i in range(max_goals + 1)]
    pa = [poisson.pmf(i, la) for i in range(max_goals + 1)]
    return ph, pa

def market_probs_from_poisson(lh: float, la: float, total_line: float) -> Dict[str, float]:
    ph, pa = poisson_arrays(lh, la, MAX_GOALS)

    win_h = draw = win_a = 0.0
    btts_yes = 0.0
    over = under = 0.0

    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]

            if i > j:
                win_h += p
            elif j > i:
                win_a += p
            else:
                draw += p

            if i > 0 and j > 0:
                btts_yes += p

            if (i + j) > total_line:
                over += p
            else:
                under += p

    return {
        "win_h": win_h,
        "draw": draw,
        "win_a": win_a,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "over": over,
        "under": under,
    }

def choose_auto_line(expected_total_goals: float) -> float:
    # escolhe a linha mais próxima do total esperado, limitada às TOTAL_LINES
    return min(TOTAL_LINES, key=lambda x: abs(x - expected_total_goals))

def predict_game(h: str, a: str, metrics: Dict[str, Dict[str, float]], home_adv: float, over_mode: str, over_fixed: Optional[float]):
    h_stats = metrics.get(h, {"att": 1.4, "def": 1.2})
    a_stats = metrics.get(a, {"att": 1.4, "def": 1.2})

    lh = (h_stats["att"] + a_stats["def"]) / 2 + home_adv
    la = (a_stats["att"] + h_stats["def"]) / 2
    lh, la = max(0.1, lh), max(0.1, la)

    exp_total = lh + la

    if over_mode == "fixed":
        line = over_fixed if over_fixed is not None else 2.5
    else:
        line = choose_auto_line(exp_total)

    probs = market_probs_from_poisson(lh, la, total_line=line)

    return {
        "xg_h": lh,
        "xg_a": la,
        "exp_total": exp_total,
        "total_line": line,
        **probs
    }

# =========================
# VALUE BET (ODDS OPCIONAIS)
# =========================
def implied_prob_from_odds(odds: Optional[float]) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and pd.isna(odds)):
        return None
    if odds <= 1.0:
        return None
    return 1.0 / float(odds)

def expected_value(prob: float, odds: Optional[float]) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and pd.isna(odds)):
        return None
    odds = float(odds)
    if odds <= 1.0:
        return None
    # EV por 1 unidade apostada: p*(odds-1) - (1-p)*1 = p*odds - 1
    return prob * odds - 1.0

# =========================
# OLLAMA
# =========================
def processar_lote_ollama(lote_texto: str) -> str:
    system_prompt = (
        "Você é um Analista Quantitativo de Futebol (Quant). "
        "Receba os dados estatísticos (Poisson & Weighted Stats) e gere APENAS uma tabela Markdown.\n"
        "REGRAS:\n"
        "1. NÃO mostre seu pensamento (<think>).\n"
        "2. A Coluna 'Aposta Value' deve indicar a melhor opção matemática baseada nas % (e EV se existir).\n"
        "3. Se 'Forma Recente' for ruim (muitos 'D'), avise na obs.\n"
        "4. Seja estritamente numérico e frio."
        "5. Não inclua o Data e hora do jogo na tabela e mostre tambem qual a aposta surgerida"
    )

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": lote_texto},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            },
            timeout=150000,
        )
        if r.status_code == 200:
            content = r.json()["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip()
        return f"Erro API: {r.status_code} - {r.text[:200]}"
    except Exception as e:
        return f"Erro Conexão: {e}"

# =========================
# MENU + FILTRO
# =========================
def escolher_liga() -> Tuple[str, str]:
    print("Selecione a liga para analisar:")
    for k, (name, file) in LEAGUE_FILES.items():
        print(f" {k}) {name}  ->  {file}")
    op = input("Opção: ").strip()
    if op not in LEAGUE_FILES:
        raise ValueError("Opção de liga inválida.")
    return LEAGUE_FILES[op]

def filtrar_jogos_do_dia(df: pd.DataFrame, dia: datetime.date) -> pd.DataFrame:
    # futuro = sem placar
    future = df[df["HOME_GOALS"].isna() | df["AWAY_GOALS"].isna()].copy()
    if future.empty:
        return future

    day_games = future[future["DATE_LOCAL_DAY"] == dia].copy()
    if "DATE_LOCAL" in day_games.columns and day_games["DATE_LOCAL"].notna().any():
        day_games = day_games.sort_values("DATE_LOCAL")
    return day_games

# =========================
# MAIN
# =========================
def main():
    print("=== ANALISADOR QUANTITATIVO (Poisson + Weighted Recency) ===")
    print(f"Modelo IA: {OLLAMA_MODEL}")
    print(f"Fator Decaimento: {DECAY_FACTOR}\n")

    league_name, csv_file = escolher_liga()

    date_in = input("Qual dia analisar? (hoje / DD/MM/AAAA / AAAA-MM-DD): ").strip()
    dia = _parse_user_date(date_in)

    over_in = input("Linha Over/Under (auto / 2.5 / 3.5 ...): ").strip()
    over_mode, over_fixed = _parse_over_mode(over_in)

    home_adv = HOME_ADV_BY_LEAGUE.get(league_name, HOME_ADV_GOALS_DEFAULT)

    print(f"\nLiga: {league_name}")
    print(f"CSV: {csv_file}")
    print(f"Dia (local {USER_TZ}): {dia}")
    print(f"HomeAdv: {home_adv}")
    print(f"Over/Under: {('AUTO' if over_mode=='auto' else over_fixed)}\n")

    df = load_league_data(csv_file)
    metrics = compute_weighted_metrics(df)

    jogos = filtrar_jogos_do_dia(df, dia)
    if jogos.empty:
        print("Sem jogos futuros para esse dia.")
        return

    buffer_jogos = []
    for _, r in jogos.iterrows():
        h, a = _safe_str(r["HOME TEAM"]), _safe_str(r["AWAY TEAM"])

        # Previsão
        pred = predict_game(h, a, metrics, home_adv, over_mode, over_fixed)

        fh = get_form_from_league_df(h, df)
        fa = get_form_from_league_df(a, df)

        # Odds opcionais (se existirem no CSV, usa; se não, ignora)
        # Sugestão de nomes de coluna (você pode adaptar depois):
        odds_home = r.get("ODDS_HOME", None)
        odds_draw = r.get("ODDS_DRAW", None)
        odds_away = r.get("ODDS_AWAY", None)
        odds_over = r.get("ODDS_OVER", None)
        odds_under = r.get("ODDS_UNDER", None)
        odds_btts_yes = r.get("ODDS_BTTS_YES", None)
        odds_btts_no = r.get("ODDS_BTTS_NO", None)

        # EVs (se odds presentes)
        ev_home = expected_value(pred["win_h"], odds_home)
        ev_draw = expected_value(pred["draw"], odds_draw)
        ev_away = expected_value(pred["win_a"], odds_away)
        ev_over = expected_value(pred["over"], odds_over)
        ev_under = expected_value(pred["under"], odds_under)
        ev_btts_yes = expected_value(pred["btts_yes"], odds_btts_yes)
        ev_btts_no = expected_value(pred["btts_no"], odds_btts_no)

        hora_local = ""
        if "DATE_LOCAL" in r and pd.notna(r["DATE_LOCAL"]):
            hora_local = str(r["DATE_LOCAL"])

        dados = (
            f"PARTIDA: {h} x {a}\n"
            f" - DATA (local): {hora_local}\n"
            f" - 1X2: Casa {pred['win_h']:.1%} | Empate {pred['draw']:.1%} | Fora {pred['win_a']:.1%}\n"
            f" - xG: {h} {pred['xg_h']:.2f} vs {pred['xg_a']:.2f} {a} | Total xG {pred['exp_total']:.2f}\n"
            f" - TOTAL: Linha {pred['total_line']} | Over {pred['over']:.1%} | Under {pred['under']:.1%}\n"
            f" - BTTS: Sim {pred['btts_yes']:.1%} | Não {pred['btts_no']:.1%}\n"
            f" - FORMA (ult.5): {h}[{fh}] vs {a}[{fa}]\n"
            f" - VALUE (se odds existirem):\n"
            f"   EV Casa={('N/A' if ev_home is None else f'{ev_home:+.3f}')}, "
            f"Empate={('N/A' if ev_draw is None else f'{ev_draw:+.3f}')}, "
            f"Fora={('N/A' if ev_away is None else f'{ev_away:+.3f}')}\n"
            f"   EV Over{pred['total_line']}={('N/A' if ev_over is None else f'{ev_over:+.3f}')}, "
            f"EV Under{pred['total_line']}={('N/A' if ev_under is None else f'{ev_under:+.3f}')}\n"
            f"   EV BTTS Sim={('N/A' if ev_btts_yes is None else f'{ev_btts_yes:+.3f}')}, "
            f"EV BTTS Não={('N/A' if ev_btts_no is None else f'{ev_btts_no:+.3f}')}\n"
        )
        buffer_jogos.append(dados)

    total = len(buffer_jogos)
    for i in range(0, total, BATCH_SIZE):
        chunk = buffer_jogos[i:i+BATCH_SIZE]
        print(f"📊 Processando Batch {i+1}-{min(i+BATCH_SIZE, total)}...")

        texto_envio = "Analise estatisticamente estes jogos:\n\n" + "\n---\n".join(chunk)
        resp = processar_lote_ollama(texto_envio)
        print("\n" + resp + "\n")
        print("=" * 60)

if __name__ == "__main__":
    main()
