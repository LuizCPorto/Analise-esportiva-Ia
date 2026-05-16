import os
import re
import pandas as pd
import requests
import numpy as np
from scipy.stats import poisson
from dotenv import load_dotenv
from typing import Any, Optional, Tuple, List, Dict
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = "http://localhost:11434/api/chat"
LEAGUES_DIR = "leagues" 

# Parâmetros Quant
DECAY_FACTOR = 0.88     # Peso dos jogos recentes
HOME_ADV_GOALS = 0.35   # Vantagem de gols do mandante
RECENT_GAMES_FOCUS = 5  # Jogos para análise de forma
TOTAL_GAMES_ANALYSIS = 12 # Jogos para análise matemática

# Parâmetro Dixon-Coles (Correção de Empates 0-0 e 1-1)
RHO = -0.13 

LEAGUE_FILES_MAP = {
    "1": ("Ligue 1", "ligue-1-2025-UTC.csv"),
    "2": ("Premier League", "epl-2025-GMTStandardTime.csv"),
    "3": ("La Liga", "la-liga-2025-UTC.csv"),
    "4": ("Bundesliga", "bundesliga-2025-UTC.csv")
}

# =========================
# PROCESSAMENTO DE DADOS
# =========================
def _parse_result(result: Any) -> Tuple[Optional[int], Optional[int]]:
    if result is None or pd.isna(result): 
        return None, None
    s = str(result).strip().replace(" ", "")
    m = re.match(r"^(\d+)-(\d+)$", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def load_data(file_path: str):
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip().upper() for c in df.columns]
    
    goals = df["RESULT"].apply(_parse_result)
    df["HOME_GOALS"], df["AWAY_GOALS"] = goals.apply(lambda x:x[0]), goals.apply(lambda x:x[1])
    
    if "DATE" in df.columns:
        df["DATE_DT"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    
    return df

# =========================
# ANÁLISE DE FORMA
# =========================
def obter_forma_recente(team_name: str, df_history: pd.DataFrame, n_jogos: int = RECENT_GAMES_FOCUS) -> Dict:
    jogos = []
    gols_marcados = []
    gols_sofridos = []
    
    df_valid = df_history.dropna(subset=["HOME_GOALS"]).copy()
    
    for _, r in df_valid.iterrows():
        if r["HOME TEAM"] == team_name:
            gf, gc = int(r["HOME_GOALS"]), int(r["AWAY_GOALS"])
            resultado = "V" if gf > gc else ("E" if gf == gc else "D")
            jogos.append(f"{resultado}")
            gols_marcados.append(gf)
            gols_sofridos.append(gc)
            
        elif r["AWAY TEAM"] == team_name:
            gf, gc = int(r["AWAY_GOALS"]), int(r["HOME_GOALS"])
            resultado = "V" if gf > gc else ("E" if gf == gc else "D")
            jogos.append(f"{resultado}")
            gols_marcados.append(gf)
            gols_sofridos.append(gc)
    
    jogos_recentes = jogos[-n_jogos:]
    gm_recentes = gols_marcados[-n_jogos:]
    gs_recentes = gols_sofridos[-n_jogos:]
    
    if not jogos_recentes:
        return {'forma': 'N/A', 'vitorias': 0, 'media_gf': 0, 'media_gs': 0}
    
    return {
        'forma': " ".join(jogos_recentes),
        'vitorias': jogos_recentes.count("V"),
        'media_gf': round(np.mean(gm_recentes), 2),
        'media_gs': round(np.mean(gs_recentes), 2),
        'total_jogos': len(jogos_recentes)
    }

def obter_confrontos_diretos(h_name: str, a_name: str, df_history: pd.DataFrame, n: int = 3) -> Dict:
    df_valid = df_history.dropna(subset=["HOME_GOALS"])
    confrontos = df_valid[
        ((df_valid["HOME TEAM"] == h_name) & (df_valid["AWAY TEAM"] == a_name)) |
        ((df_valid["HOME TEAM"] == a_name) & (df_valid["AWAY TEAM"] == h_name))
    ].tail(n)
    
    if len(confrontos) == 0:
        return {'texto': 'Sem histórico recente'}
    
    vitorias_h = 0
    empates = 0
    
    for _, r in confrontos.iterrows():
        hg, ag = int(r["HOME_GOALS"]), int(r["AWAY_GOALS"])
        if r["HOME TEAM"] == h_name:
            if hg > ag: vitorias_h += 1
            elif hg == ag: empates += 1
        else:
            if ag > hg: vitorias_h += 1
            elif hg == ag: empates += 1
            
    return {'texto': f"{vitorias_h}V {empates}E {len(confrontos)-vitorias_h-empates}D nos últimos {len(confrontos)} jogos"}

# =========================
# ENGINE MATEMÁTICA (DIXON-COLES)
# =========================
def compute_metrics(df: pd.DataFrame) -> Dict:
    team_history = {}
    df_history = df.dropna(subset=["HOME_GOALS"])
    
    for _, r in df_history.iterrows():
        h, a = r["HOME TEAM"], r["AWAY TEAM"]
        hg, ag = int(r["HOME_GOALS"]), int(r["AWAY_GOALS"])
        
        if h not in team_history: team_history[h] = []
        if a not in team_history: team_history[a] = []
        
        team_history[h].append({'gf': hg, 'ga': ag})
        team_history[a].append({'gf': ag, 'ga': hg})
        
    metrics = {}
    for team, games in team_history.items():
        w_gf, w_ga, total_weight = 0, 0, 0
        
        for i, g in enumerate(reversed(games)):
            weight = DECAY_FACTOR ** i
            w_gf += g['gf'] * weight
            w_ga += g['ga'] * weight
            total_weight += weight
            if i >= TOTAL_GAMES_ANALYSIS - 1: break
            
        metrics[team] = {
            'att': w_gf / total_weight if total_weight > 0 else 1.2,
            'def': w_ga / total_weight if total_weight > 0 else 1.2
        }
    return metrics

def predict_dixon_coles(h_name: str, a_name: str, metrics: Dict) -> Dict:
    # 1. Obter Forças de Ataque e Defesa
    h_stats = metrics.get(h_name, {'att': 1.2, 'def': 1.2})
    a_stats = metrics.get(a_name, {'att': 1.2, 'def': 1.2})
    
    # 2. Calcular Gols Esperados (xG)
    lambda_h = max(0.1, (h_stats['att'] + a_stats['def']) / 2 + HOME_ADV_GOALS)
    lambda_a = max(0.1, (a_stats['att'] + h_stats['def']) / 2)
    
    # 3. Matriz de Probabilidades (0-0 até 6-6)
    limit = 6
    matrix = np.zeros((limit+1, limit+1))
    
    # Poisson Básico
    for i in range(limit+1):
        for j in range(limit+1):
            matrix[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
            
    # 4. Ajuste Dixon-Coles (Correção de placares baixos)
    # RHO corrige a subestimação de empates 0-0 e 1-1
    correction = {
        (0, 0): 1 - (lambda_h * lambda_a * RHO),
        (0, 1): 1 + (lambda_h * RHO),
        (1, 0): 1 + (lambda_a * RHO),
        (1, 1): 1 - RHO
    }
    
    for (i, j), factor in correction.items():
        matrix[i, j] *= factor
        
    # Renormalizar (garantir soma = 1)
    matrix = matrix / np.sum(matrix)
    
    # 5. Extrair Mercados
    win_h = np.sum(np.tril(matrix, -1)) # Soma triângulo inferior
    draw = np.sum(np.diag(matrix))      # Soma diagonal
    win_a = np.sum(np.triu(matrix, 1))  # Soma triângulo superior
    
    # Over/Under e BTTS
    btts_prob = 0
    ou_probs = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
    
    for i in range(limit+1):
        for j in range(limit+1):
            prob = matrix[i, j]
            total = i + j
            
            if i > 0 and j > 0: btts_prob += prob
            
            for k in ou_probs.keys():
                if total > k: ou_probs[k] += prob
                
    return {
        'win_h': win_h, 'draw': draw, 'win_a': win_a,
        'over_under': ou_probs,
        'btts': btts_prob,
        'xg_h': lambda_h, 'xg_a': lambda_a
    }

# ============================================
# SELEÇÃO INTELIGENTE (BASEADA EM ODD JUSTA)
# ============================================
def selecionar_melhores_opcoes(probs: Dict) -> Dict:
    """
    Retorna opções baseadas na 'Odd Justa' (1 / Probabilidade).
    Cabe ao apostador comparar a Odd Justa com a Odd da Casa de Apostas.
    """
    opcoes = []
    
    # Função auxiliar
    def add_opt(tipo, prob, categoria):
        if prob < 0.01: return
        odd_justa = 1 / prob
        opcoes.append({
            'tipo': tipo,
            'prob': prob,
            'odd_justa': odd_justa,
            'categoria': categoria
        })

    # 1. Match Odds (1x2)
    add_opt("Casa Vence", probs['win_h'], 'Match Odds')
    add_opt("Empate", probs['draw'], 'Match Odds')
    add_opt("Visitante Vence", probs['win_a'], 'Match Odds')
    
    # 2. Gols
    add_opt("Over 1.5", probs['over_under'][1.5], 'Gols')
    add_opt("Over 2.5", probs['over_under'][2.5], 'Gols')
    add_opt("Under 2.5", 1 - probs['over_under'][2.5], 'Gols')
    
    # 3. BTTS
    add_opt("Ambos Marcam", probs['btts'], 'BTTS')
    add_opt("Ambos Não Marcam", 1 - probs['btts'], 'BTTS')

    # === LÓGICA DE SELEÇÃO ===
    
    # SEGURO: Alta Probabilidade (>70%), Odd Justa baixa mas aceitável (>1.20)
    seguros = [o for o in opcoes if o['prob'] >= 0.70 and o['odd_justa'] >= 1.20]
    seguros.sort(key=lambda x: x['prob'], reverse=True)
    
    # VALUE/RISCO: Probabilidade decente (>45%) mas Odd Justa sugere que o mercado paga bem
    # Aqui procuramos "erros" de precificação, geralmente em Underdogs ou Empates
    values = [o for o in opcoes if 0.40 <= o['prob'] < 0.70]
    values.sort(key=lambda x: x['prob'], reverse=True) # Prioriza os mais prováveis do grupo de risco

    melhor_seguro = seguros[0] if seguros else None
    melhor_value = values[0] if values else None
    
    # Se não tiver seguro, pega o melhor value como principal
    if not melhor_seguro and melhor_value:
        melhor_seguro = melhor_value
        melhor_value = values[1] if len(values) > 1 else None

    return {
        'seguro': melhor_seguro,
        'value': melhor_value
    }

# =========================
# IA (OLLAMA)
# =========================
def validar_com_ia(jogo: Dict, p_seguro: Dict, p_value: Dict) -> str:
    if not p_seguro: return "Sem dados suficientes"
    
    txt_seguro = f"{p_seguro['tipo']} (Odd Justa: {p_seguro['odd_justa']:.2f})"
    txt_value = f"{p_value['tipo']} (Odd Justa: {p_value['odd_justa']:.2f})" if p_value else "N/A"
    
    prompt = f"""
    Analise este jogo de futebol como um Trader Esportivo Profissional.
    
    JOGO: {jogo['mandante']} vs {jogo['visitante']}
    
    DADOS QUANTITATIVOS (MODELO DIXON-COLES):
    - Probabilidades: Casa {jogo['ph']:.0%} | Empate {jogo['pd']:.0%} | Fora {jogo['pa']:.0%}
    - Gols Esperados (xG): {jogo['xgh']:.2f} vs {jogo['xga']:.2f}
    - Prob Over 2.5: {jogo['po25']:.0%}
    - Histórico: {jogo['historico']}
    
    SUGESTÃO DO MODELO:
    1. 🔒 BANCA SEGURA: {txt_seguro}
    2. 💎 OPORTUNIDADE DE VALOR: {txt_value}
    
    TAREFA:
    Valide as sugestões. Compare a 'Odd Justa' com o que você conhece das ligas.
    Se a sugestão 'Segura' for em Odd Justa < 1.25, diga se vale o risco.
    
    Responda em UMA FRASE ÚNICA no formato:
    [APROVADO/CUIDADO] | [Comentário breve sobre o confronto]
    """
    
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2}
        }, timeout=20000)
        
        if r.status_code == 200:
            res = r.json()["message"]["content"]
            return res.split('\n')[0].replace("<think>", "").strip()
        return "Erro IA"
    except:
        return "IA Indisponível"

# =========================
# MAIN
# =========================
def main():
    print("\n" + "="*60)
    print(" ⚽ ANALISADOR PROFISSIONAL (DIXON-COLES + FAIR ODDS)")
    print("="*60)
    print("📢 COMO USAR A PLANILHA GERADA:")
    print("Compare a coluna 'ODD JUSTA' com a Odd da Bet365.")
    print("➤ Se Odd Bet365 > Odd Justa = APOSTA DE VALOR (✅)")
    print("➤ Se Odd Bet365 < Odd Justa = NÃO APOSTE (❌)\n")

    for k, v in LEAGUE_FILES_MAP.items():
        print(f" [{k}] {v[0]}")
    
    op = input("\nEscolha a liga: ").strip()
    if op not in LEAGUE_FILES_MAP: return
    
    nome_liga, arquivo = LEAGUE_FILES_MAP[op]
    df = load_data(os.path.join(LEAGUES_DIR, arquivo))
    metrics = compute_metrics(df)
    
    futuros = df[df["HOME_GOALS"].isna() & df["DATE_DT"].notna()]
    datas = sorted(futuros["DATE_DT"].unique())
    
    if not datas: return print("Sem jogos futuros.")
    
    print(f"\nDatas disponíveis:")
    for i, d in enumerate(datas):
        print(f" [{i}] {d.strftime('%d/%m/%Y')}")
        
    idx = int(input("\nÍndice da data: "))
    data_alvo = datas[idx]
    jogos = futuros[futuros["DATE_DT"] == data_alvo]
    
    resultados = []
    print(f"\nProcessando {len(jogos)} jogos...")
    
    for _, r in jogos.iterrows():
        h, a = r["HOME TEAM"], r["AWAY TEAM"]
        print(f"Analilsando {h} x {a}...", end='\r')
        
        # 1. Matemática
        preds = predict_dixon_coles(h, a, metrics)
        
        # 2. Contexto
        forma_h = obter_forma_recente(h, df)
        forma_a = obter_forma_recente(a, df)
        hist = obter_confrontos_diretos(h, a, df)
        
        # 3. Seleção
        picks = selecionar_melhores_opcoes(preds)
        seguro = picks['seguro']
        value = picks['value']
        
        # 4. IA
        jogo_resumo = {
            'mandante': h, 'visitante': a,
            'ph': preds['win_h'], 'pd': preds['draw'], 'pa': preds['win_a'],
            'xgh': preds['xg_h'], 'xga': preds['xg_a'],
            'po25': preds['over_under'][2.5],
            'historico': hist['texto']
        }
        insight = validar_com_ia(jogo_resumo, seguro, value)
        
        resultados.append({
            "Data": data_alvo,
            "Mandante": h,
            "Visitante": a,
            
            # SEGURO
            "🔒 Palpite Seguro": seguro['tipo'] if seguro else "-",
            "Prob Seguro (%)": round(seguro['prob']*100, 1) if seguro else 0,
            "Odd Justa (Seguro)": round(seguro['odd_justa'], 2) if seguro else 0,
            
            # VALUE
            "💎 Opção Value": value['tipo'] if value else "-",
            "Prob Value (%)": round(value['prob']*100, 1) if value else 0,
            "Odd Justa (Value)": round(value['odd_justa'], 2) if value else 0,
            
            "IA Insight": insight,
            "xG Casa": round(preds['xg_h'], 2),
            "xG Fora": round(preds['xg_a'], 2),
            "Forma Casa": f"{forma_h['vitorias']}V ({forma_h['media_gf']} GF)",
            "Forma Fora": f"{forma_a['vitorias']}V ({forma_a['media_gf']} GF)"
        })

    df_final = pd.DataFrame(resultados)
    file = f"Tips_{nome_liga}_{data_alvo}.xlsx"
    df_final.to_excel(file, index=False)
    print(f"\n\n✅ Sucesso! Arquivo gerado: {file}")

if __name__ == "__main__":
    main()