# ⚽ Football Quant & AI Analyzer

Um bot de análise quantitativa para ligas de futebol europeias. Este projeto utiliza o modelo matemático de **Dixon-Coles** para calcular probabilidades e *Expected Goals (xG)*, cruzando os dados com um modelo de inteligência artificial local (**Ollama**) para validar palpites e encontrar apostas de valor (Fair Odds).

## 🚀 Principais Funcionalidades

- **Motor Matemático (Dixon-Coles):** Utiliza a distribuição de Poisson (`scipy.stats`) com correção de RHO (-0.13) para prever com maior precisão placares baixos (0-0 e 1-1), mitigando a subestimação padrão do modelo de Poisson.
- **Cálculo de "Odd Justa" (Fair Odds):** Converte as probabilidades matemáticas em *Odds Justas*, separando as sugestões em "Banca Segura" (alta probabilidade) e "Value" (probabilidade média, mas com possível erro de precificação das casas de aposta).
- **Análise de Contexto:** Processa forma recente (fator de decaimento de peso para jogos antigos) e histórico de confrontos diretos.
- **Validação com IA Local:** Integração via API REST com o **Ollama** (Llama 3.1) atuando como um "Trader Esportivo Profissional" para validar as sugestões matemáticas e gerar um insight final em texto.
- **Exportação Automatizada:** Gera uma planilha Excel (`.xlsx`) pronta para leitura com todos os cálculos, probabilidades e palpites do dia.
