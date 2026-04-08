# Contexto Mestre do Projeto: TCC MBA - André Furlanetti

## 1. Visão Geral do Projeto
Este projeto de conclusão de curso para o **MBA em Economia, Investimentos e Banking** investiga a correlação entre o sentimento das notícias financeiras e o comportamento do mercado acionário brasileiro.

- **Objetivo:** Validar se a polaridade das notícias atua como indicador antecedente ou coincidente para o ETF **BOVA11**.
- **Hipótese:** Existe uma correlação estatisticamente significante entre o Índice de Sentimento Diário (ISD) e os retornos logarítmicos do ativo.
- **Período de Análise:** 01/01/2025 até 31/12/2025.

## 2. Perfil do Desenvolvedor & Padrões de Código
- **Autor:** Engenheiro de Computação / Engenheiro de Software Sênior.
- **Expectativa:** Código limpo (Clean Code), modularizado, tipado (Type Hinting) e com tratamento de exceções robusto.
- **Ambiente:** Python 3.10+.

## 3. Stack Tecnológica
- **Extração de Dados:** `requests`, `beautifulsoup4`.
- **Análise de Dados:** `pandas`, `numpy`.
- **Dados de Mercado:** `yfinance`.
- **Machine Learning:** `scikit-learn` (Naive Bayes e MLP).
- **Estatística:** `scipy.stats`, `statsmodels`.

## 4. Metodologia de Implementação

### Módulo A: Web Scraping (InfoMoney)
- **URL Alvo:** `https://www.infomoney.com.br/busca/?q=ibovespa`
- **Paginação:** Iterar via parâmetro `&paged=N`.
- **Seletores CSS Críticos:**
    - Container da Notícia: `a` com classe `group hover:bg-surface-card-three`.
    - Título: `div` com classe `text-content-pure`.
    - Data: `div` com classe `text-content-three`.
- **Regra de Parada:** O loop de coleta deve encerrar obrigatoriamente ao encontrar notícias com data anterior a **01/01/2025**.

### Módulo B: Processamento e NLP
- **Limpeza:** Remoção de pontuação, caracteres especiais e conversão para lowercase.
- **Stop-words:** Utilizar lista específica para o idioma português.
- **Vetorização:** Implementar **TF-IDF** (Term Frequency-Inverse Document Frequency).

### Módulo C: Machine Learning
Comparação de dois modelos de classificação (Positivo, Negativo, Neutro):
1. **Naive Bayes (MultinomialNB):** Modelo probabilístico baseline.
2. **Multilayer Perceptron (MLPClassifier):** Rede neural para capturar padrões não-lineares.
- **Validação:** Uso de matriz de confusão e F1-Score sobre 10% dos dados (amostra rotulada manualmente).

### Módulo D: Cálculo Financeiro e Estatístico
- **Log-Retorno Diário ($R_t$):** $\ln(Preço_t / Preço_{t-1})$.
- **Índice de Sentimento Diário (ISD):** $(Notícias Positivas - Notícias Negativas) / Total de Notícias$.
- **Testes:** Correlação de Pearson e Regressão Linear Simples.

## 5. Instruções Críticas para a IA
1. **Datas:** Converter sempre as strings de data do portal (DD/MM/AAAA) para objetos `datetime` para garantir integridade nos filtros e no join com os dados financeiros.
2. **Sincronia:** Notícias publicadas após o fechamento do mercado (18h) devem ser correlacionadas com o retorno do dia útil seguinte.
3. **Persistência:** Salvar os resultados intermediários em CSV para evitar chamadas de rede repetitivas.
4. **Modularização:** Separar claramente as funções de `scraping.py`, `nlp_processing.py`, `ml_models.py` e `stats_analysis.py`.

---

## 6. Resultados Preliminares (Fonte: InfoMoney — jan/2025 a dez/2025)

> Documento de referência: `Docs/[Resultados preliminares] - André Baconcelo Prado Furlanetti.docx`
> Autores: André Baconcelo Prado Furlanetti; Marisa Gomes da Costa (orientadora)

### 6.1 Composição do Corpus

| Etapa | Quantidade |
|-------|-----------|
| Registros brutos coletados | 32.538 |
| Manchetes únicas (após deduplicação) | 16.234 |
| Manchetes após filtragem temática | 5.909 |
| Dias com publicações | 261 |
| Média diária de notícias | 22,6 (±7,2) |
| Período de cobertura | 01/jan/2025 a 30/dez/2025 |

- Filtro temático: 30 palavras-chave macroeconômicas/bursáteis (ex.: "Ibovespa", "Bolsa", "Dólar", "Inflação", "Selic").
- Optou-se por usar apenas **manchetes** (headlines), descartando o corpo da notícia (Schmitz et al., 2022).

### 6.2 Validação do Modelo FinBERT-PT-BR

Auditoria manual em amostra aleatória estratificada de **155 manchetes**:

| Classe | Precision | Recall | F1-Score | Suporte |
|--------|-----------|--------|----------|---------|
| Negativo | 0,68 | **0,93** | 0,79 | 54 |
| Neutro | 0,80 | 0,63 | 0,70 | 51 |
| Positivo | 0,88 | 0,74 | 0,80 | 50 |
| **Macro Avg** | 0,79 | 0,76 | **0,77** | 155 |
| **Acurácia Global** | — | — | **0,7677** | 155 |

- Recall de 93% na classe Negativa → alta sensibilidade a notícias adversas (relevante para gestão de risco).
- Principal fonte de erro: manchetes neutras classificadas como negativas (15 casos) — viés do tom cauteloso do jornalismo financeiro.

### 6.3 Estatísticas Descritivas do Sentimento

| Métrica | Valor |
|---------|-------|
| Total de manchetes analisadas | 5.909 |
| Sentimento médio geral | **-0,2194** |
| Desvio padrão | 0,5871 |
| Mínimo | -0,9154 |
| Máximo | +0,9149 |
| Negativas | 2.752 (46,6%) |
| Neutras | 1.834 (31,0%) |
| Positivas | 1.323 (22,4%) |

- Sentimento diário agregado: oscilou entre -0,7931 e +0,4186 (mediana = -0,2296).
- Viés negativo persistente ao longo de todo o período, com episódios pontuais de inversão positiva.

### 6.4 Testes de Estacionariedade (ADF)

| Série | Estatística ADF | p-valor | Resultado |
|-------|----------------|---------|-----------|
| Retorno BOVA11 (Rₜ) | -16,9503 | < 0,0001 | Estacionária |
| Sentimento Médio (Sₜ) | -5,1679 | 0,00001 | Estacionária |

- Valores críticos a 1%: -3,457 (Rₜ); -3,458 (Sₜ). Ambas as séries validadas para correlação de Pearson e OLS.

### 6.5 Análise de Correlação de Pearson

| Lag (dias) | r | p-valor | Sig. |
|-----------|---|---------|------|
| 0 (contemporâneo) | **+0,2529** | 0,0001 | *** |
| 1 | +0,0333 | 0,6022 | n.s. |
| 2 | -0,0437 | 0,4939 | n.s. |
| 3–5 | ≈ 0 | n.d. | n.s. |

- Correlação significativa **apenas** na relação contemporânea (lag = 0).
- Correlação móvel (janela 30 dias): média +0,2787, mediana +0,2966, positiva em **99,6%** dos dias.

### 6.6 Regressão OLS

| Variável | Modelo 1 (Simples) | Modelo 2 (Múltiplo) |
|----------|-------------------|---------------------|
| Intercepto (α) | +0,0048*** | +0,0036** |
| Sentimento (t) | **+0,0161***` | **+0,0175***` |
| Sentimento (t-1) | — | -0,0012 (n.s.) |
| Sentimento (t-2) | — | -0,0057 (n.s.) |
| R² | **0,0640** | 0,0727 |
| R² Ajustado | 0,0601 | 0,0612 |
| F-statistic | 16,74*** | 6,35*** |
| Durbin-Watson | 2,294 | 2,330 |
| Observações | 247 | 247 |

- Um aumento de 1 unidade no score de sentimento → +1,61% no retorno logarítmico diário do BOVA11.
- Lags (t-1, t-2) não significativos; relação é essencialmente **contemporânea**.
- Durbin-Watson ≈ 2,3: ausência de autocorrelação serial nos resíduos.

### 6.7 Causalidade de Granger

| Direção | Lags testados | Resultado |
|---------|--------------|-----------|
| Sentimento → Retorno | 1–5 | **Não significativo** (p > 0,39) |
| Retorno → Sentimento | 1–4 | **Significativo** (p < 0,01) |
| Retorno → Sentimento | 5 | **Significativo** (p = 0,020) |

- **Interpretação:** O mercado influencia a narrativa jornalística (*media feedback hypothesis*, Tetlock 2007), e não o inverso.
- Consistente com a Hipótese de Mercados Eficientes (Fama, 1970) em sua forma semiforte.

### 6.8 Conclusões e Próximos Passos

- O sentimento é um **indicador coincidente** (não antecedente) do mercado com a fonte atual.
- **Próximo passo:** Expandir a coleta para, pelo menos, mais uma fonte de notícias financeiras de alta relevância.
- Hipótese: diversidade editorial poderá mitigar o viés reativo e revelar sinais antecedentes de volatilidade.
- Reavaliação da causalidade de Granger com maior volume de dados na versão final do TCC.