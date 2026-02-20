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