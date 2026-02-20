# 📊 MBA-TCC: Análise de Sentimento e Mercado

Este projeto realiza a coleta de notícias financeiras, análise de sentimento usando o modelo **FinBERT-PT-BR** e estuda a correlação desses sentimentos com os retornos do índice BOVA11.

## 🛠️ Configuração do Ambiente Python

Este projeto utiliza um ambiente virtual para gerenciar as dependências.

### Configuração Automática (Windows/PowerShell)

Execute o comando abaixo para criar/ativar o ambiente e instalar as dependências:

```powershell
.\setup_env.ps1
```

### Configuração Manual

1. **Criar o ambiente virtual:**
   ```powershell
   python -m venv .venv
   ```

2. **Ativar o ambiente:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Registrar Kernel do Jupyter (para arquivos .ipynb):**
   ```powershell
   python -m ipykernel install --user --name=mba-tcc --display-name "Python (mba-tcc)"
   ```

---

## 📅 Ordem de Execução dos Notebooks

Os notebooks estão localizados no diretório `notebooks/` e devem ser executados na seguinte ordem:

### 1️⃣ `01_news_scraper.ipynb`
**Objetivo**: Coletar notícias do InfoMoney via API para termos específicos.

**O que faz**:
- 🔍 Busca notícias por termos (ex: Itaú, Dólar, Petrobras, Vale).
- 📅 Filtra por período inicial definido.
- 💾 Salva resultados individuais e consolida em um único dataset CSV.

**Arquivos gerados**:
- `src/dataset/scraper/search/news_[termo]_[data].csv`
- `src/dataset/scraper/consolidated_news_[data].csv`

---

### 2️⃣ `02_sentiment_analysis.ipynb`
**Objetivo**: Processar as notícias coletadas e gerar scores de sentimento.

**O que faz**:
- ✅ Carrega o modelo FinBERT-PT-BR (com suporte a GPU).
- 🧪 Valida o modelo com exemplos manuais.
- 🤖 Processa o dataset consolidado de notícias.
- 📊 Gera análise exploratória (distribuição de sentimentos).
- 📅 Agrega sentimentos por data.

**Arquivos gerados**:
- `src/dataset/sentiment/news_with_sentiment.csv` - Notícias com scores individuais.
- `src/dataset/sentiment/daily_sentiment.csv` - Sentimento agregado por dia.

---

### 3️⃣ `03_sentiment_market_merge.ipynb`
**Objetivo**: Unificar os dados de sentimento com os retornos do BOVA11.

**O que faz**:
- 📂 Carrega o sentimento diário e os retornos do mercado.
- 🔄 Realiza merge inteligente por data (join entre notícias e pregões).
- 📈 Cria variáveis defasadas (lags t-1, t-2, t-3) para análise preditiva.
- 🔍 Análise de correlação e visualização de tendências.

**Arquivo gerado**:
- `src/dataset/final/sentiment_returns_merged.csv` - Dataset final pronto para modelagem estatística.

---

## 🚀 Como Executar

### Pré-requisitos

1. **Ative o ambiente virtual** (conforme seção de Setup).
2. **Dados de Mercado**: Certifique-se de ter o arquivo `src/dataset/market_data/BOVA11_log_returns_*.csv`. Caso não tenha, execute o script:
   ```powershell
   python src/cotation/calculate_log_returns.py
   ```

### Executando os Notebooks

- **VS Code**: Abra os notebooks na pasta `notebooks/`, selecione o kernel `Python (mba-tcc)` e execute as células.
- **Jupyter**: No terminal, execute `jupyter lab` ou `jupyter notebook`.

---

## 📊 Estrutura de Pastas de Dados

```
src/dataset/
├── scraper/          # Notícias brutas e consolidadas
├── market_data/      # Dados históricos do BOVA11 e retornos
├── sentiment/        # Sentimentos processados e agregados
└── final/            # Dataset final unificado
```

---

## ⚙️ Configurações e Solução de Problemas

### GPU e Performance
O processamento de sentimento é pesado. Se possuir uma GPU NVIDIA, o código a utilizará automaticamente via CUDA. Ajuste o `batch_size` no notebook `02_sentiment_analysis.ipynb` (ex: 32, 64) conforme sua VRAM.

### Erro "CUDA out of memory"
Reduza o `batch_size` (ex: 8 ou 16) caso receba este erro durante a análise de sentimento.

---

## 📚 Referências

- **Modelo**: [lucas-adrian/FinBERT-PT-BR](https://huggingface.co/lucas-adrian/FinBERT-PT-BR)
- **Paper**: Santos et al. (2023) - FinBERT-PT-BR
