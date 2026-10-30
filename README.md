# RMRC Homework I — XLK, earnings e AI-EU

Análise dos retornos e da volatilidade do ETF XLK com CAPM e
GJR-GARCH(1,1)-M. Os anúncios de resultados selecionados são os de Apple
(AAPL), Microsoft (MSFT) e Nvidia (NVDA), controlando pelo S&P 500 e VIX.
A extensão testa mudanças no índice de cobertura de inteligência artificial,
economia e incerteza [AI-EU](https://www.policyuncertainty.com/aieu_daily.html).

## Organização

| Pasta | Código e resultados |
|---|---|
| [`01_sem_aieu/`](01_sem_aieu/README.md) | Modelo sem AI-EU; inclui o trabalho original e a estimação na amostra comum |
| [`02_com_aieu/`](02_com_aieu/README.md) | Modelos com AI-EU no retorno, na variância e em ambos |
| [`03_comparacao/`](03_comparacao/README.md) | Tabelas e gráficos comparativos; lê os resultados das duas versões |

Cada versão tem as suas próprias pastas `codigo/`, `dados/`, `resultados/`
e `graficos/`. As pastas 01 e 02 podem ser executadas de forma independente.
Os ficheiros do trabalho original foram preservados dentro de `01_sem_aieu/`.

## Executar a comparação

A partir da raiz do repositório:

```bash
pip install -r requirements.txt
python 01_sem_aieu/codigo/analise.py
python 02_com_aieu/codigo/analise.py
python 03_comparacao/codigo/comparar.py
python verificar_organizacao.py
```

Para repetir o pipeline original completo, com as 2 972 observações:

```bash
python 01_sem_aieu/codigo/analise_original.py
```

Cada versão também tem o seu próprio `requirements.txt`. Os dados estão
guardados localmente; a extensão utiliza o snapshot AI-EU incluído, sem o
substituir por um download mais recente durante a execução.

## Amostras e interpretação

- Trabalho original: 02/01/2015 a 27/10/2026, 2 972 observações diárias.
- Comparação: 05/01/2015 a 27/10/2026, as mesmas 2 971 observações nos quatro modelos.
- Estimação da avaliação reservada: primeiras 2 376 observações.
- Período reservado: últimas 595 observações, de 13/06/2024 a 27/10/2026.

O AI-EU entra como mudança de `log(1 + índice)`, com atraso de um dia de
calendário; na equação da variância usa-se a magnitude da mudança.
O nível do índice apresenta sinais de não estacionariedade. A magnitude das
mudanças também rejeita KPSS, pelo que a extensão na variância é exploratória.

Nas versões testadas, a inclusão do AI-EU não apresenta uma melhoria
convincente: o BIC favorece o modelo original, a vantagem de AIC na variância
é pequena e não se traduz em menor perda de variância no período reservado.
As diferenças de perdas são descritivas, sem um teste da sua significância.

A avaliação é condicional aos valores contemporâneos observados do S&P 500
e VIX e mantém as classificações de earnings do projeto original, que usam
medianas da amostra completa. O AI-EU está sujeito a revisões. Estes resultados
não demonstram causalidade nem constituem um backtest de trading em tempo real.

Os README individuais descrevem os modelos e os ficheiros. A metodologia
da extensão está documentada em
[`03_comparacao/METODOLOGIA_ORIGINAL.md`](03_comparacao/METODOLOGIA_ORIGINAL.md),
com uma nota sobre os caminhos do pacote anterior.

## Fontes

- Preços e earnings: Yahoo Finance, conforme o projeto original.
- AI-EU: [Economic Policy Uncertainty](https://www.policyuncertainty.com/aieu_daily.html),
  série `daily_aieu_index`, snapshot de 30/10/2026. Licença CC BY 4.0.
- Verificações: [`validacao_organizacao.json`](validacao_organizacao.json)
  e `python verificar_organizacao.py`.
