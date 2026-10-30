# XLK: comparação com o índice AI-EU

Extensão do projeto original. `analysis.py`, os dados financeiros, `results/`
e `figures/` foram preservados. O ZIP original não foi alterado.

## Pergunta e resultado

Acrescentar informação sobre a evolução da cobertura de IA, economia e
incerteza melhora o GJR-GARCH(1,1)-M com earnings na média e na variância?

Nas transformações e amostra testadas, não se encontrou uma melhoria
convincente. Na média, o coeficiente adicional é -0,00798 pontos percentuais
por um desvio-padrão de variação logarítmica do índice, com p robusto = 0,3949.
Na variância, a intensidade da variação tem coeficiente 0,002275, com p robusto
aproximado = 0,2210. A extensão da variância melhora o AIC apenas 1,05 pontos,
mas piora o BIC em 4,95 pontos e a perda QLIKE no período reservado em 0,62%.
As diferenças no período reservado são descritivas; não foi demonstrada a
sua significância estatística.

## Fonte e preparação

- Página: https://www.policyuncertainty.com/aieu_daily.html
- CSV oficial: https://www.policyuncertainty.com/media/All_Daily_AIEU_Data.csv
- Série escolhida: `daily_aieu_index`, interseção de IA, economia e incerteza.
- Os índices AI e AI-E não foram usados.
- A fonte mede frequência de artigos em jornais dos EUA, ajustada pelo total
  de artigos. Não é uma medida direta da opinião dos investidores nem um
  indicador de notícias exclusivamente negativas.
- Licença: CC BY 4.0. Atribuição: Economic Policy Uncertainty / policyuncertainty.com.
- Snapshot descarregado em 30/10/2026, com observações até 29/10/2026.
- Uma linha de 22/12/2025 estava duplicada com valores idênticos; foi removida
  antes de juntar. Não havia datas de calendário em falta.
- O ficheiro bruto está guardado em `data/All_Daily_AIEU_Data.csv`; hash e
  verificações estão em `results_aieu/source_and_checks.json`.

## Transformações e modelos

O nível AI-EU não passa os diagnósticos de estacionariedade: ADF p ≈ 0,506,
KPSS p <= 0,01. O logaritmo do nível também não resolve essa questão.
Por isso, esta extensão testa mudanças do índice, não o nível bruto:

```
c_t = log(1 + AI_EU_t) - log(1 + AI_EU_(t-1))
u_t = (c_(t-1) - media(c_lag)) / desvio_padrao(c_lag)
v_t = abs(c_(t-1)) / desvio_padrao(abs(c_lag))
```

O `+1` permite tratar os zeros genuínos, sem os confundir com dados em falta.
O atraso é de um dia de calendário: segunda-feira usa a mudança de domingo
face a sábado. Não é uma média de fim de semana. O atraso é aplicado antes
da junção com as datas de negociação. Dias sem história suficiente não são
preenchidos artificialmente. A mudança assinada passa ADF e KPSS; a magnitude
absoluta passa ADF mas rejeita KPSS, pelo que a extensão na variância deve
ser considerada exploratória, sujeita a instabilidade ao longo do período.

Equações (as restantes variáveis preservam a definição do trabalho original):

```
r_XLK,t = alpha + beta_SP*r_SP,t + beta_VIX*dVIX_t
          + d_good*GOOD_t + d_bad*BAD_t + lambda*sqrt(h_t)
          + theta*u_t + e_t
h_t = omega + (alpha_arch + gamma*I[e_(t-1)<0])*e_(t-1)^2
      + beta_garch*h_(t-1) + phi*NEWS_t + psi*v_t
```

Modelos comparados: original (theta=psi=0), theta livre, psi livre, ambos livres.
As estimações são QML gaussianas, com erros-padrão sandwich robustos e vários
pontos iniciais. As restrições garantem variâncias positivas: omega>0,
alpha_arch>=0, alpha_arch+gamma>=0, beta_garch>=0, phi>=0, psi>=0 e
alpha_arch+gamma/2+beta_garch<1. No código original phi podia ser negativo;
a restrição explícita não altera materialmente a solução positiva encontrada.
Os testes relativos a psi=0 têm uma hipótese nula na fronteira do espaço
de parâmetros: os p-values Wald e LR de referência são apenas aproximações.

## Comparação na mesma amostra

Todos os modelos usam 2 971 observações, de 05/01/2015 a 27/10/2026.
A observação de 02/01/2015 foi excluída de TODOS por não haver dois dias de
história AI-EU anteriores. Isto explica a pequena diferença dos valores
absolutos de AIC/BIC face ao resumo original com 2 972 observações.

| Modelo | AIC | BIC | Delta AIC | Delta BIC |
|---|---:|---:|---:|---:|
| Original reestimado | 4814,35 | 4880,31 | 0 | 0 |
| AI-EU no retorno | 4815,59 | 4887,55 | +1,25 | +7,24 |
| AI-EU na variância | 4813,30 | 4885,26 | -1,05 | +4,95 |
| AI-EU em ambos | 4814,43 | 4892,39 | +0,08 | +12,08 |

Menor AIC/BIC é preferível. O BIC favorece o original em todas as comparações.
A inclusão de AI-EU não resolve os efeitos ARCH residuais nem a rejeição
do RESET na regressão OLS. O efeito positivo de GOOD mantém-se.

## Avaliação condicional em período reservado

Primeiros 80% (2 376 observações, até 12/06/2024) para estimar; últimos 20%
(595 observações, de 13/06/2024 a 27/10/2026) para avaliar. Os parâmetros e as
escalas ficam fixos após a estimação. O estado da variância é atualizado com
os erros anteriores; o estado inicial usa apenas a amostra de treino.

| Modelo | Variação do MSE do retorno | Variação da perda QLIKE |
|---|---:|---:|
| AI-EU no retorno | -0,030% | +0,059% |
| AI-EU na variância | +0,214% | +0,623% |
| AI-EU em ambos | +0,186% | +0,697% |

Menor perda é preferível. QLIKE usa o MESMO proxy de variância realizada para
todos: o erro ao quadrado do modelo original no período reservado. O proxy
é ruidoso e depende da especificação da média do original. A perda é
`q/h - log(max(q,1e-12)/h) - 1`. Também são guardados os log scores gaussianos.

Esta avaliação NÃO é um backtest em tempo real: condiciona nos valores
observados contemporâneos do S&P 500 e VIX; mantém as classificações de
earnings do projeto original, que usam medianas de toda a amostra; e o
AI-EU está sujeito a revisões históricas (explicitamente nos últimos 30 dias).
O atraso reduz a possibilidade de usar notícias publicadas após o fecho,
mas não prova a disponibilidade da versão final do índice nesse momento.
Os resultados não demonstram causalidade nem capacidade de trading.

## Execução e ficheiros

```
pip install -r requirements_aieu.txt
python compare_aieu.py
python verify_aieu.py
```

- `results_aieu/model_comparison.csv`: AIC/BIC, diagnósticos e convergência.
- `results_aieu/coef_model_0.csv` a `coef_model_3.csv`: coeficientes e incerteza.
- `results_aieu/holdout_comparison.csv`: avaliação no período reservado.
- `results_aieu/stationarity.csv`: testes das transformações.
- `results_aieu/ols_comparison.csv`: comparação complementar OLS com HAC.
- `figures_aieu/`: gráficos da série, AIC/BIC, perdas e volatilidade.
- `comparison_run.log`: saída da execução realizada.

Verificações: retornos recalculados a partir dos preços originais com a mesma
regra de datas; datas sem duplicação após a junção; mesma amostra entre modelos;
convergência de várias inicializações; positividade das variâncias;
verosimilhança dos modelos aninhados não decrescente; AIC/BIC e perdas
recalculados independentemente; atraso conferido em datas de fronteira e
numa segunda-feira. Os gráficos foram abertos e inspecionados.
