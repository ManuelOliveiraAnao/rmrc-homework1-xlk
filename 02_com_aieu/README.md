# Modelos com AI-EU

Esta pasta funciona de forma independente. Todos os modelos aqui executados
incluem informação do AI-EU. Não estima o modelo sem índice.

- `codigo/analise.py`: executa as três variantes com o índice.
- `codigo/estimacao.py`: preparação do AI-EU e estimação.
- `dados/`: dados financeiros, earnings e snapshot oficial do AI-EU.
- `resultados/ai_no_retorno/`: mudança do AI-EU na equação do retorno.
- `resultados/ai_na_variancia/`: magnitude da mudança na variância.
- `resultados/ai_em_ambos/`: as duas componentes adicionais.
- `resultados/qualidade_dados/`: dados unidos, fonte e estacionariedade.
- `graficos/`: figuras individuais de cada variante e série AI-EU.

Executar dentro desta pasta:
```
pip install -r requirements.txt
python codigo/analise.py
```
São usadas mudanças de log(1+AI-EU), com atraso de um dia de calendário,
e a mesma amostra comum de 2 971 datas da outra versão. As escalas do
período reservado são estimadas apenas nos primeiros 80% dos dados.
Fonte: https://www.policyuncertainty.com/aieu_daily.html (CC BY 4.0).
O índice é sujeito a revisões. A magnitude das mudanças rejeita KPSS:
a extensão na variância é exploratória. São mantidas as classificações
de earnings originais e a avaliação é condicional ao mercado e VIX
observados; não constitui um backtest de trading em tempo real.
