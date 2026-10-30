# Modelo sem AI-EU

Esta pasta funciona de forma independente e não necessita do ficheiro AI-EU.

- `codigo/analise.py`: modelo sem índice na amostra comum de 2 971 datas.
- `codigo/estimacao.py`: estimação sem parâmetros do índice.
- `codigo/analise_original.py`: pipeline completo original; apenas os caminhos
  foram adaptados à nova organização. Usa 2 972 observações.
- `dados/`: preços, retornos e earnings originais.
- `resultados/amostra_comum/`: coeficientes, série, diagnósticos e avaliação
  do modelo sem índice para comparar com a outra versão.
- `resultados/amostra_original/`: todas as tabelas e logs do trabalho original.
- `graficos/`: figuras separadas pelas duas amostras.

Executar dentro desta pasta:
```
pip install -r requirements.txt
python codigo/analise.py
```
Para repetir o pipeline original completo: `python codigo/analise_original.py`.
Datas da amostra comum: 05/01/2015 a 27/10/2026. Na avaliação reservada, os
primeiros 2 376 dias servem para estimar e os últimos 595 para avaliar.
Esta avaliação é condicional ao mercado e VIX observados, com classificações
de earnings herdadas; não é uma previsão de trading em tempo real.
