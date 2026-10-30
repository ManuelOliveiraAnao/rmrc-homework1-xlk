# Comparação das duas versões

As tabelas e os gráficos que misturam os modelos ficam exclusivamente aqui.
`codigo/comparar.py` lê os resultados das pastas 01 e 02, sem voltar a estimar.
Executar depois de executar os códigos individuais:
```
python codigo/comparar.py
```
Todos os modelos são comparados na amostra comum de 2 971 observações.
A QLIKE usa o mesmo proxy de variância para todos: os erros ao quadrado do
modelo sem AI-EU no período reservado. As perdas não são testes de trading.
`METODOLOGIA_ORIGINAL.md` preserva a documentação da análise anterior;
os caminhos ali mencionados referem-se ao pacote anterior. Os caminhos
atuais estão nos README de cada pasta.
As tabelas OLS/LR são complementares da execução anterior; o comparador
reconstrói AIC/BIC, perdas e gráficos a partir dos resultados separados.
