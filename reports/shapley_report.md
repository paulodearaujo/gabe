# Relatório de Atribuição pelo Valor de Shapley

## Visão Geral
- Total de jornadas analisadas: **13,357**
- Jornadas convertidas: **13,357**
- Taxa de conversão: **100.00%**
- Toques médios por jornada: **1.11**
- Toques médios (conversões): **1.11**

## Atribuição por Canal (Top N)
| Canal | Valor Shapley | Conversões Atribuídas |
| --- | ---: | ---: |
| `/blog/cobranca` | 0.181818 | 607.14 |
| `/blog/sistema-pdv-gratuito` | 0.181818 | 607.14 |
| `/gestao-de-cobranca-2` | 0.181818 | 607.14 |
| `/jim` | 0.181818 | 607.14 |
| `/loja-online` | 0.181818 | 607.14 |
| `/maquina-cartao` | 0.181818 | 607.14 |
| `/maquininha` | 0.181818 | 607.14 |
| `/maquininha-celular` | 0.181818 | 607.14 |
| `/materiais` | 0.181818 | 607.14 |
| `/pdv` | 0.181818 | 607.14 |
| `/taxas` | 0.181818 | 607.14 |
| `/jc512151/vc1d-1gflntn2lh-5000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-1zom5k4ysp-5000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-1zycyiutbj-2300,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-39ckfo8ezz-7000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-39gpfgvsfz-21700,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-39ooji7sml-21700,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-3a3gctunol-5000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-6icksgbetj-23000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-6jeaypto8d-10000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-hvcms0x6l-7000,00` | 0.166667 | 556.54 |
| `/jc512151/vc1d-ljl9gtkmz-4500,00` | 0.166667 | 556.54 |
| `/legal/termos-de-uso` | 0.166667 | 556.54 |
| `/abrilino/6kmnedxi2z` | 0.000000 | 0.00 |
| `/abrilino/vc1dltetsq-26vqh0ocwh-5000,00` | 0.000000 | 0.00 |

## Tabela Completa
Tabela detalhada com os canais ordenados por conversões atribuídas.
| Canal                                   |   Valor Shapley |   Conversões Atribuídas |
|:----------------------------------------|----------------:|------------------------:|
| /blog/cobranca                          |        0.181818 |                  607.14 |
| /blog/sistema-pdv-gratuito              |        0.181818 |                  607.14 |
| /gestao-de-cobranca-2                   |        0.181818 |                  607.14 |
| /jim                                    |        0.181818 |                  607.14 |
| /loja-online                            |        0.181818 |                  607.14 |
| /maquina-cartao                         |        0.181818 |                  607.14 |
| /maquininha                             |        0.181818 |                  607.14 |
| /maquininha-celular                     |        0.181818 |                  607.14 |
| /materiais                              |        0.181818 |                  607.14 |
| /pdv                                    |        0.181818 |                  607.14 |
| /taxas                                  |        0.181818 |                  607.14 |
| /jc512151/vc1d-1gflntn2lh-5000,00       |        0.166667 |                  556.54 |
| /jc512151/vc1d-1zom5k4ysp-5000,00       |        0.166667 |                  556.54 |
| /jc512151/vc1d-1zycyiutbj-2300,00       |        0.166667 |                  556.54 |
| /jc512151/vc1d-39ckfo8ezz-7000,00       |        0.166667 |                  556.54 |
| /jc512151/vc1d-39gpfgvsfz-21700,00      |        0.166667 |                  556.54 |
| /jc512151/vc1d-39ooji7sml-21700,00      |        0.166667 |                  556.54 |
| /jc512151/vc1d-3a3gctunol-5000,00       |        0.166667 |                  556.54 |
| /jc512151/vc1d-6icksgbetj-23000,00      |        0.166667 |                  556.54 |
| /jc512151/vc1d-6jeaypto8d-10000,00      |        0.166667 |                  556.54 |
| /jc512151/vc1d-hvcms0x6l-7000,00        |        0.166667 |                  556.54 |
| /jc512151/vc1d-ljl9gtkmz-4500,00        |        0.166667 |                  556.54 |
| /legal/termos-de-uso                    |        0.166667 |                  556.54 |
| /abrilino/6kmnedxi2z                    |        0        |                    0    |
| /abrilino/vc1dltetsq-26vqh0ocwh-5000,00 |        0        |                    0    |

## Metodologia
1. Agrupamos jornadas por conjunto único de canais (ordem ignorada).
2. Para cada subconjunto de canais, estimamos a taxa de conversão observada.
3. Calculamos o valor de Shapley exato para conjuntos até o limite configurado.
4. Para conjuntos maiores, aplicamos uma aproximação de crédito igualitário.
5. Escalamos as contribuições positivas para distribuir todas as conversões observadas.