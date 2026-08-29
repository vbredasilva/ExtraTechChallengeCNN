# Tech Challenge – Fase 1
## Visão Computacional para Classificação de Câncer de Mama em Mamografias

Este projeto foi desenvolvido como parte do **Tech Challenge – Fase 1** da pós-graduação em Inteligência Artificial.

O objetivo é aplicar conceitos de **Machine Learning, Deep Learning e Visão Computacional** na área da saúde da mulher, utilizando uma **Rede Neural Convolucional (CNN)** para classificar mamografias como **benignas** ou **malignas**.

> **Importante:** este projeto possui finalidade acadêmica e experimental. O modelo não deve ser utilizado como ferramenta autônoma de diagnóstico médico. Em um cenário real, uma solução desse tipo deve atuar apenas como apoio à decisão, mantendo o profissional de saúde como responsável pelo diagnóstico final.

---

## 1. Problema

O problema abordado é a classificação de mamografias contendo alterações mamárias.

A tarefa é um problema de:

- **Aprendizado supervisionado**, pois cada imagem possui um rótulo conhecido;
- **Classificação**, pois o objetivo é escolher entre categorias;
- **Classificação binária**, pois existem duas classes finais:

```text
0 = BENIGNO
1 = MALIGNO
```

A variável-alvo é construída a partir da coluna `pathology` do dataset.

Os registros `BENIGN` e `BENIGN_WITHOUT_CALLBACK` são agrupados na classe benigna, enquanto `MALIGNANT` representa a classe maligna.

---

## 2. Dataset

Foi utilizado o dataset público **CBIS-DDSM: Breast Cancer Image Dataset**.

Kaggle:

```text
https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
```

O dataset contém mamografias, informações de massas e calcificações, metadados dos exames, diagnóstico anatomopatológico e uma divisão oficial entre treino e teste.

Os principais arquivos utilizados são:

```text
mass_case_description_train_set.csv
mass_case_description_test_set.csv
calc_case_description_train_set.csv
calc_case_description_test_set.csv
dicom_info.csv
```

O arquivo `dicom_info.csv` auxilia no relacionamento entre os registros dos CSVs e as respectivas imagens JPEG.

---

## 3. Tecnologias utilizadas

O projeto foi desenvolvido em **Python**, utilizando principalmente:

- Google Colab;
- TensorFlow;
- Keras;
- Scikit-learn;
- Pandas;
- NumPy;
- Matplotlib;
- Pillow;
- KaggleHub.

O Google Colab foi escolhido pela facilidade de execução e pela possibilidade de utilização de GPU para acelerar o treinamento da rede neural.

---

## 4. Análise exploratória

Antes da construção da CNN, foi realizada uma análise exploratória dos dados.

Foram avaliados:

- distribuição entre diagnósticos benignos e malignos;
- distribuição entre massas e calcificações;
- lateralidade da mama;
- incidências mamográficas;
- densidade mamária;
- valores ausentes;
- estatísticas descritivas;
- correlação entre variáveis numéricas disponíveis nos metadados.

Após a limpeza inicial, foram encontrados:

```text
BENIGNO: 2111 registros
MALIGNO: 1457 registros
```

Foi observada uma quantidade maior de casos benignos, caracterizando um desbalanceamento moderado entre as classes.

A análise de correlação foi utilizada apenas como ferramenta exploratória. Os metadados analisados não são utilizados como entrada da CNN.

---

## 5. Preparação dos dados

O pré-processamento incluiu:

- padronização dos diagnósticos;
- conversão da variável-alvo para valores numéricos;
- remoção de registros sem diagnóstico válido;
- associação entre registros dos CSVs e mamografias JPEG;
- tratamento de imagens repetidas;
- verificação física da existência das imagens;
- separação entre treino, validação e teste;
- redimensionamento das imagens;
- normalização;
- criação de pipelines com `tf.data`.

Uma mesma mamografia pode aparecer em mais de um registro devido à existência de múltiplas anormalidades anotadas. Para evitar duplicidade, os registros são agrupados pelo identificador da mamografia.

Caso pelo menos uma anormalidade associada à imagem seja maligna, a mamografia recebe o rótulo maligno.

---

## 6. Separação entre treino, validação e teste

O CBIS-DDSM já fornece uma divisão oficial entre treinamento e teste, e essa divisão foi preservada.

Uma parte do conjunto oficial de treinamento foi separada para validação. A validação foi criada **por paciente**, e não apenas por imagem, reduzindo o risco de **data leakage**.

```text
Treino
→ utilizado para ajustar os pesos da CNN.

Validação
→ utilizado para acompanhar o desempenho durante o treinamento.

Teste
→ utilizado somente ao final para avaliar o modelo em dados não vistos.
```

---

## 7. Pré-processamento das imagens

As mamografias são preparadas antes de serem enviadas à rede neural.

O pipeline realiza:

1. leitura do arquivo JPEG;
2. conversão para escala de cinza;
3. redimensionamento para `256 x 256`;
4. preservação da proporção da imagem;
5. conversão dos pixels para valores numéricos adequados;
6. normalização;
7. criação de batches;
8. utilização de `prefetch`.

Configuração utilizada:

```python
IMG_SIZE = 256
BATCH_SIZE = 16
```

As imagens são processadas em lotes de 16 exemplos para melhor utilização da memória e da GPU.

---

## 8. Data Augmentation

Durante o treinamento é utilizada a técnica de **Data Augmentation**.

São aplicadas pequenas transformações aleatórias, como:

- espelhamento horizontal;
- pequenas rotações;
- pequeno zoom;
- pequenas translações.

O objetivo é apresentar versões ligeiramente diferentes das imagens para a CNN e reduzir o risco de **overfitting**.

Como se trata de imagens médicas, foram utilizadas transformações moderadas para preservar a estrutura anatômica.

---

## 9. Tratamento do desbalanceamento

Como existe diferença na quantidade de exemplos benignos e malignos, são utilizados pesos de classe por meio de:

```python
compute_class_weight(class_weight="balanced")
```

O `class_weight` não cria nem remove imagens. Ele modifica a importância dos erros durante o treinamento, atribuindo maior peso à classe menos frequente.

---

## 10. Rede Neural Convolucional

Foi construída uma **CNN do zero utilizando TensorFlow/Keras**.

A entrada da rede possui o formato:

```text
256 x 256 x 1
```

O último valor é `1` porque as mamografias são processadas em escala de cinza.

### Arquitetura

```text
Mamografia 256 x 256
        ↓
Data Augmentation
        ↓
Conv2D - 32 filtros - ReLU
BatchNormalization
MaxPooling2D
        ↓
Conv2D - 64 filtros - ReLU
BatchNormalization
MaxPooling2D
        ↓
Conv2D - 128 filtros - ReLU
BatchNormalization
MaxPooling2D
        ↓
Conv2D - 256 filtros - ReLU
BatchNormalization
MaxPooling2D
        ↓
GlobalAveragePooling2D
        ↓
Dropout 40%
        ↓
Dense 128 - ReLU
        ↓
Dropout 30%
        ↓
Dense 1 - Sigmoid
        ↓
Probabilidade de malignidade
```

### Principais componentes

**Conv2D:** aprende características visuais como bordas, contrastes, texturas e padrões mais complexos.

**ReLU:** função de ativação utilizada nas camadas internas para introduzir não linearidade.

**BatchNormalization:** ajuda a estabilizar as ativações internas da rede durante o treinamento.

**MaxPooling2D:** reduz progressivamente as dimensões espaciais dos mapas de características.

**GlobalAveragePooling2D:** resume os mapas de características e produz uma representação mais compacta.

**Dropout:** desativa aleatoriamente parte das ativações durante o treinamento para reduzir overfitting.

**Dense:** combina as características aprendidas antes da classificação final.

**Sigmoid:** função de saída que produz um valor entre `0` e `1`, interpretado como probabilidade de malignidade.

O limiar inicialmente utilizado é:

```text
probabilidade < 0.50  → BENIGNO
probabilidade >= 0.50 → MALIGNO
```

---

## 11. Compilação do modelo

A função de perda escolhida foi:

```python
binary_crossentropy
```

Ela é apropriada para problemas de classificação binária.

O otimizador utilizado foi o **Adam**:

```python
Adam(learning_rate=0.001)
```

As métricas acompanhadas durante o treinamento são:

- Accuracy;
- Recall;
- Precision;
- AUC.

---

## 12. Callbacks

Foram utilizados três callbacks:

### EarlyStopping
Interrompe o treinamento caso o modelo deixe de melhorar durante determinado número de épocas.

### ReduceLROnPlateau
Reduz a taxa de aprendizagem quando a perda de validação deixa de melhorar.

### ModelCheckpoint
Salva automaticamente o melhor modelo encontrado durante o treinamento.

---

## 13. Treinamento

O treinamento é realizado utilizando:

```python
model.fit()
```

Cada passagem completa pelo conjunto de treinamento corresponde a uma **época**.

Durante uma época, a CNN recebe as imagens, realiza previsões, compara as previsões com os rótulos verdadeiros, calcula o erro e atualiza seus pesos.

Para testes rápidos do pipeline podem ser utilizadas poucas épocas. Para o experimento final recomenda-se utilizar um número maior de épocas em conjunto com `EarlyStopping`.

---

## 14. Curvas de aprendizagem

Após o treinamento são analisadas curvas de:

- Loss;
- AUC;
- Recall.

Essas curvas permitem comparar treino e validação e ajudam a identificar:

- **overfitting**: bom desempenho no treino e piora na validação;
- **underfitting**: desempenho ruim em ambos;
- **boa generalização**: evolução consistente entre treino e validação.

---

## 15. Métricas de avaliação

O modelo é avaliado no conjunto de teste utilizando diferentes métricas.

### Accuracy
Proporção total de previsões corretas.

### Precision
Entre os exames classificados como malignos, indica quantos realmente eram malignos.

### Recall / Sensibilidade
Entre todos os casos realmente malignos, indica quantos foram corretamente detectados.

Neste problema, o Recall merece atenção especial porque um falso negativo representa um caso maligno classificado como benigno.

### F1-score
Combina Precision e Recall em uma única métrica.

### Specificity
Mede a capacidade do modelo de reconhecer corretamente os casos benignos.

### ROC-AUC
Avalia a capacidade do modelo de separar casos benignos e malignos considerando diferentes limiares de decisão.

### Average Precision
Resume o comportamento da relação entre Precision e Recall.

---

## 16. Matriz de confusão

A matriz de confusão apresenta:

```text
TN → benigno corretamente classificado como benigno
FP → benigno classificado incorretamente como maligno
FN → maligno classificado incorretamente como benigno
TP → maligno corretamente classificado como maligno
```

Os falsos negativos são particularmente importantes neste problema, pois representam casos malignos que não foram corretamente identificados pelo modelo.

---

## 17. Curva ROC

A curva ROC relaciona a taxa de verdadeiros positivos com a taxa de falsos positivos e permite avaliar o comportamento do classificador em diferentes limiares.

Uma ROC-AUC próxima de `0.5` indica desempenho semelhante ao acaso, enquanto valores próximos de `1.0` representam maior capacidade de separação entre as classes.

---

## 18. Curva Precision-Recall

A curva Precision-Recall permite analisar o compromisso entre detectar uma quantidade maior de casos malignos e manter maior confiabilidade nas previsões positivas.

Ela é especialmente útil quando existe desbalanceamento entre as classes.

---

## 19. Análise dos falsos negativos

Os falsos negativos são analisados separadamente.

Um falso negativo ocorre quando:

```text
Diagnóstico real = MALIGNO
Previsão da CNN = BENIGNO
```

Esse tipo de erro é particularmente crítico no contexto do problema e deve ser considerado durante a avaliação do modelo.

---

## 20. Explicabilidade

O projeto também considera o uso de **Grad-CAM** como técnica de explicabilidade para a CNN.

O Grad-CAM utiliza informações das camadas convolucionais para criar mapas de calor que indicam quais regiões da imagem tiveram maior influência na previsão do modelo.

O mapa não representa necessariamente a localização clínica de uma lesão. Ele mostra apenas quais regiões tiveram maior influência matemática na decisão da rede neural.

---

## 21. Resultados preliminares

Durante uma execução rápida de teste utilizando **5 épocas**, foram obtidos os seguintes resultados no conjunto de teste:

```text
Accuracy:               0.5689
Precision:              0.4850
Recall / Sensibilidade: 0.5843
F1-score:               0.5301
Specificity:            0.5579
ROC-AUC:                0.5952
Average Precision:      0.4954
```

A matriz de confusão apresentou:

```text
                PREVISTO
              Benigno   Maligno

REAL Benigno     130       103
REAL Maligno      69        97
```

Foram observados **69 falsos negativos**.

Esses resultados são considerados **preliminares**, pois foram obtidos em uma execução curta utilizada principalmente para validar o funcionamento do pipeline.

Os resultados finais devem ser atualizados após o treinamento definitivo.

---

## 22. Limitações

Entre as principais limitações do experimento estão:

- redução das mamografias para `256 x 256`, podendo eliminar detalhes pequenos;
- utilização de apenas um dataset;
- possibilidade de diferenças entre a distribuição do dataset e dados de hospitais reais;
- ausência de validação externa;
- classificação realizada no nível da imagem;
- ausência de localização formal da lesão por bounding box;
- necessidade de análise mais aprofundada dos falsos negativos;
- necessidade de validação por especialistas antes de qualquer uso clínico.

---

## 23. Uso em cenário real

O modelo desenvolvido não está pronto para utilização clínica.

Antes de uma aplicação real seriam necessários:

- validação externa;
- avaliação por radiologistas;
- análise de vieses;
- avaliação em diferentes populações;
- testes em imagens de resolução clínica;
- definição adequada do limiar de decisão;
- calibração das probabilidades;
- estudos de segurança;
- análise regulatória;
- integração controlada ao fluxo assistencial.

Uma possível aplicação futura seria como **ferramenta de apoio à decisão ou priorização de exames**, nunca como substituto do profissional médico.

---

## 24. Como executar

O projeto foi desenvolvido para execução no **Google Colab**.

### Passos

1. Abra o notebook `.ipynb` no Google Colab.
2. Ative a GPU em:

```text
Ambiente de execução
→ Alterar tipo de ambiente de execução
→ Acelerador de hardware
→ GPU
```

3. Execute as células sequencialmente.
4. O dataset será baixado por meio do KaggleHub.
5. Aguarde o processamento e o treinamento da CNN.
6. Ao final, analise as métricas, gráficos e matriz de confusão.

---

## 25. Estrutura sugerida do repositório

```text
/
├── README.md
├── TechChallenge_EXTRA_CNN_CBIS_DDSM_Colab.ipynb
├── resultados/
│   ├── matriz_confusao.png
│   ├── curva_roc.png
│   ├── curva_precision_recall.png
│   ├── evolucao_loss.png
│   ├── evolucao_auc.png
│   └── evolucao_recall.png
```

O dataset não precisa necessariamente ser armazenado no repositório devido ao seu tamanho. O link oficial para download pode ser utilizado.

---

## 26. Conclusão

O projeto apresenta uma aplicação completa de Visão Computacional para classificação de mamografias utilizando uma Rede Neural Convolucional.

O fluxo desenvolvido contempla:

```text
Dataset
↓
Exploração dos dados
↓
Limpeza
↓
Associação entre imagem e diagnóstico
↓
Pré-processamento
↓
Separação treino / validação / teste
↓
Data Augmentation
↓
CNN
↓
Treinamento
↓
Avaliação
↓
Análise dos erros
↓
Explicabilidade
```

A solução demonstra conceitos importantes de Deep Learning, incluindo convoluções, pooling, funções de ativação, regularização, classificação binária e avaliação por diferentes métricas.

O desempenho do modelo deve ser analisado de forma crítica, com atenção especial ao Recall e aos falsos negativos.

Este trabalho demonstra a viabilidade técnica de utilização de CNNs para análise de mamografias em um contexto acadêmico, mas não representa um sistema validado para diagnóstico médico.
