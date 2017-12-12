O objetivo deste trabalho é implementar uma rede neuronal capaz de classificar digitos do MNIST dataset.

Foi implementada uma rede neuronal com três camadas:

1. Camada de entrada: cada unidade representa uma dimensão do dado de entrada.

2. Camada oculta: cada unidade representa uma transformação a partir das unidades de entrada.

3. Camada de saída: cada unidade representa a chance da saída correspondente ser a correta.

Foi utilizada a função de a ativação Sigmóide para obter não-linearidade. Além disso, a função de perda a ser minimizada é a log-loss.

O base de dados utilizada trata-se de 5000 entradas, onde cada entrada refere-se a um dígito escrito manualmente (i.e., MNIST dataset). Dessa forma, m=5000 e K=10. Cada entrada é dada por uma matriz de dimensões 28 por 28, ou seja, um vetor de 784 dimensões. A primeira coluna do arquivo sempre é o rótulo do dígito correto.

A rede neuronal a implementada tem 784 unidades de entrada e 10 unidades de saída. O número de unidades na camada oculta (25, 50, 100) é variável.

Além disso, foram comparados os seguintes algoritmos de cálculo de gradiente:

1. Gradient Descent: o gradiente é calculado após cada época (após as 5000 entradas serem processadas).

2. Stochastic Gradient Descent: o gradiente é calculado após cada entrada.

3. Mini-Batch: o gradiente é calculado após um certo número de entradas (considere 10 e 50).

Também foi variada a taxa de aprendizado: 0.5, 1, 10.