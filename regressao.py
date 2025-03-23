import numpy as np
import matplotlib.pyplot as plt

## Questão 1: Plot de um gráfico de espalhamento (scatter plot) para visualizar a relação entre a variável independente e a variável dependente ##

# Caminho para o arquivo CSV
file_path = r"C:\Users\Bruno Matos\Desktop\IA_modelosderegressaoeclassificacao\dados\atividade_enzimatica.csv"

linhas = np.genfromtxt(file_path, delimiter='\n', dtype=str)

# Separa as strings por vírgulas e converte para float
dados = np.array([list(map(float, linha.split(','))) for linha in linhas])

# Coluna 0: Temperatura
# Coluna 1: pH
# Coluna 2: Atividade enzimática
temperatura = dados[:, 0]
ph = dados[:, 1]
atividade = dados[:, 2]

# Visualização inicial dos dados
plt.figure(figsize=(12, 5))

# Gráfico: Atividade enzimática vs Temperatura
plt.subplot(1, 2, 1)
plt.scatter(temperatura, atividade, color='blue', alpha=0.7)
plt.xlabel("Temperatura")
plt.ylabel("Atividade enzimática")
plt.title("Atividade enzimática vs Temperatura")

# Gráfico: Atividade enzimática vs pH
plt.subplot(1, 2, 2)
plt.scatter(ph, atividade, color='green', alpha=0.7)
plt.xlabel("pH")
plt.ylabel("Atividade enzimática")
plt.title("Atividade enzimática vs pH")

plt.tight_layout()
plt.show()

## Questão 2: Organizando os dados para a modelagem ##

N = dados.shape[0]

# Número de variáveis independentes (p). Aqui, p = 2 (Temperatura e pH).
p = 2

# Construindo a matriz de variáveis regressoras:
# Primeiro, criando uma matriz com as colunas temperaturas e pH.
X_regressoras = np.column_stack((temperatura, ph))  # dimensão: (N, 2)

# Em seguida, adicionando uma coluna de '1's para incluir o intercepto no modelo:
X = np.concatenate((np.ones((N, 1)), X_regressoras), axis=1)  # dimensão: (N, 3)

# Organizando a variável dependente 'Atividade enzimática' em um vetor coluna:
y = atividade.reshape(-1, 1)  # dimensão: (N, 1)

## Questão 3 e 4: Implementações dos modelos de regressão linear ##

# Método dos mínimos quadrados ordinários (MQO tradicional) #

B = np.linalg.pinv(X.T @ X) @ X.T @ y

print("Parâmetros estimados (MQO tradicional):")
print(B)

# MQO Regularizado (Tikhonov) #

# Definindo os valores de lambda a serem testados:
lambdas = [0, 0.25, 0.5, 0.75, 1]

# Cria um dicionário para armazenar as estimativas para cada valor de lambda.
beta_estimates = {}

# A solução regularizada é dada por: B = (X^T X + λ I)^(-1) X^T y
for lam in lambdas:
    reg_matrix = lam * np.eye(X.shape[1])  # Matriz identidade de tamanho (p+1) x (p+1)
    B_reg = np.linalg.inv(X.T @ X + reg_matrix) @ X.T @ y
    beta_estimates[f"λ = {lam}"] = B_reg

# Modelo de Média de valores observáveis #

# Neste modelo, o intercepto é igual à média dos valores observados de y e os coeficientes par as variáveis (Temperatura e pH) são definidos como zero, pois, a ideia é prever, para qualquer entrada, o valor médio observado da variável dependente (y)
beta_media = np.array([[np.mean(y)], [0.0], [0.0]])
beta_estimates["Média dos Valores Observáveis"] = beta_media

# Exibe as 6 estimativas do vetor β:
print("\nEstimativas dos coeficientes:")
for key, beta_val in beta_estimates.items():
    print(key)
    print(beta_val)
    print("---------------")

## Questão 5: Validação dos modelos ##

R = 500
n_train = int(0.8 * N)  # Considerando 80% dos dados para treinamento

# Inicializa, aqui, um dicionário para armazenar os valores de RSS para cada modelo
rss_results = {
    "Média dos Valores Observáveis": [],
    "MQO tradicional": [],
    "MQO regularizado (λ=0.25)": [],
    "MQO regularizado (λ=0.5)": [],
    "MQO regularizado (λ=0.75)": [],
    "MQO regularizado (λ=1)": []
}

# Loop de Monte Carlo
for r in range(R):
    # Embaralha os índices e dividir em treino/teste
    indices = np.random.permutation(N)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    X_train = X[train_idx, :]
    y_train = y[train_idx, :]
    X_test = X[test_idx, :]
    y_test = y[test_idx, :]
    
    # Modelo da Média: intercepto = mean(y_train) e coeficientes das variáveis = 0
    beta_media_sim = np.array([[np.mean(y_train)], [0.0], [0.0]])
    y_pred_media = X_test @ beta_media_sim
    rss_media = np.sum((y_test - y_pred_media)**2)
    rss_results["Média dos Valores Observáveis"].append(rss_media)
    
    # MQO Regularizado para cada valor de lambda (λ = 0 corresponde ao MQO tradicional)
    for lam in lambdas:
        reg_matrix = lam * np.eye(X_train.shape[1])
        beta_sim = np.linalg.inv(X_train.T @ X_train + reg_matrix) @ (X_train.T @ y_train)
        y_pred = X_test @ beta_sim
        rss_value = np.sum((y_test - y_pred)**2)
        if lam == 0:
            rss_results["MQO tradicional"].append(rss_value)
        elif lam == 0.25:
            rss_results["MQO regularizado (λ=0.25)"].append(rss_value)
        elif lam == 0.5:
            rss_results["MQO regularizado (λ=0.5)"].append(rss_value)
        elif lam == 0.75:
            rss_results["MQO regularizado (λ=0.75)"].append(rss_value)
        elif lam == 1:
            rss_results["MQO regularizado (λ=1)"].append(rss_value)

## Questão 6: Resumo dos Resultados ##

# Após as R rodadas, para cada modelo, calcula-se média, desvio-padrão, valor máximo e mínimo do RSS.
print("\nResumo dos Resultados (RSS - Residual Sum of Squares) para cada modelo em 500 simulações:")
header = "{:<40} {:>10} {:>10} {:>15} {:>15}".format("Modelo", "Média", "Std", "Maior Valor", "Menor Valor")
print(header)
print("-" * len(header))
for model, rss_list in rss_results.items():
    rss_arr = np.array(rss_list)
    media = np.mean(rss_arr)
    std = np.std(rss_arr)
    max_val = np.max(rss_arr)
    min_val = np.min(rss_arr)
    linha = "{:<40} {:10.4f} {:10.4f} {:15.4f} {:15.4f}".format(model, media, std, max_val, min_val)
    print(linha)

############## RESULTADOS E DISCURSÕES ################

# Resultados Observados:
# ---------------------------------------------------------------------------
# Modelo                                        Média        Std     Maior Valor     Menor Valor
# ----------------------------------------------------------------------------------------------
# Média dos Valores Observáveis               22.8645     1.2090         26.4052         19.5957
# MQO tradicional                              4.3381     0.4164          5.5275          3.1499
# MQO regularizado (λ=0.25)                    4.3384     0.4164          5.5307          3.1578
# MQO regularizado (λ=0.5)                     4.3391     0.4166          5.5413          3.1662
# MQO regularizado (λ=0.75)                    4.3403     0.4168          5.5549          3.1750
# MQO regularizado (λ=1)                       4.3420     0.4171          5.5690          3.1842
# ---------------------------------------------------------------------------
#
# 1. O modelo de Média dos Valores Observáveis, que simplesmente prevê uma constante igual 
#    à média de y (ignorando as variáveis explicativas), apresentou um RSS médio de aproximadamente 22.86.
#    Esse valor alto indica que este modelo não captura bem  variabilidade dos dados.
#
# 2. Os modelos de regressão linear (tanto o MQO tradicional quanto os regularizados) apresentaram 
#    RSS médios muito mais baixos, em torno de 4.34, demonstrando uma melhora significativa no ajuste aos dados.
#    
# 3. A diferença entre o MQO tradicional (λ = 0) e os modelos regularizados (λ = 0.25, 0.5, 0.75, 1)
#    é mínima. Por exemplo, a média do RSS varia de 4.3381 para 4.3420 conforme λ aumenta de 0 a 1.
#    Essa pequena variação indica que, para este conjunto de dados, a regularização não tem impacto
#    expressivo no desempenho, sugerindo que o problema é bem condicionado e que não há grande risco de overfitting, conforme estudamos em sala de aula. 
#
# 4. O desvio-padrão dos valores de RSS também é baixo (por volta de 0.416), o que demonstra que 
#    os modelos de regressão possuem um desempenho consistente em diferentes particionamentos dos dados.


