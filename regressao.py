import numpy as np
import matplotlib.pyplot as plt

# 1. Carregando os dados
# [Temperatura, pH, Atividade enzimática]
data_path = r"C:\Users\Bruno Matos\Desktop\IA_modelosderegressaoeclassificacao\dados\atividade_enzimatica.csv"
data = np.genfromtxt(data_path, delimiter=",")

# Extraindo as variáveis:
temperatura = data[:, 0].reshape(-1, 1)
ph = data[:, 1].reshape(-1, 1)
atividade = data[:, 2].reshape(-1, 1)

# 2. Visualização inicial dos dados
# Cria dois gráficos de dispersão: um para Temperatura vs Atividade e outro para pH vs Atividade.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(temperatura, atividade, color='blue', edgecolors='k')
plt.xlabel("Temperatura")
plt.ylabel("Atividade Enzimática")
plt.title("Temperatura vs Atividade Enzimática")

plt.subplot(1, 2, 2)
plt.scatter(ph, atividade, color='red', edgecolors='k')
plt.xlabel("pH da Solução")
plt.ylabel("Atividade Enzimática")
plt.title("pH vs Atividade Enzimática")

plt.tight_layout()
plt.show()

# Discussão inicial:

# A partir dos gráficos, podemos observar a relação entre as variáveis preditoras (temperatura e pH) e a variável resposta (atividade enzimática).
# Um modelo é capaz de entender o padrão entre as variáveis regressoras e as observadas deve capturar, a influência tanto da temperatura
# quanto do pH na atividade enzimática. Em um modelo linear, essa relação é representada pela combinação linear de cada preditor com seu coeficiente mais um intercepto
# (que reflete a atividade básica quando os preditores são nulos).
#
# Entretanto, se a relação verdadeira for não linear ou se houver interação entre as
# variáveis, o modelo pode precisar de transformações ou de técnicas que capturem essa 
# não linearidade. A regularização Tikhonov (Ridge) introduz um parâmetro lambda que corrige garndes 
# coeficientes, ajudando a evitar sobreajuste, etc.

# 3. Organização dos dados para o modelo
# Monta a matriz X (incluindo o intercepto) e o vetor y.
N = data.shape[0]
# X terá 3 colunas: a primeira para o intercepto, a segunda para temperatura e a terceira para o pH.
X = np.concatenate((np.ones((N, 1)), temperatura, ph), axis=1)
y = atividade

# 4. Definição das funções dos modelos de regressão

def ols_regression(X, y):
    """Estimativa MQO tradicional via Equação Normal."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def ridge_regression(X, y, lbda):
    """Estimativa do modelo regularizado (Tikhonov/Ridge)."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lbda * I) @ X.T @ y

def predict(X, beta):
    """Calcula as predições para um vetor de coeficientes beta."""
    return X @ beta

# 5. Estimativa dos modelos

print("---- Estimativas do vetor β ----\n")

# a) MQO tradicional (equivalente a lambda = 0)
beta_ols = ols_regression(X, y)
print("MQO Tradicional (λ = 0):")
print(beta_ols)

# b) MQO Regularizado (Tikhonov/Ridge) para diferentes valores de λ:
lambdas = [0, 0.25, 0.5, 0.75, 1]
beta_ridge = {}  # dicionário para armazenar os coeficientes para cada λ

print("\nMQO Regularizado (Tikhonov):")
for lbda in lambdas:
    beta = ridge_regression(X, y, lbda)
    beta_ridge[lbda] = beta
    print(f"λ = {lbda}:")
    print(beta)
    
# c) Modelo da Média dos Valores Observáveis:
# Aqui, a predição é feita simplesmente utilizando a média de y. Podemos representar essa
# "estimativa" também como um vetor β em que o intercepto é igual à média e os coeficientes dos
# demais preditores são zero.
mean_y = np.mean(y)
beta_mean = np.array([[mean_y], [0], [0]])
print("\nModelo da Média dos Valores Observáveis:")
print(beta_mean)

# =====================================================================
# 5. Validação Monte Carlo 
# =====================================================================

def compute_rss(y_true, y_pred):
    """Calcula a Soma dos Quadrados dos Resíduos (RSS)."""
    return np.sum((y_true - y_pred) ** 2)

R = 500  # 500 rodadas

# Inicialização das listas para armazenar o RSS de cada modelo
rss_mean_list    = []
rss_ols_list     = []
rss_ridge_025_list = []
rss_ridge_05_list  = []
rss_ridge_075_list = []
rss_ridge_1_list   = []

for r in range(R):
    # Embaralha os índices e realiza o particionamento em 80% treino e 20% teste
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx],  y[test_idx]

    # Modelo da Média dos Valores Observáveis
    mean_y_train = np.mean(y_train)
    y_pred_mean = np.full_like(y_test, mean_y_train)
    rss_mean = compute_rss(y_test, y_pred_mean)
    rss_mean_list.append(rss_mean)

    # MQO tradicional (λ = 0)
    beta_ols_sim = ols_regression(X_train, y_train)
    y_pred_ols = predict(X_test, beta_ols_sim)
    rss_ols = compute_rss(y_test, y_pred_ols)
    rss_ols_list.append(rss_ols)

    # MQO regularizado (λ = 0.25)
    beta_ridge_025_sim = ridge_regression(X_train, y_train, lbda=0.25)
    y_pred_ridge_025 = predict(X_test, beta_ridge_025_sim)
    rss_ridge_025 = compute_rss(y_test, y_pred_ridge_025)
    rss_ridge_025_list.append(rss_ridge_025)

    # MQO regularizado (λ = 0.5)
    beta_ridge_05_sim = ridge_regression(X_train, y_train, lbda=0.5)
    y_pred_ridge_05 = predict(X_test, beta_ridge_05_sim)
    rss_ridge_05 = compute_rss(y_test, y_pred_ridge_05)
    rss_ridge_05_list.append(rss_ridge_05)

    # MQO regularizado (λ = 0.75)
    beta_ridge_075_sim = ridge_regression(X_train, y_train, lbda=0.75)
    y_pred_ridge_075 = predict(X_test, beta_ridge_075_sim)
    rss_ridge_075 = compute_rss(y_test, y_pred_ridge_075)
    rss_ridge_075_list.append(rss_ridge_075)

    # MQO regularizado (λ = 1)
    beta_ridge_1_sim = ridge_regression(X_train, y_train, lbda=1)
    y_pred_ridge_1 = predict(X_test, beta_ridge_1_sim)
    rss_ridge_1 = compute_rss(y_test, y_pred_ridge_1)
    rss_ridge_1_list.append(rss_ridge_1)

# =====================================================================
# 6. Cálculo das Estatísticas e Exibição dos Resultados
# =====================================================================

def stats_from_list(rss_list):
    """Retorna (média, desvio padrão, maior valor, menor valor) dos valores da lista."""
    arr = np.array(rss_list)
    return np.mean(arr), np.std(arr), np.max(arr), np.min(arr)

mean_stats     = stats_from_list(rss_mean_list)
ols_stats      = stats_from_list(rss_ols_list)
ridge_025_stats = stats_from_list(rss_ridge_025_list)
ridge_05_stats  = stats_from_list(rss_ridge_05_list)
ridge_075_stats = stats_from_list(rss_ridge_075_list)
ridge_1_stats   = stats_from_list(rss_ridge_1_list)

# Exibe os resultados em uma tabela
print("\nResultados da Validação Monte Carlo (RSS) - 500 Rodadas:\n")
print(f"{'Modelo':<40} {'Média':>10} {'Std':>10} {'Maior Valor':>15} {'Menor Valor':>15}")
print("-" * 90)
print(f"{'Média de valores observáveis':<40} {mean_stats[0]:10.4f} {mean_stats[1]:10.4f} {mean_stats[2]:15.4f} {mean_stats[3]:15.4f}")
print(f"{'MQO tradicional':<40} {ols_stats[0]:10.4f} {ols_stats[1]:10.4f} {ols_stats[2]:15.4f} {ols_stats[3]:15.4f}")
print(f"{'MQO regularizado (λ=0.25)':<40} {ridge_025_stats[0]:10.4f} {ridge_025_stats[1]:10.4f} {ridge_025_stats[2]:15.4f} {ridge_025_stats[3]:15.4f}")
print(f"{'MQO regularizado (λ=0.5)':<40} {ridge_05_stats[0]:10.4f} {ridge_05_stats[1]:10.4f} {ridge_05_stats[2]:15.4f} {ridge_05_stats[3]:15.4f}")
print(f"{'MQO regularizado (λ=0.75)':<40} {ridge_075_stats[0]:10.4f} {ridge_075_stats[1]:10.4f} {ridge_075_stats[2]:15.4f} {ridge_075_stats[3]:15.4f}")
print(f"{'MQO regularizado (λ=1)':<40} {ridge_1_stats[0]:10.4f} {ridge_1_stats[1]:10.4f} {ridge_1_stats[2]:15.4f} {ridge_1_stats[3]:15.4f}")

"""
Resultados e Discussões:
-------------------------------------------------
1. Estimativas do vetor β:
   - MQO Tradicional (λ = 0) apresenta os seguintes coeficientes:
       [[2.50236534]
        [0.07237152]
        [0.08504044]]
     Isso indica que, com a inclusão dos preditores (temperatura e pH), o modelo ajusta um intercepto de cerca de 2.50,
     e coeficientes positivos relativamente pequenos para os preditores, sugerindo uma relação positiva, mas moderada,
     entre as variáveis independentes e a atividade enzimática.

   - Nos modelos regularizados (Tikhonov/Ridge), para λ = 0 o resultado é idêntico ao OLS. Conforme λ aumenta 
     (0.25, 0.5, 0.75, 1), observa-se um pequeno encolhimento dos coeficientes. Por exemplo, o intercepto passa de 
     ~2.50236534 para ~2.49757089 quando λ = 1, enquanto os coeficientes dos preditores também sofrem variações mínimas. 
     Esse comportamento é esperado, pois a regularização penaliza os coeficientes, promovendo maior estabilidade, 
     sem alterar de forma significativa os valores estimados quando o ajuste é robusto.

   - O Modelo da Média dos Valores Observáveis, que simplesmente utiliza a média da atividade enzimática como predição,
     resulta em:
       [[2.58165178]
        [0.        ]
        [0.        ]]
     Isso evidencia que, sem utilizar os preditores, o modelo ignora a variação explicada pelas variáveis 
     temperatura e pH e, portanto, não é ineficiente ao capturar a relação entre elas e a atividade enzimática.

2. Validação Monte Carlo (RSS) - 500 Rodadas:
   - O RSS do Modelo da Média dos Valores Observáveis é substancialmente maior (Média ≈ 22.7952) com uma variabilidade 
     moderada (Std ≈ 1.2612), o que confirma que a simples predição pela média não captura a variabilidade dos dados.
   
   - O MQO Tradicional apresenta um RSS bem menor (Média ≈ 4.2931, Std ≈ 0.4540), indicando um muito bom ajuste 
     ao utilizar os preditores.
   
   - Para valores de λ testados (0.25, 0.5, 0.75 e 1), o RSS médio varia levemente entre ~4.2929 e ~4.2952,
     com um padrão consistente de aumento muito suave à medida que λ cresce.
-------------------------------------------------
"""