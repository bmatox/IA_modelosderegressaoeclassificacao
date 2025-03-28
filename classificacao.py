import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Leitura e Organização dos Dados
# ============================================================================
file_path = r"C:\Users\Bruno Matos\Desktop\IA_modelosderegressaoeclassificacao\dados\EMGsDataset.csv"
dados = np.genfromtxt(file_path, delimiter=',')

# Extraindo os sinais captados dos sensores e os rótulos
sensor1 = dados[0, :]      # Sinais do Sensor 1 (Corrugador do Supercílio)
sensor2 = dados[1, :]      # Sinais do Sensor 2 (Zigomático Maior)
labels   = dados[2, :].astype(int)  # Rótulos: 1 = Neutro, 2 = Sorriso, 3 = Sobrancelhas elevadas, 4 = Surpreso, 5 = Rabugento

N = dados.shape[1]  # Número total de amostras (50000)
p = 2               # Número de características (2 sensores)
C = 5               # Número de classes

# Matrizes de entrada:
# X_raw: cada linha corresponde a uma amostra com os dados dos dois sensores; dimensão: (N, p)
X_raw = np.column_stack((sensor1, sensor2))  

# Para o modelo MQO tradicional, adiciona-se uma coluna de 1's (para o intercepto):
X_MQO = np.concatenate((np.ones((N, 1)), X_raw), axis=1)  # dimensão: (N, p+1)

# Codificação one-hot dos rótulos para o modelo MQO; dimensão: (N, C)
Y_MQO = np.zeros((N, C))
for i in range(N):
    classe = int(labels[i])
    Y_MQO[i, classe - 1] = 1

# Para os modelos gaussianos bayesianos, organiza-se os dados em uma matriz de dimensão (C, N)
# onde cada linha representará informações sobre os exemplos de uma classe.
X_bayes = np.zeros((C, N))
for c in range(1, C+1):
    idx_c = np.where(labels == c)[0]
    # Calcula, para cada exemplo da classe, um valor representativo, por exemplo a média dos sinais:
    media_sinal = np.mean(X_raw[idx_c, :], axis=1)
    X_bayes[c-1, idx_c] = media_sinal

# ============================================================================
# Visualização Inicial dos Dados: Gráfico de Espalhamento com Categorias
# ============================================================================
plt.figure(figsize=(8, 6))
cores = ['red', 'green', 'blue', 'orange', 'purple']
marcadores = ['o', 's', '^', 'd', '*']
for classe in range(1, 6):
    idx = np.where(labels == classe)[0]
    plt.scatter(sensor1[idx], sensor2[idx],
                color=cores[classe-1],
                marker=marcadores[classe-1],
                label=f'Classe {classe}')
plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")
plt.title("Gráfico de Dispersão dos Sinais de EMG por Classe")
plt.legend()
plt.show()

# ============================================================================
# Funções Auxiliares
# ============================================================================
def one_hot_encode(y, num_classes):
    Y = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        Y[i, y[i]-1] = 1
    return Y

def gaussian_pdf(x, mu, cov, epsilon=1e-6):
    p_dim = len(mu)
    diff = x - mu
    # Tenta calcular a inversa; se a matriz for singular, adiciona um pequeno valor à diagonal.
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov + epsilon * np.eye(cov.shape[0]))
    exponent = -0.5 * diff.T @ inv_cov @ diff
    # Da mesma forma, se o determinante for zero (ou muito próximo), use a matriz regularizada:
    det_cov = np.linalg.det(cov)
    if det_cov < epsilon:
        det_cov = np.linalg.det(cov + epsilon * np.eye(cov.shape[0]))
    denom = np.sqrt((2 * np.pi) ** p_dim * det_cov)
    return np.exp(exponent) / denom

def parameters_by_class(X, y):
    """
    Estima o vetor médio (μ) e a matriz de covariância (Σ) para cada classe,
    além de calcular as probabilidades a priori.
    """
    classes = np.unique(y)
    params = {}
    priors = {}
    total = len(y)
    for c in classes:
        idx = np.where(y == c)[0]
        X_c = X[idx, :]
        mu = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False, bias=False)
        params[c] = (mu, cov)
        priors[c] = len(idx) / total
    return params, priors

# ============================================================================
# Implementação dos Modelos de Classificação
# ============================================================================
# Modelo 1: MQO Tradicional (Regressão para Classificação via Pseudoinversa)
def classify_MQO(X_train, Y_train, X_test, num_classes):
    B = np.linalg.pinv(X_train) @ Y_train   # B terá dimensão ((p+1) x C)
    preds = np.argmax(X_test @ B, axis=1) + 1  # +1 para converter de índice zero para classes 1,...,C
    return preds

# Modelo 2: Classificador Gaussiano Tradicional
def classify_gaussian_traditional(X_train, y_train, X_test):
    classes = np.unique(y_train)
    params, priors = parameters_by_class(X_train, y_train)
    preds = []
    for x in X_test:
        posteriors = {}
        for c in classes:
            mu, cov = params[c]
            likelihood = gaussian_pdf(x, mu, cov)
            posteriors[c] = likelihood * priors[c]
        preds.append(max(posteriors, key=posteriors.get))
    return np.array(preds)

# Modelo 3: Classificador Gaussiano com Covariâncias Iguais
def classify_gaussian_equal_cov(X_train, y_train, X_test):
    classes = np.unique(y_train)
    p_dim = X_train.shape[1]
    total = X_train.shape[0]
    mu_dict = {}
    priors = {}
    pooled_cov = np.zeros((p_dim, p_dim))
    for c in classes:
        idx = np.where(y_train == c)[0]
        X_c = X_train[idx, :]
        mu = np.mean(X_c, axis=0)
        mu_dict[c] = mu
        priors[c] = len(idx) / total
        pooled_cov += np.cov(X_c, rowvar=False, bias=False) * (len(idx) - 1)
    pooled_cov /= (total - len(classes))
    inv_pooled = np.linalg.inv(pooled_cov)
    preds = []
    for x in X_test:
        dists = {}
        for c in classes:
            diff = x - mu_dict[c]
            dists[c] = diff.T @ inv_pooled @ diff
        preds.append(min(dists, key=dists.get))
    return np.array(preds)

# Modelo 4: Classificador Gaussiano com Matriz Agregada
def classify_gaussian_aggregated(X_train, y_train, X_test):
    classes = np.unique(y_train)
    mu_dict = {}
    for c in classes:
        idx = np.where(y_train == c)[0]
        X_c = X_train[idx, :]
        mu_dict[c] = np.mean(X_c, axis=0)
    cov_aggregated = np.cov(X_train, rowvar=False, bias=False)
    inv_cov = np.linalg.inv(cov_aggregated)
    preds = []
    for x in X_test:
        dists = {}
        for c in classes:
            diff = x - mu_dict[c]
            dists[c] = diff.T @ inv_cov @ diff
        preds.append(min(dists, key=dists.get))
    return np.array(preds)

# Modelo 5: Classificador Gaussiano Regularizado (Friedman)
def regularize_cov(cov, lambda_val):
    diag_cov = np.diag(np.diag(cov))
    return (1 - lambda_val) * cov + lambda_val * diag_cov

def classify_gaussian_regularized(X_train, y_train, X_test, lambda_val):
    classes = np.unique(y_train)
    p_dim = X_train.shape[1]
    total = X_train.shape[0]
    mu_dict = {}
    pooled_cov = np.zeros((p_dim, p_dim))
    priors = {}
    for c in classes:
        idx = np.where(y_train == c)[0]
        X_c = X_train[idx, :]
        mu = np.mean(X_c, axis=0)
        mu_dict[c] = mu
        priors[c] = len(idx) / total
        pooled_cov += np.cov(X_c, rowvar=False, bias=False) * (len(idx) - 1)
    pooled_cov /= (total - len(classes))
    reg_cov = regularize_cov(pooled_cov, lambda_val)
    inv_reg_cov = np.linalg.inv(reg_cov)
    preds = []
    for x in X_test:
        dists = {}
        for c in classes:
            diff = x - mu_dict[c]
            dists[c] = diff.T @ inv_reg_cov @ diff
        preds.append(min(dists, key=dists.get))
    return np.array(preds)

# Modelo 6: Classificador de Bayes Ingênuo
def classify_naive_bayes(X_train, y_train, X_test):
    classes = np.unique(y_train)
    params = {}
    priors = {}
    total = len(y_train)
    for c in classes:
        idx = np.where(y_train == c)[0]
        X_c = X_train[idx, :]
        mu = np.mean(X_c, axis=0)
        var = np.var(X_c, axis=0)  # Variância de cada característica (assumindo independência)
        params[c] = (mu, var)
        priors[c] = len(idx) / total
    def naive_pdf(x, mu, var):
        prob = 1.0
        for i in range(len(x)):
            prob *= (1 / np.sqrt(2 * np.pi * var[i])) * np.exp(-0.5 * ((x[i] - mu[i])**2) / var[i])
        return prob
    preds = []
    for x in X_test:
        posteriors = {}
        for c in classes:
            mu, var = params[c]
            likelihood = naive_pdf(x, mu, var)
            posteriors[c] = likelihood * priors[c]
        preds.append(max(posteriors, key=posteriors.get))
    return np.array(preds)

# Validação via Monte Carlo
# --------------------------------------------------------------------
R = 500  # Número de iterações
rss_results = {
    "MQO tradicional": [],
    "Gaussiano Tradicional": [],
    "Gaussiano (Cov. de todo cj. treino)": [],
    "Gaussiano (Cov. Agregada)": [],
    "Bayes Ingênuo": [],
    "Gaussiano Reg. (λ=0.25)": [],
    "Gaussiano Reg. (λ=0.5)": [],
    "Gaussiano Reg. (λ=0.75)": []
}

lambdas_reg = [0.25, 0.5, 0.75]

for r in range(R):
    # Embaralhar os índices e particionar os dados
    indices = np.random.permutation(N)
    n_train = int(0.8 * N)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    # Para os classificadores que usam os sinais brutos
    X_train_raw = X_raw[train_idx, :]   # dimensão (n_train, 2)
    X_test_raw  = X_raw[test_idx, :]    # dimensão (n_test, 2)
    y_train = labels[train_idx]         # Vetor de rótulos para treinamento
    y_test  = labels[test_idx]          # Vetor de rótulos para teste
    
    # Para o modelo MQO tradicional, usar versões com intercepto e one-hot encode
    X_train_MQO = X_MQO[train_idx, :]   # dimensão (n_train, 3)
    X_test_MQO  = X_MQO[test_idx, :]    # dimensão (n_test, 3)
    Y_train = np.zeros((len(y_train), C))
    for i in range(len(y_train)):
        Y_train[i, y_train[i]-1] = 1
    
    # Modelo 1: MQO Tradicional
    preds_MQO = classify_MQO(X_train_MQO, Y_train, X_test_MQO, C)
    rss_MQO = np.sum((y_test - preds_MQO)**2)
    rss_results["MQO tradicional"].append(rss_MQO)
    
    # Modelo 2: Classificador Gaussiano Tradicional
    preds_gauss = classify_gaussian_traditional(X_train_raw, y_train, X_test_raw)
    rss_gauss = np.sum((y_test - preds_gauss)**2)
    rss_results["Gaussiano Tradicional"].append(rss_gauss)
    
    # Modelo 3: Gaussiano com Covariâncias Iguais
    preds_equal_cov = classify_gaussian_equal_cov(X_train_raw, y_train, X_test_raw)
    rss_equal_cov = np.sum((y_test - preds_equal_cov)**2)
    rss_results["Gaussiano (Cov. de todo cj. treino)"].append(rss_equal_cov)
    
    # Modelo 4: Gaussiano com Matriz Agregada
    preds_agg = classify_gaussian_aggregated(X_train_raw, y_train, X_test_raw)
    rss_agg = np.sum((y_test - preds_agg)**2)
    rss_results["Gaussiano (Cov. Agregada)"].append(rss_agg)
    
    # Modelo 5: Bayes Ingênuo
    preds_naive = classify_naive_bayes(X_train_raw, y_train, X_test_raw)
    rss_naive = np.sum((y_test - preds_naive)**2)
    rss_results["Bayes Ingênuo"].append(rss_naive)
    
    # Modelos 6, 7 e 8: Gaussiano Regularizado para os valores de lambda (λ=0.25, 0.5, 0.75)
    for lam in lambdas_reg:
        preds_reg = classify_gaussian_regularized(X_train_raw, y_train, X_test_raw, lam)
        rss_reg = np.sum((y_test - preds_reg)**2)
        if lam == 0.25:
            rss_results["Gaussiano Reg. (λ=0.25)"].append(rss_reg)
        elif lam == 0.5:
            rss_results["Gaussiano Reg. (λ=0.5)"].append(rss_reg)
        elif lam == 0.75:
            rss_results["Gaussiano Reg. (λ=0.75)"].append(rss_reg)

# --------------------------------------------------------------------
# Resumo dos Resultados: Impressão da Tabela
# --------------------------------------------------------------------
print("\nResumo dos Resultados (RSS - Soma dos Desvios Quadráticos) após 500 simulações:")
header = "{:<50} {:>10} {:>15} {:>15} {:>15}".format("Modelo", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor")
print(header)
print("-" * len(header))
for model in ["MQO tradicional",
              "Gaussiano Tradicional",
              "Gaussiano (Cov. de todo cj. treino)",
              "Gaussiano (Cov. Agregada)",
              "Bayes Ingênuo",
              "Gaussiano Reg. (λ=0.25)",
              "Gaussiano Reg. (λ=0.5)",
              "Gaussiano Reg. (λ=0.75)"]:
    rss_arr = np.array(rss_results[model])
    media_val = np.mean(rss_arr)
    std_val = np.std(rss_arr)
    max_val = np.max(rss_arr)
    min_val = np.min(rss_arr)
    linha = "{:<50} {:10.4f} {:15.4f} {:15.4f} {:15.4f}".format(model, media_val, std_val, max_val, min_val)
    print(linha)
