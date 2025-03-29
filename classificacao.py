import numpy as np
import matplotlib.pyplot as plt

# 1. Carregando e validando os dados ===========================================
data_path = r"C:\Users\Bruno Matos\Desktop\IA_modelosderegressaoeclassificacao\dados\EMGsDataset.csv"
data = np.genfromtxt(data_path, delimiter=",")

# Verificação e limpeza dos dados
sensor1 = data[0, :]
sensor2 = data[1, :]
labels = data[2, :].astype(int)

# Filtra amostras com rótulos inválidos
valid_mask = np.logical_and(labels >= 1, labels <= 5)
sensor1 = sensor1[valid_mask]
sensor2 = sensor2[valid_mask]
labels = labels[valid_mask]

# Organização dos dados
X_mqo = np.column_stack((sensor1, sensor2))
C = 5
Y_mqo = np.eye(C)[labels - 1]

X_bayes = X_mqo.T
Y_bayes = Y_mqo.T

# 2. Visualização dos dados ====================================================
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['blue', 'green', 'red', 'purple', 'orange']
class_names = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']

for class_label in range(1, C + 1):
    idx = (labels == class_label)
    ax.scatter(sensor1[idx], sensor2[idx], c=colors[class_label - 1],
               label=class_names[class_label - 1], s=10, alpha=0.6)

ax.set_xlabel('Sensor 1 (Corrugador do Supercílio)')
ax.set_ylabel('Sensor 2 (Zigomático Maior)')
ax.set_title('Distribuição dos Dados de EMG por Classe')
ax.legend()
plt.tight_layout()
plt.show()

# Discussão inicial:
# 
# A análise inicial sugere que, embora algumas classes possam ser mais claramente separadas,
# outras apresentam maior dificuldade devido à proximidade dos valores medidos pelos sensores.
# Modelos que conseguem capturar interdependências e características não lineares dos dados 
# provavelmente terão desempenho superior. Assim, os classificadores gaussianos tradicionais 
# ou regularizados podem ser mais eficazes para separar as classes do problema.

# 3. Implementação dos modelos ================================================
class GaussianClassifier:
    @staticmethod
    def traditional(X_train, y_train, X_test):
        classes = np.unique(y_train)
        params = {}
        
        for c in classes:
            X_c = X_train[y_train == c]
            mu = np.mean(X_c, axis=0)
            sigma = np.cov(X_c, rowvar=False) + 1e-6 * np.eye(X_train.shape[1])
            params[c] = (mu, sigma)
        
        return GaussianClassifier._predict(X_test, params, 'max')

    @staticmethod
    def equal_covariance(X_train, y_train, X_test):
        classes = np.unique(y_train)
        means = {}
        pooled_cov = np.zeros((X_train.shape[1], X_train.shape[1]))
        
        total = len(X_train)
        for c in classes:
            X_c = X_train[y_train == c]
            means[c] = np.mean(X_c, axis=0)
            n_c = len(X_c)
            pooled_cov += (n_c - 1) * np.cov(X_c, rowvar=False)
        
        pooled_cov /= (total - len(classes))
        inv_cov = np.linalg.pinv(pooled_cov)
        
        return GaussianClassifier._predict(X_test, (means, inv_cov), 'min')

    @staticmethod
    def aggregated_covariance(X_train, y_train, X_test):
        cov = np.cov(X_train, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        means = {c: np.mean(X_train[y_train == c], axis=0) for c in np.unique(y_train)}
        
        return GaussianClassifier._predict(X_test, (means, inv_cov), 'min')

    @staticmethod
    def friedman(X_train, y_train, X_test, lambda_):
        classes = np.unique(y_train)
        params = {}
        
        for c in classes:
            X_c = X_train[y_train == c]
            mu = np.mean(X_c, axis=0)
            sigma = np.cov(X_c, rowvar=False)
            sigma_reg = (1 - lambda_) * sigma + lambda_ * np.diag(np.diag(sigma)) + 1e-6 * np.eye(sigma.shape[0])
            params[c] = (mu, sigma_reg)
        
        return GaussianClassifier._predict(X_test, params, 'max')

    @staticmethod
    def _predict(X, params, mode):
        if isinstance(params, tuple):  # Para covariância igual/agregada
            means, inv_cov = params
            diffs = X[:, None] - np.array(list(means.values()))
            distances = np.einsum('...i,ij,...j->...', diffs, inv_cov, diffs)
            return np.array(list(means.keys()))[np.argmin(distances, axis=1)]
        else:  # Para covariância específica por classe
            scores = []
            for x in X:
                class_scores = []
                for c, (mu, sigma) in params.items():
                    diff = x - mu
                    inv_sigma = np.linalg.pinv(sigma)
                    score = -0.5 * (np.log(np.linalg.det(sigma)) + diff @ inv_sigma @ diff)
                    class_scores.append(score)
                scores.append(class_scores)
            return np.array(list(params.keys()))[np.argmax(scores, axis=1)]

class NaiveBayes:
    @staticmethod
    def classify(X_train, y_train, X_test):
        classes = np.unique(y_train)
        params = {}
        
        for c in classes:
            X_c = X_train[y_train == c]
            params[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-8
            }
        
        y_pred = []
        for x in X_test:
            scores = []
            for c in classes:
                log_prob = -0.5 * np.sum(np.log(2 * np.pi * params[c]['var']))
                log_prob -= 0.5 * np.sum((x - params[c]['mean'])**2 / params[c]['var'])
                scores.append(log_prob)
            y_pred.append(classes[np.argmax(scores)])
        
        return np.array(y_pred)

class MQO:
    @staticmethod
    def classify(X_train, y_train, X_test):
        C = len(np.unique(y_train))
        Y_train = np.eye(C)[y_train - 1]
        X_bias = np.hstack((np.ones((len(X_train), 1)), X_train))
        beta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ Y_train
        X_test_bias = np.hstack((np.ones((len(X_test), 1)), X_test))
        return np.argmax(X_test_bias @ beta, axis=1) + 1

# 4. Validação Monte Carlo ====================================================
def monte_carlo_simulation(X, y, R=500):
    # Alterando a estrutura para armazenar acurácias
    results = {
        'MQO': [],
        'Gaussiano Tradicional': [],
        'Gaussiano Cov. Iguais': [],
        'Gaussiano Cov. Agregada': [],
        'Friedman': {lam: [] for lam in [0, 0.25, 0.5, 0.75, 1]},
        'Naive Bayes': []
    }
    
    N = len(X)
    
    for _ in range(R):
        idx = np.random.permutation(N)
        split = int(0.8 * N)
        
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]
        
        # Função auxiliar para calcular acurácia
        def calcular_acuracia(y_true, y_pred):
            return np.mean(y_true == y_pred)
        
        # Avaliação dos modelos
        # MQO
        y_pred = MQO.classify(X_train, y_train, X_test)
        results['MQO'].append(calcular_acuracia(y_test, y_pred))
        
        # Gaussianos Tradicional
        y_pred = GaussianClassifier.traditional(X_train, y_train, X_test)
        results['Gaussiano Tradicional'].append(calcular_acuracia(y_test, y_pred))
        
        # Gaussianos Cov. Iguais
        y_pred = GaussianClassifier.equal_covariance(X_train, y_train, X_test)
        results['Gaussiano Cov. Iguais'].append(calcular_acuracia(y_test, y_pred))
        
        # Gaussianos Cov. Agregada
        y_pred = GaussianClassifier.aggregated_covariance(X_train, y_train, X_test)
        results['Gaussiano Cov. Agregada'].append(calcular_acuracia(y_test, y_pred))
        
        # Friedman
        for lam in results['Friedman']:
            y_pred = GaussianClassifier.friedman(X_train, y_train, X_test, lam)
            results['Friedman'][lam].append(calcular_acuracia(y_test, y_pred))
        
        # Naive Bayes
        y_pred = NaiveBayes.classify(X_train, y_train, X_test)
        results['Naive Bayes'].append(calcular_acuracia(y_test, y_pred))
    
    return results

if __name__ == "__main__":
    # Configurações
    R = 500
    
    # Executa simulação
    results = monte_carlo_simulation(X_mqo, labels, R)
    
    # Função para calcular estatísticas
    def get_stats(data):
        return {
            'media': np.mean(data),
            'desvio': np.std(data),
            'max': np.max(data),
            'min': np.min(data)
        }
    
    # Tabela de resultados
    print("\nResultados da Validação Monte Carlo (Acurácia) - 500 Rodadas:\n")
    print("{:<30} {:>10} {:>10} {:>10} {:>10}".format(
        "Modelo", "Média", "Desvio", "Máximo", "Mínimo"))
    print("-" * 70)

    # Modelos principais
    modelos = [
        'MQO', 
        'Gaussiano Tradicional',
        'Gaussiano Cov. Iguais',
        'Gaussiano Cov. Agregada',
        'Naive Bayes'
    ]

    for modelo in modelos:
        stats = get_stats(results[modelo])
        print("{:<30} {:>10.2%} {:>10.2%} {:>10.2%} {:>10.2%}".format(
            modelo,
            stats['media'],
            stats['desvio'],
            stats['max'],
            stats['min']
        ))

    # Friedman com diferentes lambdas
    for lam in [0, 0.25, 0.5, 0.75, 1]:
        stats = get_stats(results['Friedman'][lam])
        print("{:<30} {:>10.2%} {:>10.2%} {:>10.2%} {:>10.2%}".format(
            f"Friedman (λ={lam})",
            stats['media'],
            stats['desvio'],
            stats['max'],
            stats['min']
        ))

"""
Resultados e Discussões:

Resultados da Validação Monte Carlo (Acurácia) - 500 Rodadas:

Modelo                              Média     Desvio     Máximo     Mínimo
----------------------------------------------------------------------
MQO                                72.39%      0.64%     74.10%     70.65%
Gaussiano Tradicional              96.70%      0.17%     97.28%     96.20%
Gaussiano Cov. Iguais              96.26%      0.18%     96.78%     95.72%
Gaussiano Cov. Agregada            94.85%      0.21%     95.52%     94.26%
Naive Bayes                        79.86%      0.37%     80.88%     78.74%
Friedman (λ=0)                     96.70%      0.17%     97.28%     96.20%
Friedman (λ=0.25)                  96.67%      0.17%     97.23%     96.18%
Friedman (λ=0.5)                   96.64%      0.17%     97.18%     96.16%
Friedman (λ=0.75)                  96.60%      0.17%     97.11%     96.12%
Friedman (λ=1)                     96.55%      0.17%     97.08%     96.05%
-------------------------------------------------
Os resultados da Validação Monte Carlo (Acurácia) fornecem uma análise detalhada do desempenho
dos modelos testados ao longo de 500 rodadas. A seguir, as principais observações:

1. MQO (Mínimos Quadrados Ordinários):
   - Média: 72.39%; Desvio Padrão: 0.64%; Máximo: 74.10%; Mínimo: 70.65%
   - O modelo MQO apresentou desempenho significativamente inferior aos classificadores gaussianos.
     Esse resultado é esperado, já que o MQO não é otimizado para problemas de classificação
     e tende a ser menos eficaz em contextos onde as fronteiras entre as classes não são
     linearmente separáveis. A pequena variação no desvio padrão (0.64%) indica estabilidade no desempenho.

2. Classificador Gaussiano Tradicional:
   - Média: 96.70%; Desvio Padrão: 0.17%; Máximo: 97.28%; Mínimo: 96.20%
   - O modelo Gaussiano Tradicional alcançou uma das maiores médias de acurácia. Isso reforça
     a adequação do modelo para capturar as características estatísticas das classes, levando
     a uma separação eficaz. O pequeno desvio padrão demonstra alta consistência no desempenho.

3. Classificador Gaussiano com Covariâncias Iguais:
   - Média: 96.26%; Desvio Padrão: 0.18%; Máximo: 96.78%; Mínimo: 95.72%
   - Embora ligeiramente inferior ao Gaussiano Tradicional, o desempenho foi excelente. Isso
     indica que a hipótese de covariâncias iguais para todas as classes é razoável para este
     conjunto de dados. O desvio padrão permanece baixo, destacando a estabilidade do modelo.

4. Classificador Gaussiano com Covariância Agregada:
   - Média: 94.85%; Desvio Padrão: 0.21%; Máximo: 95.52%; Mínimo: 94.26%
   - Este modelo apresentou um desempenho mais baixo em relação aos outros gaussianos. A
     utilização de uma única matriz de covariância agregada provavelmente limitou a capacidade
     do modelo de capturar diferenças entre as classes. No entanto, a performance ainda é
     sólida, demonstrando que mesmo uma abordagem simplificada pode ser eficaz.

5. Classificador de Bayes Ingênuo:
   - Média: 79.86%; Desvio Padrão: 0.37%; Máximo: 80.88%; Mínimo: 78.74%
   - O Naive Bayes apresentou desempenho moderado, consistente com a suposição de independência
     entre as características. Embora essa abordagem seja simplificada, ela forneceu uma base
     de comparação útil, superando o desempenho do MQO, mas ficando atrás dos modelos gaussianos.

6. Classificador Gaussiano Regularizado (Friedman):
   - λ=0: Média: 96.70%; Desvio Padrão: 0.17%; Máximo: 97.28%; Mínimo: 96.20%
   - λ=0.25: Média: 96.67%; Desvio Padrão: 0.17%; Máximo: 97.23%; Mínimo: 96.18%
   - λ=0.5: Média: 96.64%; Desvio Padrão: 0.17%; Máximo: 97.18%; Mínimo: 96.16%
   - λ=0.75: Média: 96.60%; Desvio Padrão: 0.17%; Máximo: 97.11%; Mínimo: 96.12%
   - λ=1: Média: 96.55%; Desvio Padrão: 0.17%; Máximo: 97.08%; Mínimo: 96.05%
   - A regularização apresentou resultados ligeiramente inferiores ao Gaussiano Tradicional (λ=0),
     indicando que o modelo com matriz de covariância não regularizada já estava bem ajustado.
     Os valores de λ testados demonstraram um leve efeito no desempenho, mas o impacto foi
     pequeno, sugerindo que o conjunto de dados não sofre significativamente de problemas de
     covariância mal-condicionada.


Por fim, os resultados confirmam que os classificadores gaussianos (tradicional, com covariâncias iguais
ou regularizados) são altamente eficazes para este conjunto de dados, alcançando acurácias
superiores a 96%. Os modelos com hipóteses mais simples"""