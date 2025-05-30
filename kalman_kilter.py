import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
dt = 1  # Passo de tempo
num_steps = 50  # Número de passos

# Matrizes do modelo
F = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])  # Matriz de transição de estado

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])  # Matriz de observação

# Covariâncias do ruído (suposições devido à ambiguidade no enunciado)
Q = np.diag([1, 1, 1, 1])  # Ruído do processo
R = np.diag([10.0, 10.0])  # Ruído da medição

# Condições iniciais
x0 = np.array([8.5, 8.5, 8.5/2, 0.0])  # Estado inicial
P0 = np.diag([10.0, 10.0, 10.0, 10.0])  # Covariância inicial alta

# Gerar trajetória real e observações
np.random.seed(40)  # Para reprodutibilidade

true_states = np.zeros((num_steps + 1, 4))
true_states[0] = x0

for k in range(num_steps):
    w = np.random.multivariate_normal(mean=np.zeros(4), cov=Q)
    true_states[k + 1] = F @ true_states[k] + w

measurements = np.zeros((num_steps + 1, 2))
for k in range(num_steps + 1):
    v = np.random.multivariate_normal(mean=np.zeros(2), cov=R)
    measurements[k] = H @ true_states[k] + v

# Função do Filtro de Kalman
def kalman_filter(measurements, F, H, Q, R, x0, P0):
    n_states = F.shape[0]
    num_steps = measurements.shape[0] - 1

    x_est = np.zeros((num_steps + 1, n_states))
    P_est = np.zeros((num_steps + 1, n_states, n_states))

    x_est[0] = x0
    P_est[0] = P0

    for k in range(num_steps):
        # Predição
        x_pred = F @ x_est[k]
        P_pred = F @ P_est[k] @ F.T + Q

        # Atualização
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_est[k + 1] = x_pred + K @ (measurements[k + 1] - H @ x_pred)
        P_est[k + 1] = P_pred - K @ H @ P_pred

    return x_est, P_est

# Executar filtro para diferentes R
R_cases = {
    'R': R,
    '10R': 10 * R,
    '0.1R': 0.1 * R
}

results = {}
for case_name, case_R in R_cases.items():
    x_est, _ = kalman_filter(measurements, F, H, Q, case_R, x0, P0)
    results[case_name] = x_est

# Plotar resultados
plt.figure(figsize=(15, 10))

# Trajetória real
plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label='Trajetória Real')

# Observações
plt.scatter(measurements[:, 0], measurements[:, 1], c='r', s=20, alpha=0.5, label='Observações')

# Estimativas para cada caso de R
colors = {'R': 'blue', '10R': 'orange', '0.1R': 'purple'}
for case_name, color in colors.items():
    x_est = results[case_name]
    plt.plot(x_est[:, 0], x_est[:, 1], linestyle='--', color=color, label=f'Estimativa ({case_name})')

plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.title('Comparação da Trajetória Real, Observações e Estimativas')
plt.legend()
plt.grid(True)
plt.show()

# Discussão:
# - Com R original: O filtro segue as observações, mas com algum atraso.
# - Com R aumentado (10R): O filtro confia menos nas medições, resultando em suavização.
# - Com R reduzido (0.1R): O filtro confia mais nas medições, acompanhando-as de perto.