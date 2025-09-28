// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;

    if (path === '/') {
        loadDashboardData();
    } else if (path === '/simulator') {
        setupSimulatorForm();
    } else if (path === '/success-analysis') {
        loadAnalysisData();
    }
});

async function fetchData(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error('A resposta da rede não foi bem-sucedida');
    }
    return response.json();
}

async function loadDashboardData() {
    try {
        const data = await fetchData('/api/dashboard_data');
        document.getElementById('total-convos').textContent = data.total_conversations;
        document.getElementById('success-rate').textContent = `${data.success_rate}%`;
        document.getElementById('var-nome').textContent = `${data.variable_collection.Nome}%`;
        document.getElementById('var-email').textContent = `${data.variable_collection.Email}%`;
        document.getElementById('var-seguro').textContent = `${data.variable_collection.Seguro}%`;
        document.getElementById('var-horario').textContent = `${data.variable_collection.Horario}%`;

        // Podemos adicionar a lógica do gráfico aqui mais tarde
        const ctx = document.getElementById('performance-chart');
        if (ctx) {
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Teste 1', 'Teste 2', 'Teste 3', 'Teste 4'], // Placeholder
                    datasets: [{
                        label: 'Taxa de Sucesso',
                        data: [65, 72, 80, 85], // Dados de exemplo
                        borderColor: '#4f46e5',
                        tension: 0.1
                    }]
                }
            });
        }
    } catch (error) {
        console.error('Falha ao carregar dados do dashboard:', error);
    }
}

function setupSimulatorForm() {
    const form = document.getElementById('simulation-form');
    const log = document.getElementById('conversation-log');
    const spinner = document.getElementById('loading-spinner');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        spinner.style.display = 'block';
        log.innerHTML = '<p class="placeholder">Executando simulação...</p>';

        const personality = document.getElementById('personality').value;

        try {
            const response = await fetch('/api/start_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ personality }),
            });
            const result = await response.json();
            displayTranscript(result.transcript);
        } catch (error) {
            log.innerHTML = `<p class="placeholder">Erro: ${error.message}</p>`;
            console.error('Simulação falhou:', error);
        } finally {
            spinner.style.display = 'none';
        }
    });
}

function displayTranscript(transcript) {
    const log = document.getElementById('conversation-log');
    log.innerHTML = '';
    transcript.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.sender.toLowerCase()}`;
        messageDiv.innerHTML = `<strong>${msg.sender}</strong><p>${msg.text}</p>`;
        log.appendChild(messageDiv);
    });
}

async function loadAnalysisData() {
    try {
        const data = await fetchData('/api/analysis_data');
        
        // Preencher Matriz
        const matrixBody = document.querySelector('#collection-matrix tbody');
        matrixBody.innerHTML = '';
        for (const personality in data.matrix) {
            const rowData = data.matrix[personality];
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${personality.charAt(0).toUpperCase() + personality.slice(1)}</td>
                <td>${rowData.Nome}%</td>
                <td>${rowData.Email}%</td>
                <td>${rowData.Seguro}%</td>
                <td>${rowData.Horario}%</td>
            `;
            matrixBody.appendChild(row);
        }

        // Preencher Motivos de Falha
        const failureList = document.getElementById('failure-reasons-list');
        failureList.innerHTML = '';
        const totalFailures = Object.values(data.failure_reasons).reduce((a, b) => a + b, 0);
        for (const reason in data.failure_reasons) {
            const count = data.failure_reasons[reason];
            const percentage = totalFailures > 0 ? Math.round((count / totalFailures) * 100) : 0;
            const listItem = document.createElement('li');
            listItem.innerHTML = `<span>${reason}</span> <span>${percentage}%</span>`;
            failureList.appendChild(listItem);
        }

    } catch (error) {
        console.error('Falha ao carregar dados de análise:', error);
    }
}