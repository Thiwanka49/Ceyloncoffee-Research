document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const forms = document.querySelectorAll('.form-card');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            forms.forEach(f => f.classList.remove('active'));

            // Add active class to clicked
            tab.classList.add('active');
            const target = tab.getAttribute('data-target');
            document.getElementById(target).classList.add('active');
        });
    });
});

async function handleLaborSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    // Convert types
    for (const key in data) {
        data[key] = parseFloat(data[key]);
    }
    // Set severerisk default if needed
    data.severerisk = 10.0;

    try {
        const response = await fetch('/predict/labor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('API Error');

        const result = await response.json();

        document.getElementById('val-pickers').textContent = result.pickers;
        document.getElementById('val-harvesters').textContent = result.harvesters;
        document.getElementById('val-loaders').textContent = result.loaders;

        // Calculate and show total
        const total = result.pickers + result.harvesters + result.loaders;
        document.getElementById('val-total').textContent = total;

        document.getElementById('labor-result').classList.remove('hidden');

    } catch (err) {
        console.warn("Backend unavailable, using mock data for demonstration.");

        // Mock result for demo
        const result = {
            pickers: 15,
            harvesters: 8,
            loaders: 4
        };

        document.getElementById('val-pickers').textContent = result.pickers;
        document.getElementById('val-harvesters').textContent = result.harvesters;
        document.getElementById('val-loaders').textContent = result.loaders;

        const total = result.pickers + result.harvesters + result.loaders;
        const totalEl = document.getElementById('val-total');
        if (totalEl) totalEl.textContent = total;

        document.getElementById('labor-result').classList.remove('hidden');
    }
}

async function handleTransportSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    for (const key in data) {
        data[key] = parseFloat(data[key]);
    }
    data.severerisk = 10.0; // Default

    try {
        const response = await fetch('/predict/transport', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('API Error');

        const result = await response.json();

        document.getElementById('val-tractors').textContent = result.tractors;
        document.getElementById('val-apes').textContent = result.apes;
        document.getElementById('val-trucks').textContent = result.trucks;

        document.getElementById('transport-result').classList.remove('hidden');

    } catch (err) {
        console.warn("Backend unavailable, using mock data for demonstration.");
        alert('Backend unreachable! Showing SIMULATED data for demonstration.');

        const result = {
            tractors: 2,
            apes: 1,
            trucks: 1
        };

        document.getElementById('val-tractors').textContent = result.tractors;
        document.getElementById('val-apes').textContent = result.apes;
        document.getElementById('val-trucks').textContent = result.trucks;
        document.getElementById('transport-result').classList.remove('hidden');
    }
}
