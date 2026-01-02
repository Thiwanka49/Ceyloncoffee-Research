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

    // Mock response for demo since backend is not running
    const result = {
        pickers: 2,
        harvesters: 2,
        loaders: 1
    };

    // Simulate API delay
    await new Promise(r => setTimeout(r, 500));

    /* 
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
        
        document.getElementById('labor-result').classList.remove('hidden');

    } catch (err) {
        console.error(err);
        alert('Failed to get prediction. Ensure backend is running.');
    }
    */

    document.getElementById('val-pickers').textContent = result.pickers;
    document.getElementById('val-harvesters').textContent = result.harvesters;
    document.getElementById('val-loaders').textContent = result.loaders;
    document.getElementById('labor-result').classList.remove('hidden');
}

async function handleTransportSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    for (const key in data) {
        data[key] = parseFloat(data[key]);
    }
    data.severerisk = 10.0; // Default

    // Mock response for demo
    const result = {
        tractors: 1,
        apes: 0,
        trucks: 1
    };

    // Simulate API delay
    await new Promise(r => setTimeout(r, 500));

    /*
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
        console.error(err);
        alert('Failed to get prediction. Ensure backend is running.');
    }
    */

    document.getElementById('val-tractors').textContent = result.tractors;
    document.getElementById('val-apes').textContent = result.apes;
    document.getElementById('val-trucks').textContent = result.trucks;
    document.getElementById('transport-result').classList.remove('hidden');
}
