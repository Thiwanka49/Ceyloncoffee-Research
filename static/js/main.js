document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const globalPriceRange = document.getElementById('global_price');
    const globalPriceVal = document.getElementById('global_price_val');
    const usdLkrRange = document.getElementById('usd_lkr');
    const usdLkrVal = document.getElementById('usd_lkr_val');

    const priceOutput = document.getElementById('price-output');
    const demandOutput = document.getElementById('demand-output');
    const demandStatus = document.getElementById('demand-status');
    const predictBtn = document.getElementById('predict-btn');

    // Update range values
    globalPriceRange.addEventListener('input', (e) => {
        globalPriceVal.textContent = e.target.value;
    });

    usdLkrRange.addEventListener('input', (e) => {
        usdLkrVal.textContent = e.target.value;
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        predictBtn.disabled = true;
        predictBtn.textContent = 'Calculating...';

        const payload = {
            year: document.getElementById('year').value,
            month: document.getElementById('month').value,
            global_price: globalPriceRange.value,
            usd_lkr: usdLkrRange.value,
            yield: document.getElementById('yield').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.status === 'success') {
                animateValue(priceOutput, data.predictions.local_price_lkr, 1000);
                demandOutput.textContent = data.predictions.demand_index;

                // Update Status
                if (data.predictions.demand_index > 1.05) {
                    demandStatus.textContent = 'High Demand';
                    demandStatus.style.background = 'rgba(34, 197, 94, 0.1)';
                    demandStatus.style.color = '#4ade80';
                } else if (data.predictions.demand_index < 0.95) {
                    demandStatus.textContent = 'Low Demand';
                    demandStatus.style.background = 'rgba(239, 68, 68, 0.1)';
                    demandStatus.style.color = '#f87171';
                } else {
                    demandStatus.textContent = 'Stable';
                    demandStatus.style.background = 'rgba(248, 250, 252, 0.1)';
                    demandStatus.style.color = '#94a3b8';
                }

                // AI Advisory
                const advisoryCard = document.getElementById('advisory-card');
                const advisoryTitle = document.getElementById('advisory-title');
                const advisoryMessage = document.getElementById('advisory-message');
                const advisoryAction = document.getElementById('advisory-action');

                advisoryCard.style.display = 'block';
                advisoryCard.style.opacity = '0';
                advisoryTitle.textContent = data.advisory.title;
                advisoryMessage.textContent = data.advisory.message;
                advisoryAction.textContent = data.advisory.action;
                advisoryAction.style.borderColor = data.advisory.color;
                advisoryAction.style.color = data.advisory.color;
                advisoryAction.style.background = `${data.advisory.color}15`;

                // Fade in effect
                setTimeout(() => {
                    advisoryCard.style.transition = 'opacity 0.5s ease';
                    advisoryCard.style.opacity = '1';
                }, 100);

            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Could not connect to the backend server.');
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Generate Prediction';
        }
    });

    function animateValue(obj, end, duration) {
        let startTimestamp = null;
        const currentText = obj.textContent.replace(/,/g, '');
        const start = parseFloat(currentText) || 0;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = Math.floor(progress * (end - start) + start);
            obj.textContent = current.toLocaleString();
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
