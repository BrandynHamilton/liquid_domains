document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('valuationForm');
    const txHashElement = document.getElementById('txHash');
    const responseElement = document.getElementById('response');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const domain = document.getElementById('domain').value;

        // Replace with your actual API endpoint
        const apiUrl = 'https://www.optimizerfinance.com/api/evaluate';

        try {
            // Make the request to the API
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ domain }),
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Request failed');
            }

            // Display the results
            txHashElement.textContent = `Transaction Hash: ${result.txHash}`;
            responseElement.textContent = `Domain: ${result.domain}, Value: ${result.value}`;

        } catch (error) {
            console.error('Error making request:', error);
            responseElement.textContent = `Error: ${error.message}`;
        }
    });
});
