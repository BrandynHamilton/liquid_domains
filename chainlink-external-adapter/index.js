const express = require('express');
const bodyParser = require('body-parser');
require('dotenv').config();

const app = express();
app.use(bodyParser.json());

const PORT = process.env.PORT || 8080;
const API_ENDPOINT = 'http://localhost:5555/api/evaluate';

app.post('/api/evaluate', async (req, res) => {
    const data = req.body;
    const domain = data.domain;
    const apiKey = data.api_key;

    console.log('Received request body:', data);

    if (!domain || !apiKey) {
        console.log('Missing domain or API key');
        return res.status(400).json({ error: 'Missing domain or API key' });
    }

    try {
        const { default: fetch } = await import('node-fetch');

        console.log('Sending request to external API with domain and API key:', { domain, api_key: apiKey });

        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ domain, api_key: apiKey })
        });

        const responseData = await response.json();

        console.log('Received response from external API:', responseData);

        res.status(200).json({
            data: responseData,
            result: responseData.value,
            statusCode: 200
        });
    } catch (error) {
        console.error('Error occurred:', error.message);
        res.status(500).json({
            status: 'errored',
            error: error.message
        });
    }
});

app.listen(PORT, () => console.log(`External adapter listening on port ${PORT}`));
