const { Command } = require('commander');
const axios = require('axios'); // Assuming you're using axios for HTTP requests
const program = new Command();

// Define command-line options
program
  .version('1.0.0')
  .requiredOption('-d, --domain <domain>', 'Domain to evaluate')
  .parse(process.argv);

// Get the domain option
const options = program.opts();
const domain = options.domain;

if (!domain) {
  console.error("Please provide a domain as an argument.");
  process.exit(1);
}

async function evaluateDomain() {
  const API_KEY = "superhack2024";
  const url = "https://www.optimizerfinance.com/api/evaluate";

  console.log(`Evaluating domain: ${domain}`);
  console.log(`HTTP POST Request to ${url}`);

  try {
    const domainResponse = await axios.post(url, {
      domain: domain,
      api_key: API_KEY,
    }, {
      headers: {
        "Content-Type": "application/json",
      }
    });

    const responseData = domainResponse.data;

    if (!responseData || responseData.error) {
      throw new Error(`Error in response: ${responseData ? responseData.error : "Unknown error"}`);
    }

    console.log("Domain evaluation response", responseData);

    const result = {
      domain: responseData.domain,
      value: responseData.value,
    };

    // Use JSON.stringify() to convert from JSON object to JSON string
    // Finally, use the helper Functions.encodeString() to encode from string to bytes
    // Assuming Functions.encodeString() is available in your context
    return JSON.stringify(result);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}

// Execute the function
evaluateDomain();
