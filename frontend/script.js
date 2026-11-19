
const predictionForm = document.getElementById("prediction-form");
const predictionResult = document.getElementById("prediction-result");
const formMessage = document.getElementById("form-message");

// Connect to the deployed FastAPI backend
const API_URL = "/api/predict";

// Handle prediction form submission
predictionForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const loanAmount = Number(
        document.getElementById("loan_amount").value
    );

    const term = Number(
        document.getElementById("term_in_months").value
    );

    const sector = document.getElementById("sector").value;

    const repaymentInterval = document.getElementById(
        "repayment_interval"
    ).value;

    // Validate input
    if (
        !Number.isFinite(loanAmount) ||
        !Number.isFinite(term) ||
        loanAmount <= 0 ||
        term <= 0
    ) {
        formMessage.textContent =
            "Enter valid positive loan details.";
        return;
    }

    const requestData = {
        loan_amount: loanAmount,
        term_in_months: term,
        sector: sector,
        repayment_interval: repaymentInterval
    };

    formMessage.textContent =
        "Analyzing your loan profile...";

    predictionResult.innerHTML = `
        <div class="result-icon">⏳</div>
        <span class="eyebrow">PROCESSING</span>
        <h3>Analyzing loan profile</h3>
        <p>Please wait while the K-Means model processes your details.</p>
    `;

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.detail || "Unable to process prediction."
            );
        }

        predictionResult.innerHTML = `
            <div class="result-icon">✦</div>
            <span class="eyebrow">PREDICTION SUCCESSFUL</span>

            <div class="result-cluster">
                Cluster ${result.cluster}
            </div>

            <h3>${result.cluster_profile}</h3>

            <div class="result-details">
                <p>
                    <strong>Average cluster loan:</strong>
                    ₹${result.training_cluster_profile.average_loan_inr.toLocaleString("en-IN")}
                </p>

                <p>
                    <strong>Average loan term:</strong>
                    ${result.training_cluster_profile.average_term_months} months
                </p>

                <p>
                    <strong>Dominant sectors:</strong>
                    ${result.training_cluster_profile.dominant_sectors.join(", ")}
                </p>

                <p>
                    <strong>Repayment pattern:</strong>
                    ${result.training_cluster_profile.repayment_pattern}
                </p>
            </div>

            <p class="disclaimer">${result.notice}</p>
        `;

        formMessage.textContent =
            "Prediction completed successfully!";

    } catch (error) {
        console.error("Prediction error:", error);

        predictionResult.innerHTML = `
            <div class="result-icon">!</div>
            <span class="eyebrow">PREDICTION FAILED</span>
            <h3>Unable to connect to the API</h3>
            <p>${error.message}</p>
            <p>Please check the backend connection.</p>
        `;

        formMessage.textContent =
            "Please check your backend connection.";
    }
});
