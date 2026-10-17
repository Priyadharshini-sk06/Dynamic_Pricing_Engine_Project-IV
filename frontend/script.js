
"use strict";

// FastAPI serves the frontend and API on port 8000.
// If you use a different backend port, update API_BASE_URL.
const API_BASE_URL =
  window.location.port === "8000" || window.location.port === ""
    ? ""
    : "http://127.0.0.1:8000";

const form = document.getElementById("pricing-form");
const submitBtn = document.getElementById("submit-btn");
const errorBox = document.getElementById("error-box");
const resultBox = document.getElementById("result");

const defaultButtonHTML =
  '<span>Calculate recommended price</span><span class="button-arrow">→</span>';

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  hideError();
  resultBox.hidden = true;

  const payload = {
    product_category: document.getElementById("product_category").value,
    base_price: parseFloat(document.getElementById("base_price").value),
    competitor_price: parseFloat(
      document.getElementById("competitor_price").value
    ),
    inventory_level: parseInt(
      document.getElementById("inventory_level").value,
      10
    ),
    demand_signal: parseFloat(
      document.getElementById("demand_signal").value
    ),
    day_of_week: document.getElementById("day_of_week").value,
    is_holiday: document.getElementById("is_holiday").checked,
    customer_segment: document.getElementById("customer_segment").value,
  };

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/predict-price`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      const detail = errorBody.detail
        ? JSON.stringify(errorBody.detail)
        : `Request failed (${response.status})`;

      throw new Error(detail);
    }

    const data = await response.json();
    renderResult(data);
  } catch (error) {
    showError(
      `Could not reach the pricing API at ${
        API_BASE_URL || window.location.origin
      }. Make sure the FastAPI backend is running. (${error.message})`
    );
  } finally {
    setLoading(false);
  }
});

function renderResult(data) {
  document.getElementById("recommended-price").textContent =
    formatCurrency(data.recommended_price);

  document.getElementById("raw-prediction").textContent =
    formatCurrency(data.adjustments.raw_model_prediction);

  document.getElementById("blended-price").textContent =
    formatCurrency(data.adjustments.competitor_blended_price);

  document.getElementById("inventory-adj").textContent =
    formatPct(data.adjustments.inventory_adjustment_pct);

  document.getElementById("demand-adj").textContent =
    formatPct(data.adjustments.demand_adjustment_pct);

  resultBox.hidden = false;
}

function formatCurrency(value) {
  return `$${Number(value).toFixed(2)}`;
}

function formatPct(value) {
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(2)}%`;
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.innerHTML = isLoading
    ? "<span>Calculating recommendation...</span><span>…</span>"
    : defaultButtonHTML;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

