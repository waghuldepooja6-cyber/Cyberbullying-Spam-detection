// ================================
// Flask Backend URL
// ================================
const API_URL = "https://cyberbullying-spam-detection-3.onrender.com";

// ================================
// Detect Message
// ================================
async function detectMessage() {
    const messageInput = document.getElementById("message");

    if (!messageInput) {
        console.error("Message input not found");
        return;
    }

    const message = messageInput.value.trim();

    if (!message) {
        alert("Please enter a message.");
        return;
    }

    try {
        const response = await fetch(`${API_URL}/detect`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        console.log("Detection result:", data);

        // Result दाखवण्यासाठी
        const resultElement = document.getElementById("result");

        if (resultElement) {
            resultElement.textContent =
                data.result ||
                data.prediction ||
                data.label ||
                JSON.stringify(data);
        }

    } catch (error) {
        console.error("Detection error:", error);

        alert(
            "Unable to connect to the detection server. " +
            "Please try again later."
        );
    }
}


// ================================
// Button
// ================================
document.addEventListener("DOMContentLoaded", () => {

    const button = document.getElementById("detectBtn");

    if (button) {
        button.addEventListener("click", detectMessage);
    }

});
