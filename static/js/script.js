
// ==================================================
// 1. MATCH PREDICTION ELEMENTS
// ==================================================

const homeTeam = document.getElementById("home-team");
const awayTeam = document.getElementById("away-team");
const matchDate = document.getElementById("match-date");

const predictButton = document.getElementById("predict-button");
const errorMessage = document.getElementById("error-message");

const resultSection = document.getElementById("result-section");
const matchTitle = document.getElementById("match-title");
const predictedOutcome = document.getElementById("predicted-outcome");
const probabilityResults = document.getElementById("probability-results");
const eloResult = document.getElementById("elo-result");


// ==================================================
// 2. MATCH PREDICTION FUNCTIONALITY
// ==================================================

predictButton.addEventListener("click", async function () {

    const home = homeTeam.value;
    const away = awayTeam.value;
    const date = matchDate.value;

    errorMessage.textContent = "";
    resultSection.classList.add("hidden");

    if (!home || !away || !date) {

        errorMessage.textContent =
            "Please select both teams and a match date.";

        return;
    }

    if (home === away) {

        errorMessage.textContent =
            "Please select two different teams.";

        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Analyzing Match...";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                home_team: home,
                away_team: away,
                match_date: date
            })

        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error || "Prediction failed."
            );

        }

        // Display match details

        matchTitle.textContent =
            `${data.home_team} vs ${data.away_team} | ${data.match_date}`;

        predictedOutcome.textContent =
            `Predicted Outcome: ${data.predicted_outcome}`;


        // Display probabilities

        probabilityResults.innerHTML = "";

        const outcomeOrder = [
            "Home Win",
            "Draw",
            "Away Win"
        ];

        outcomeOrder.forEach(outcome => {

            const probability = data.probabilities[outcome];

            const row = document.createElement("div");

            row.className = "probability-row";

            const label = document.createElement("div");
            label.className = "probability-label";

            const name = document.createElement("span");
            name.textContent = outcome;

            const percentage = document.createElement("strong");
            percentage.textContent = `${probability.toFixed(2)}%`;

            label.append(name, percentage);

            const track = document.createElement("div");
            track.className = "progress-track";

            const fill = document.createElement("div");
            fill.className = "progress-fill";
            fill.style.width = `${probability}%`;

            track.appendChild(fill);

            row.append(label, track);

            probabilityResults.appendChild(row);

        });


        // Display Elo difference

        eloResult.textContent =
            `Elo Difference (Home - Away): ${data.elo_difference}`;


        resultSection.classList.remove("hidden");

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {

        errorMessage.textContent = error.message;

    } finally {

        predictButton.disabled = false;
        predictButton.textContent = "Generate Prediction →";

    }

});


// ==================================================
// 3. FOOTIQ AI CHAT ELEMENTS
// ==================================================

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");

const sendChatButton =
    document.getElementById("send-chat-button");

const clearChatButton =
    document.getElementById("clear-chat-button");


// ==================================================
// 4. SCROLL CHAT TO LATEST MESSAGE
// ==================================================

function scrollChatToBottom() {

    chatMessages.scrollTop = chatMessages.scrollHeight;

}


// ==================================================
// 5. CREATE CHAT MESSAGE
// ==================================================



function addMessage(role, message) {

    const messageContainer = document.createElement("div");

    messageContainer.className =
        role === "user"
            ? "message user-message"
            : "message assistant-message";

    const avatar = document.createElement("div");

    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "⚽";

    const content = document.createElement("div");

    content.className = "message-content";

    if (role === "user") {

        const paragraph = document.createElement("p");
        paragraph.textContent = message;
        content.appendChild(paragraph);

    } else {

        const lines = String(message).split("\n");

        let currentList = null;
        let skippingSources = false;

        lines.forEach(line => {

            const trimmed = line.trim();
            // Remove Knowledge Sources from Gemini's answer
            if (/^#{1,4}\s*knowledge sources/i.test(trimmed)) {
                skippingSources = true;
                currentList = null;
                return;
            }
            if (skippingSources) {
                return;
            }

            // Ignore empty lines
            if (!trimmed) {
                currentList = null;
                return;
            }

            // Horizontal divider (--- or \---)
            if (/^\\?---+$/.test(trimmed)) {

                currentList = null;

                const divider = document.createElement("hr");
                content.appendChild(divider);

                return;
            }

            // Remove unnecessary leading escape character
            const cleanLine = trimmed.replace(/^\\(?=---)/, "");

            // Headings (check longer headings first)
            if (cleanLine.startsWith("#### ")) {

                currentList = null;

                const heading = document.createElement("h4");
                formatInlineMarkdown(
                    heading,
                    cleanLine.substring(5)
                );

                content.appendChild(heading);
                return;
            }

            if (cleanLine.startsWith("### ")) {

                currentList = null;

                const heading = document.createElement("h4");
                formatInlineMarkdown(
                    heading,
                    cleanLine.substring(4)
                );

                content.appendChild(heading);
                return;
            }

            if (cleanLine.startsWith("## ")) {

                currentList = null;

                const heading = document.createElement("h3");
                formatInlineMarkdown(
                    heading,
                    cleanLine.substring(3)
                );

                content.appendChild(heading);
                return;
            }

            // Bullet points
            if (/^[-*]\s+/.test(cleanLine)) {

                if (!currentList || currentList.tagName !== "UL") {

                    currentList = document.createElement("ul");
                    content.appendChild(currentList);

                }

                const item = document.createElement("li");

                formatInlineMarkdown(
                    item,
                    cleanLine.replace(/^[-*]\s+/, "")
                );

                currentList.appendChild(item);
                return;
            }

            // Numbered lists
            if (/^\d+\.\s+/.test(cleanLine)) {

                if (!currentList || currentList.tagName !== "OL") {

                    currentList = document.createElement("ol");
                    content.appendChild(currentList);

                }

                const item = document.createElement("li");

                formatInlineMarkdown(
                    item,
                    cleanLine.replace(/^\d+\.\s+/, "")
                );

                currentList.appendChild(item);
                return;
            }

            // Normal paragraph
            currentList = null;

            const paragraph = document.createElement("p");

            formatInlineMarkdown(paragraph, cleanLine);

            content.appendChild(paragraph);

        });

    }

    messageContainer.append(avatar, content);

    chatMessages.appendChild(messageContainer);

    scrollChatToBottom();

    return messageContainer;
}


// Convert basic inline Markdown into safe HTML elements
function formatInlineMarkdown(element, text) {

    const regex = /\*\*(.*?)\*\*|\*(.*?)\*|`(.*?)`/g;

    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {

        // Add text before the formatting
        element.appendChild(
            document.createTextNode(
                text.substring(lastIndex, match.index)
            )
        );

        let formattedElement;

        if (match[1] !== undefined) {

            formattedElement = document.createElement("strong");
            formattedElement.textContent = match[1];

        } else if (match[2] !== undefined) {

            formattedElement = document.createElement("em");
            formattedElement.textContent = match[2];

        } else {

            formattedElement = document.createElement("code");
            formattedElement.textContent = match[3];

        }

        element.appendChild(formattedElement);

        lastIndex = regex.lastIndex;

    }

    // Add remaining text
    element.appendChild(
        document.createTextNode(text.substring(lastIndex))
    );

}


// ==================================================
// 6. DISPLAY RETRIEVED KNOWLEDGE SOURCES
// ==================================================

function displaySources(messageContainer, sources) {

    if (!sources || sources.length === 0) {
        return;
    }

    const content =
        messageContainer.querySelector(".message-content");

    const sourcesContainer =
        document.createElement("div");

    sourcesContainer.className = "chat-sources";


    const heading = document.createElement("h4");

    heading.textContent = "Knowledge Sources";

    sourcesContainer.appendChild(heading);


    sources.forEach(source => {

        const sourceText = document.createElement("p");

        sourceText.textContent =
            `${source.topic} (Distance: ${source.distance})`;

        sourcesContainer.appendChild(sourceText);

    });


    content.appendChild(sourcesContainer);

    scrollChatToBottom();

}


// ==================================================
// 7. DISPLAY ML PREDICTION INSIDE CHAT
// ==================================================

function displayChatPrediction(messageContainer, prediction) {

    if (!prediction) {
        return;
    }

    const content =
        messageContainer.querySelector(".message-content");

    const predictionCard =
        document.createElement("div");

    predictionCard.className = "chat-prediction";


    const heading = document.createElement("h4");

    heading.textContent = "⚽ ML Match Prediction";

    predictionCard.appendChild(heading);


    const match = document.createElement("p");

    match.textContent =
        `${prediction.home_team} vs ${prediction.away_team}`;

    predictionCard.appendChild(match);


    const date = document.createElement("p");

    date.textContent =
        `Match Date: ${prediction.match_date}`;

    predictionCard.appendChild(date);


    const outcome = document.createElement("p");

    outcome.textContent =
        `Predicted Outcome: ${prediction.predicted_outcome}`;

    predictionCard.appendChild(outcome);


    const probabilities = prediction.probabilities;

    const outcomeOrder = [
        "Home Win",
        "Draw",
        "Away Win"
    ];


    outcomeOrder.forEach(name => {

        const probability = Number(probabilities[name]);

        if (!Number.isFinite(probability)) {
            return;
        }

        const row = document.createElement("div");

        row.className = "chat-probability";


        const label =
            document.createElement("div");

        label.className = "chat-probability-label";


        const outcomeName = document.createElement("span");

        outcomeName.textContent = name;


        const percentage = document.createElement("strong");

        percentage.textContent =
            `${probability.toFixed(2)}%`;


        label.append(outcomeName, percentage);


        const track =
            document.createElement("div");

        track.className = "chat-probability-track";


        const fill =
            document.createElement("div");

        fill.className = "chat-probability-fill";

        fill.style.width =
            `${Math.max(0, Math.min(100, probability))}%`;


        track.appendChild(fill);

        row.append(label, track);

        predictionCard.appendChild(row);

    });


    content.appendChild(predictionCard);

    scrollChatToBottom();

}


// ==================================================
// DISPLAY OVERALL TEAM COMPARISON TABLE
// ==================================================

function displayTeamComparison(messageContainer, sqlAnalytics) {

    // Only handle overall team comparisons
    if (!sqlAnalytics || sqlAnalytics.type !== "team_comparison") {
        return;
    }

    const teams = sqlAnalytics.results;

    if (!teams) {
        return;
    }

    const teamNames = Object.keys(teams);

    if (teamNames.length < 2) {
        return;
    }

    const content = messageContainer.querySelector(".message-content");

    const comparisonCard = document.createElement("div");
    comparisonCard.className = "team-comparison-card";

    const heading = document.createElement("h4");
    heading.textContent = "⚽ Overall Team Statistics Comparison";

    comparisonCard.appendChild(heading);

    const tableWrapper = document.createElement("div");
    tableWrapper.className = "team-comparison-table-wrapper";

    const table = document.createElement("table");
    table.className = "team-comparison-table";

    // Table header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    ["Statistic", teamNames[0], teamNames[1]].forEach(name => {

        const th = document.createElement("th");
        th.textContent = name;
        headerRow.appendChild(th);

    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Statistics to display
    const statistics = [
        {
            label: "Matches Played",
            key: "matches_played",
            decimals: 0
        },
        {
            label: "Wins",
            key: "wins",
            decimals: 0
        },
        {
            label: "Draws",
            key: "draws",
            decimals: 0
        },
        {
            label: "Losses",
            key: "losses",
            decimals: 0
        },
        {
            label: "Average Goals Scored",
            key: "avg_goals_scored",
            decimals: 2
        },
        {
            label: "Average Goals Conceded",
            key: "avg_goals_conceded",
            decimals: 2
        }
    ];

    const tbody = document.createElement("tbody");

    statistics.forEach(stat => {

        const row = document.createElement("tr");

        const labelCell = document.createElement("td");
        labelCell.textContent = stat.label;

        row.appendChild(labelCell);

        teamNames.slice(0, 2).forEach(teamName => {

            const cell = document.createElement("td");

            const value = teams[teamName][stat.key];

            cell.textContent =
                value !== undefined && value !== null
                    ? Number(value).toFixed(stat.decimals)
                    : "N/A";

            row.appendChild(cell);

        });

        tbody.appendChild(row);

    });

    table.appendChild(tbody);

    tableWrapper.appendChild(table);
    comparisonCard.appendChild(tableWrapper);

    const note = document.createElement("p");
    note.className = "comparison-note";
    note.textContent =
        "Historical English Premier League data ending 10 April 2022.";

    comparisonCard.appendChild(note);

    content.appendChild(comparisonCard);

    scrollChatToBottom();

}


// ==================================================
// 8. LOADING INDICATOR
// ==================================================

function showTypingIndicator() {

    const messageContainer =
        document.createElement("div");

    messageContainer.className =
        "message assistant-message";

    messageContainer.id = "typing-indicator";


    const avatar =
        document.createElement("div");

    avatar.className = "message-avatar";

    avatar.textContent = "⚽";


    const content =
        document.createElement("div");

    content.className = "message-content";


    const typing =
        document.createElement("div");

    typing.className = "typing-indicator";


    for (let i = 0; i < 3; i++) {

        const dot = document.createElement("span");

        typing.appendChild(dot);

    }


    content.appendChild(typing);

    messageContainer.append(avatar, content);

    chatMessages.appendChild(messageContainer);

    scrollChatToBottom();

}


function removeTypingIndicator() {

    const indicator =
        document.getElementById("typing-indicator");

    if (indicator) {
        indicator.remove();
    }

}


// ==================================================
// 9. SEND QUESTION TO FLASK /chat API
// ==================================================

async function sendChatMessage() {

    const question = chatInput.value.trim();

    if (!question) {
        return;
    }


    // Display user's question

    addMessage("user", question);

    chatInput.value = "";


    // Disable input while waiting for Gemini

    sendChatButton.disabled = true;

    sendChatButton.textContent = "Thinking...";

    chatInput.disabled = true;


    showTypingIndicator();


    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Unable to process your question."
            );

        }


        removeTypingIndicator();


        // Display Gemini's answer

        const assistantMessage =
            addMessage("assistant", data.answer);


        // Display ChromaDB sources

        displaySources(
            assistantMessage,
            data.sources
        );

        // Display SQL team comparison table

        displayTeamComparison(
            assistantMessage,
            data.sql_analytics
        );


        // Display ML prediction if available

        displayChatPrediction(
            assistantMessage,
            data.prediction
        );


    } catch (error) {

        removeTypingIndicator();

        addMessage(
            "assistant",
            `Sorry, something went wrong: ${error.message}`
        );

    } finally {

        sendChatButton.disabled = false;

        sendChatButton.textContent = "Send ↑";

        chatInput.disabled = false;

        chatInput.focus();

    }

}


// ==================================================
// 10. SEND BUTTON EVENT
// ==================================================

sendChatButton.addEventListener(
    "click",
    sendChatMessage
);


// ==================================================
// 11. ENTER KEY TO SEND MESSAGE
// ==================================================

chatInput.addEventListener("keydown", function (event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        if (!sendChatButton.disabled) {
            sendChatMessage();
        }

    }

});


// ==================================================
// 12. CLEAR CHAT
// ==================================================

clearChatButton.addEventListener("click", function () {

    chatMessages.innerHTML = "";

    addMessage(
        "assistant",
        "Hello! I'm FootIQ, your intelligent football analytics assistant.\n\nAsk me about football tactics, team statistics, historical matches or machine learning predictions."
    );

});
