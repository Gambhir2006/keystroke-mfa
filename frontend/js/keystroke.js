const typingBox =
    document.getElementById("typingBox");

const startBtn =
    document.getElementById("startBtn");

const statusBox =
    document.getElementById("status");

const progressBar =
    document.getElementById("progressBar");

const progressText =
    document.getElementById("progressText");


const REQUIRED_SAMPLES =
    window.REQUIRED_SAMPLES || 15;


let collecting = false;

let currentEvents = [];

let activeKeys = new Map();

let sampleCount = 0;

let startTime = null;

let finishTimer = null;


// ============================================================
// PROGRESS
// ============================================================

function updateProgress() {

    const percentage =
        Math.min(
            100,
            (sampleCount / REQUIRED_SAMPLES) * 100
        );

    if (progressBar) {

        progressBar.style.width =
            percentage + "%";

    }

    if (progressText) {

        progressText.textContent =
            `${sampleCount} / ${REQUIRED_SAMPLES} samples collected`;

    }
}


// ============================================================
// STATUS
// ============================================================

function setStatus(message) {

    if (statusBox) {

        statusBox.textContent =
            message;

    }
}


// ============================================================
// RESET CURRENT SAMPLE
// ============================================================

function resetSample() {

    currentEvents = [];

    activeKeys.clear();

    startTime = null;

    typingBox.value = "";

    typingBox.focus();

}


// ============================================================
// START ENROLLMENT
// ============================================================

startBtn.addEventListener(
    "click",
    () => {

        collecting = true;

        sampleCount = 0;

        currentEvents = [];

        activeKeys.clear();

        startTime = null;

        typingBox.disabled = false;

        startBtn.disabled = true;

        startBtn.style.display =
            "none";

        updateProgress();

        setStatus(
            "Type your password and press Enter."
        );

        typingBox.focus();

    }
);


// ============================================================
// KEY DOWN
// ============================================================

typingBox.addEventListener(
    "keydown",
    (event) => {

        if (!collecting) {

            return;

        }

        // Ignore repeated keydown events
        if (event.repeat) {

            return;

        }

        // Do not record Enter
        if (event.key === "Enter") {

            event.preventDefault();

            finishSample();

            return;

        }

        if (!startTime) {

            startTime =
                performance.now();

        }

        const identifier =
            event.code +
            ":" +
            event.key;

        activeKeys.set(
            identifier,
            performance.now()
        );

    }
);


// ============================================================
// KEY UP
// ============================================================

typingBox.addEventListener(
    "keyup",
    (event) => {

        if (!collecting) {

            return;

        }

        // Ignore Enter
        if (event.key === "Enter") {

            event.preventDefault();

            return;

        }

        const identifier =
            event.code +
            ":" +
            event.key;

        const downTime =
            activeKeys.get(
                identifier
            );

        if (
            downTime === undefined
        ) {

            return;

        }

        const upTime =
            performance.now();

        currentEvents.push({

            key:
                event.key,

            code:
                event.code,

            down:
                Number(
                    downTime.toFixed(3)
                ),

            up:
                Number(
                    upTime.toFixed(3)
                )

        });

        activeKeys.delete(
            identifier
        );

    }
);


// ============================================================
// FINISH SAMPLE
// ============================================================

async function finishSample() {

    if (!collecting) {

        return;

    }

    // --------------------------------------------------------
    // Basic validation
    // --------------------------------------------------------

    const typedPassword =
        typingBox.value;

    if (
        !typedPassword ||
        typedPassword.length < 2
    ) {

        setStatus(
            "Please type the complete password."
        );

        return;

    }

    // At least two key events
    if (
        currentEvents.length < 2
    ) {

        setStatus(
            "Not enough keystroke data. Please type the password again."
        );

        return;

    }

    // --------------------------------------------------------
    // Make sure no key is currently held
    // --------------------------------------------------------

    if (
        activeKeys.size > 0
    ) {

        setStatus(
            "Please release all keys and press Enter again."
        );

        return;

    }

    // --------------------------------------------------------
    // Stop collecting while sending
    // --------------------------------------------------------

    collecting = false;

    setStatus(
        "Analyzing your typing pattern..."
    );

    // --------------------------------------------------------
    // Send EXACT payload expected by Flask
    // --------------------------------------------------------

    const payload = {

        typed_password:
            typedPassword,

        events:
            currentEvents

    };

    try {

        const response =
            await fetch(
                "/api/enroll/sample",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify(
                            payload
                        )

                }
            );

        const data =
            await response.json();

        // ----------------------------------------------------
        // Backend error
        // ----------------------------------------------------

        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error ||
                "Enrollment failed."
            );

        }

        // ----------------------------------------------------
        // Update sample count
        // ----------------------------------------------------

        sampleCount =
            Number(
                data.count || 0
            );

        updateProgress();

        // ----------------------------------------------------
        // COMPLETE
        // ----------------------------------------------------

        if (
            data.trained ||
            sampleCount >= REQUIRED_SAMPLES
        ) {

            typingBox.disabled =
                true;

            setStatus(
                "Enrollment complete! Your keystroke model has been trained."
            );

            // Login button
            const loginButton =
                document.createElement(
                    "a"
                );

            loginButton.href =
                "/login";

            loginButton.textContent =
                "Continue to Login";

            loginButton.className =
                "button";

            loginButton.style.display =
                "inline-block";

            loginButton.style.marginTop =
                "20px";

            document
                .querySelector(".card")
                .appendChild(
                    loginButton
                );

            return;

        }

        // ----------------------------------------------------
        // NEXT SAMPLE
        // ----------------------------------------------------

        setStatus(
            `Sample ${sampleCount} saved. Type the password again.`
        );

        setTimeout(
            () => {

                resetSample();

                collecting = true;

                setStatus(
                    `Sample ${sampleCount + 1} of ${REQUIRED_SAMPLES}: type your password and press Enter.`
                );

            },
            500
        );

    }

    catch (error) {

        console.error(
            "Enrollment error:",
            error
        );

        setStatus(
            error.message
        );

        collecting = true;

    }

}


// ============================================================
// PREVENT FORM SUBMIT / ENTER DEFAULT
// ============================================================

typingBox.addEventListener(
    "keypress",
    (event) => {

        if (
            event.key === "Enter"
        ) {

            event.preventDefault();

        }

    }
);


// ============================================================
// INITIAL STATE
// ============================================================

updateProgress();