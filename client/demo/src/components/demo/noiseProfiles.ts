// One-click starting points for the noise settings; tuned on a real recording with rumble.
export const NOISE_PROFILES = [
    { name: "Casa tranquila", denoise: 1, denoiseStrength: 1, denoiseStrong: 0, vad: 1, vadThreshold: 0.3 },
    { name: "Con ruido", denoise: 1, denoiseStrength: 1, denoiseStrong: 1, vad: 1, vadThreshold: 0.4 },
    { name: "Mucho ruido", denoise: 1, denoiseStrength: 1, denoiseStrong: 1, vad: 1, vadThreshold: 0.5 },
];

type NoiseValues = { denoise: number; denoiseStrength: number; denoiseStrong: number; vad: number; vadThreshold: number };

export const currentNoiseProfile = (s: NoiseValues) =>
    NOISE_PROFILES.findIndex((p) => p.denoise == s.denoise && p.denoiseStrength == s.denoiseStrength && p.denoiseStrong == s.denoiseStrong && p.vad == s.vad && p.vadThreshold == s.vadThreshold);

export const noiseProfileValues = (index: number) => {
    const p = NOISE_PROFILES[index];
    if (!p) return null;
    const { name, ...values } = p;
    return values;
};

// simple (default) or advanced screen; switching reloads the page so audio devices start clean
const UI_MODE_KEY = "vc-ui-mode";
export const getUiMode = (): "simple" | "advanced" => {
    try {
        const stored = localStorage.getItem(UI_MODE_KEY);
        if (stored == "simple" || stored == "advanced") return stored;
        const param = new URLSearchParams(window.location.search).get("mode");
        if (param == "advanced") return "advanced";
    } catch (e) {
        // storage blocked: fall through to the default
    }
    return "simple";
};
export const setUiMode = (mode: "simple" | "advanced") => {
    try {
        localStorage.setItem(UI_MODE_KEY, mode);
    } catch (e) {
        // storage blocked: the reload below still shows the default
    }
    window.location.reload();
};
