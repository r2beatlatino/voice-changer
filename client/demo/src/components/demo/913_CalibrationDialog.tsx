import React, { useMemo, useRef, useState } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { useAppState } from "../../001_provider/001_AppStateProvider";

// Setup assistant: records the room and the user reading, and the server picks the noise
// settings and the pitch shift (server/voice_changer/common/robust/Calibration.py).

const NOISE_SECONDS = 10;
const VOICE_SECONDS = 15;
const READING =
    "Hola, estoy probando mi nueva voz. Uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve, diez, once, doce. " +
    "El perro de San Roque no tiene rabo, porque Ramón Ramírez se lo ha cortado. ¿Qué tal suena? Creo que bastante bien.";

const TIPS: { [key: string]: string } = {
    no_voice: "Casi no se oyó tu voz en la lectura. Repite el paso 2 hablando un poco más cerca del micro.",
    mic_clipping: "Tu micro satura (se oye «roto» en los picos). Baja su volumen en Windows o aléjate un poco.",
    mic_low: "Tu micro suena muy bajo. Súbelo en Windows (Sonido > Grabación > Propiedades > Niveles).",
    low_rumble: "Hay un ruido grave de fondo (ventilador, aire acondicionado o vibración del escritorio). Aleja el micro del PC o ponlo sobre algo blando.",
    noisy_room: "Hay bastante ruido comparado con tu voz. La limpieza fuerte ayuda, pero hablar más cerca del micro ayuda todavía más.",
    vad_cuts: "Incluso con el VAD poco estricto se cortaría parte de tu voz. Si notas palabras cortadas, apaga el VAD.",
    speech_like_noise: "El ruido de fondo se parece a una voz (tele, música u otras personas). El VAD lo dejará pasar a ratos.",
    no_pitch: "No pude medir tu tono. Repite el paso 2 leyendo en voz alta.",
};

type Recording = { sampleRate: number; data: Float32Array };
type CalibrationResult = {
    settings: { [key: string]: number };
    measure: { [key: string]: number };
    tips: string[];
};

const recordMic = async (deviceId: string | null, seconds: number, onTick: (left: number) => void): Promise<Recording> => {
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            // the raw microphone, as the server will hear it
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: 1,
        },
    });
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks: Float32Array[] = [];
    proc.onaudioprocess = (e) => {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    src.connect(proc);
    proc.connect(ctx.destination); // the output buffer stays silent
    for (let left = seconds; left > 0; left--) {
        onTick(left);
        await new Promise((r) => setTimeout(r, 1000));
    }
    onTick(0);
    proc.disconnect();
    src.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const sampleRate = ctx.sampleRate;
    await ctx.close();
    const data = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
    let offset = 0;
    chunks.forEach((c) => {
        data.set(c, offset);
        offset += c.length;
    });
    return { sampleRate, data };
};

const toBase64Pcm = (data: Float32Array) => {
    const pcm = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) {
        pcm[i] = Math.max(-1, Math.min(1, data[i])) * 32767;
    }
    const bytes = new Uint8Array(pcm.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
};

const wavUrl = (pcm: Int16Array, sampleRate: number) => {
    const header = new ArrayBuffer(44);
    const v = new DataView(header);
    const w = (o: number, s: string) => s.split("").forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    w(0, "RIFF");
    v.setUint32(4, 36 + pcm.byteLength, true);
    w(8, "WAVE");
    w(12, "fmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, 1, true);
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    w(36, "data");
    v.setUint32(40, pcm.byteLength, true);
    return URL.createObjectURL(new Blob([header, pcm.buffer], { type: "audio/wav" }));
};

const postJson = async (path: string, body: object) => {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const json = await res.json();
    if (!res.ok || json.error) {
        throw new Error(json.error || `HTTP ${res.status}`);
    }
    return json;
};

export const CalibrationDialog = () => {
    const guiState = useGuiState();
    const { serverSetting } = useAppState();
    const [step, setStep] = useState<number>(0);
    const [countdown, setCountdown] = useState<number>(0);
    const [busy, setBusy] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [target, setTarget] = useState<string>("female");
    const [result, setResult] = useState<CalibrationResult | null>(null);
    const [applied, setApplied] = useState<boolean>(false);
    const [before, setBefore] = useState<string>("");
    const [after, setAfter] = useState<string>("");
    const noiseRef = useRef<Recording | null>(null);
    const voiceRef = useRef<Recording | null>(null);

    const s = serverSetting.serverSetting;
    const live = guiState.isConverting || s.serverAudioStated == 1;
    const deviceId = ["none", "file", "screen", "-1", "default", ""].includes(guiState.audioInputForGUI) ? null : guiState.audioInputForGUI;

    const close = () => {
        guiState.stateControls.showCalibrationCheckbox.updateState(false);
        setStep(0);
        setResult(null);
        setApplied(false);
        setError("");
        setBefore("");
        setAfter("");
    };

    const run = async (label: string, f: () => Promise<void>) => {
        setError("");
        setBusy(label);
        try {
            await f();
        } catch (e) {
            setError(`${e}`);
        } finally {
            setBusy("");
            setCountdown(0);
        }
    };

    const recordNoise = () =>
        run("rec", async () => {
            noiseRef.current = await recordMic(deviceId, NOISE_SECONDS, setCountdown);
            setStep(2);
        });

    const recordVoice = () =>
        run("rec", async () => {
            voiceRef.current = await recordMic(deviceId, VOICE_SECONDS, setCountdown);
            setBusy("analyze");
            const noise = noiseRef.current!;
            const voice = voiceRef.current;
            // both recordings come from the same AudioContext rate
            const res = await postJson("/calibrate", { sampleRate: voice.sampleRate, noise: toBase64Pcm(noise.data), voice: toBase64Pcm(voice.data), target: target });
            setResult(res);
            setStep(3);
        });

    const apply = () =>
        run("apply", async () => {
            if (!result) return;
            await serverSetting.updateServerSettings({ ...s, ...result.settings });
            setApplied(true);
        });

    const listen = () =>
        run("convert", async () => {
            const voice = voiceRef.current!;
            const pcm = new Int16Array(voice.data.length);
            voice.data.forEach((x, i) => (pcm[i] = Math.max(-1, Math.min(1, x)) * 32767));
            setBefore(wavUrl(pcm, voice.sampleRate));
            const res = await postJson("/convert_clip", { sampleRate: voice.sampleRate, voice: toBase64Pcm(voice.data) });
            const bin = atob(res.audio);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            setAfter(wavUrl(new Int16Array(bytes.buffer), res.sampleRate));
        });

    const dialog = useMemo(() => {
        const button = (label: string, onClick: () => void, disabled = false) => (
            <div className={`body-button ${disabled ? "body-button-disabled" : ""}`} onClick={() => !disabled && !busy && onClick()} style={disabled || busy ? { opacity: 0.4 } : {}}>
                {label}
            </div>
        );
        const status = busy == "rec" ? `Grabando... ${countdown} s` : busy == "analyze" ? "Analizando..." : busy == "apply" ? "Aplicando..." : busy == "convert" ? "Convirtiendo tu grabación..." : "";

        let body: JSX.Element;
        if (step == 0) {
            body = (
                <>
                    <div className="dialog-content-part">Te ayudo a ajustar la limpieza del ruido, el VAD y el tono en 2 grabaciones cortas. Usaré el micrófono elegido en «AUDIO IN».</div>
                    {live ? <div className="dialog-content-part">Primero para la conversión (botón «stop»), porque el asistente necesita el micrófono y el servidor libres.</div> : <></>}
                    <div className="body-button-container body-button-container-space-around">{button("Empezar", () => setStep(1), live)}</div>
                </>
            );
        } else if (step == 1) {
            body = (
                <>
                    <div className="dialog-content-part">
                        <b>Paso 1 de 2:</b> quédate en silencio {NOISE_SECONDS} segundos, con el ruido normal de tu cuarto (ventilador, tele, lo que haya siempre).
                    </div>
                    <div className="body-button-container body-button-container-space-around">{button("Grabar silencio", recordNoise)}</div>
                </>
            );
        } else if (step == 2) {
            body = (
                <>
                    <div className="dialog-content-part">
                        <b>Paso 2 de 2:</b> cuando pulses el botón, lee esto en voz alta, con tu voz normal ({VOICE_SECONDS} s):
                    </div>
                    <div className="dialog-content-part" style={{ fontStyle: "italic" }}>
                        {READING}
                    </div>
                    <div className="dialog-content-part">
                        La voz que vas a usar es:{" "}
                        <select className="body-select" value={target} onChange={(e) => setTarget(e.target.value)}>
                            <option value="female">femenina (más aguda)</option>
                            <option value="male">masculina (más grave)</option>
                            <option value="keep">no cambiar mi tono</option>
                        </select>
                    </div>
                    <div className="body-button-container body-button-container-space-around">{button("Grabar lectura", recordVoice)}</div>
                </>
            );
        } else {
            const r = result!;
            const st = r.settings;
            const m = r.measure;
            body = (
                <>
                    <div className="dialog-content-part">
                        <b>Esto es lo que medí:</b>
                        <ul>
                            <li>
                                Tu voz está {Math.round(m.snrDb)} dB por encima del ruido {m.snrDb >= 35 ? "(cuarto tranquilo)" : m.snrDb >= 20 ? "(algo de ruido)" : "(mucho ruido)"}.
                            </li>
                            {m.userPitchHz > 0 ? <li>Tu tono normal: {Math.round(m.userPitchHz)} Hz.</li> : <></>}
                        </ul>
                    </div>
                    <div className="dialog-content-part">
                        <b>Te propongo:</b>
                        <ul>
                            <li>Limpieza: {st.denoiseStrong == 1 ? "Strong (DeepFilterNet3)" : "normal (GTCRN)"}</li>
                            <li>
                                VAD: {st.vadThreshold} (deja pasar el {Math.round(m.speechPass * 100)}% de tu voz)
                            </li>
                            {st.tran !== undefined ? <li>Tono (TUNE): {st.tran > 0 ? `+${st.tran}` : st.tran}</li> : <></>}
                        </ul>
                    </div>
                    {r.tips.length > 0 ? (
                        <div className="dialog-content-part">
                            <b>Consejos:</b>
                            <ul>
                                {r.tips.map((t) => (
                                    <li key={t}>{TIPS[t] || t}</li>
                                ))}
                            </ul>
                        </div>
                    ) : (
                        <></>
                    )}
                    <div className="body-button-container body-button-container-space-around">
                        {button(applied ? "Aplicado ✓" : "Aplicar", apply, applied)}
                        {button("Escuchar antes y después", listen, !applied)}
                        {button("Repetir", () => {
                            setStep(1);
                            setApplied(false);
                            setBefore("");
                            setAfter("");
                        })}
                    </div>
                    {before ? (
                        <div className="dialog-content-part">
                            <div>Tu voz:</div>
                            <audio controls src={before}></audio>
                            {after ? (
                                <>
                                    <div>Convertida:</div>
                                    <audio controls src={after}></audio>
                                </>
                            ) : (
                                <></>
                            )}
                        </div>
                    ) : (
                        <></>
                    )}
                </>
            );
        }

        return (
            <div className="dialog-frame">
                <div className="dialog-title">Asistente</div>
                <div className="dialog-content">
                    {body}
                    {status ? <div className="dialog-content-part">{status}</div> : <></>}
                    {error ? <div className="dialog-content-part">Algo falló: {error}</div> : <></>}
                    <div className="body-button-container body-button-container-space-around">{button("Cerrar", close)}</div>
                </div>
            </div>
        );
    }, [step, countdown, busy, error, target, result, applied, before, after, live, s, deviceId]);
    return dialog;
};
