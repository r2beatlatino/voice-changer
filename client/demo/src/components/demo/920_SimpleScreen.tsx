import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { useGuiState } from "./001_GuiStateProvider";
import { ModelSlotArea } from "./components2/100_ModelSlotArea";
import { DeviceArea } from "./components2/102-3_DeviceArea";
import { NOISE_PROFILES, currentNoiseProfile, noiseProfileValues, setUiMode } from "./noiseProfiles";

// The simple screen: pick a voice, turn it on, and three controls. Everything else stays in
// the advanced screen, one click away.

export const SimpleScreen = () => {
    const { serverSetting, initializedRef, start, stop } = useAppState();
    const guiState = useGuiState();
    const s = serverSetting.serverSetting;
    const update = serverSetting.updateServerSettings;

    const selected = useMemo(() => {
        if (s.modelSlotIndex == undefined) return undefined;
        if (s.modelSlotIndex == "Beatrice-JVS") return s.modelSlots.find((v) => v.slotIndex == "Beatrice-JVS");
        return s.modelSlots[s.modelSlotIndex as number];
    }, [s.modelSlotIndex, s.modelSlots]);
    const hasPitch = selected && ["RVC", "so-vits-svc-40", "DDSP-SVC", "Diffusion-SVC"].includes(selected.voiceChangerType);

    // as in the advanced screen: the audio context can only start after a click
    const [startAfterInit, setStartAfterInit] = useState<boolean>(false);
    useEffect(() => {
        if (!startAfterInit) return;
        guiState.setIsConverting(true);
        start();
    }, [startAfterInit]);

    const onStart = async () => {
        setWarning("");
        if (s.enableServerAudio == 0 && ["none", "-1", ""].includes(guiState.audioInputForGUI)) {
            setWarning("No hay micrófono elegido. Elígelo abajo, en «Micrófono y salida» → input.");
            return;
        }
        if (s.enableServerAudio == 1 && s.serverInputDeviceId == -1) {
            setWarning("No hay micrófono elegido. Elígelo abajo, en «Micrófono y salida» → input.");
            return;
        }
        if (s.enableServerAudio == 0) {
            if (!initializedRef.current) {
                while (!initializedRef.current) {
                    await new Promise<void>((resolve) => setTimeout(resolve, 500));
                }
                setStartAfterInit(true);
            } else {
                guiState.setIsConverting(true);
                await start();
            }
        } else {
            update({ ...s, serverAudioStated: 1 });
            guiState.setIsConverting(true);
        }
    };
    const onStop = async () => {
        guiState.setIsConverting(false);
        if (s.enableServerAudio == 0) {
            await stop();
        } else {
            update({ ...s, serverAudioStated: 0 });
        }
    };

    const profile = currentNoiseProfile(s);
    const on = guiState.isConverting;
    // until the first /info arrives the settings are the client's defaults, not the user's
    const loaded = s.voiceChangerParams.model_dir != "";
    const [warning, setWarning] = useState<string>("");

    if (!loaded) {
        return (
            <div className="simple-screen">
                <div className="simple-title">Cambiador de voz</div>
                <div className="simple-hint">Conectando con el servidor…</div>
            </div>
        );
    }

    return (
        <div className="simple-screen">
            <div className="simple-header">
                <div className="simple-title">Cambiador de voz</div>
                <div
                    className={`simple-link ${on ? "simple-disabled" : ""}`}
                    title={on ? "Apaga la conversión antes de cambiar de pantalla" : ""}
                    onClick={() => !on && setUiMode("advanced")}
                >
                    Modo avanzado
                </div>
            </div>
            <div className="simple-credits">
                Basado en{" "}
                <a href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                    w-okada voice-changer
                </a>{" "}
                (
                <a href="https://github.com/w-okada/voice-changer/blob/master/LICENSE-NOTICE" target="_blank" rel="noopener noreferrer">
                    licencias
                </a>
                )
            </div>

            <div className="simple-card">
                <div className="simple-card-title">1. Elige una voz</div>
                <ModelSlotArea simple={true}></ModelSlotArea>
                {loaded && !selected ? <div className="simple-hint">Todavía no hay ninguna voz. Pulsa «Añadir o cambiar voces».</div> : <></>}
            </div>

            <div className="simple-card">
                <div className="simple-card-title">2. Enciéndelo</div>
                <div className={`simple-power ${on ? "simple-power-on" : ""}`} onClick={on ? onStop : onStart}>
                    {on ? "Apagar" : "Encender"}
                </div>
                <div className="simple-hint">{warning ? warning : on ? `Convirtiendo tu voz${selected ? ` en «${selected.name}»` : ""}.` : "Apagado: tu voz no se está convirtiendo."}</div>
            </div>

            <div className="simple-card">
                <div className="simple-card-title">3. Ajusta</div>
                {hasPitch ? (
                    <div className="simple-row">
                        <div className="simple-label">Tono</div>
                        <span className="simple-end">más grave</span>
                        <input type="range" min={-24} max={24} step={1} value={s.tran} onChange={(e) => update({ ...s, tran: Number(e.target.value) })} />
                        <span className="simple-end">más agudo</span>
                        <span className="simple-value">{s.tran > 0 ? `+${s.tran}` : s.tran}</span>
                    </div>
                ) : (
                    <></>
                )}
                <div className="simple-row">
                    <div className="simple-label">Ruido</div>
                    <select
                        className="body-select"
                        value={profile}
                        onChange={(e) => {
                            const values = noiseProfileValues(Number(e.target.value));
                            if (values) update({ ...s, ...values });
                        }}
                    >
                        {profile == -1 ? <option value={-1}>Personalizado</option> : <></>}
                        {NOISE_PROFILES.map((p, i) => (
                            <option key={p.name} value={i}>
                                {p.name}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="simple-row">
                    <div className="simple-label">Silencio</div>
                    <span className="simple-end">corta menos</span>
                    <input type="range" min={0.2} max={0.6} step={0.05} value={s.vadThreshold} disabled={s.vad != 1} onChange={(e) => update({ ...s, vadThreshold: Number(e.target.value) })} />
                    <span className="simple-end">quita más ruido</span>
                    <span className="simple-value">{s.vad == 1 ? s.vadThreshold : "off"}</span>
                </div>
                <div className="simple-hint">«Silencio» decide cuándo te callas. Si se cortan palabras, muévelo a la izquierda; si se oye ruido cuando no hablas, a la derecha.</div>
                <div className="simple-buttons">
                    <div className="simple-button" onClick={() => guiState.stateControls.showCalibrationCheckbox.updateState(true)}>
                        Asistente: ajústalo por mí
                    </div>
                </div>
            </div>

            {/* always mounted: it holds the audio elements that play the converted voice */}
            <div className="simple-card">
                <div className="simple-card-title">Micrófono y salida</div>
                <div className="simple-hint">Elige tu micrófono en «input» y dónde quieres oír o enviar la voz convertida en «output».</div>
                <div className="config-area simple-devices">
                    <DeviceArea simple={true}></DeviceArea>
                </div>
            </div>
        </div>
    );
};
