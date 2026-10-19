import React, { useMemo } from "react";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { F0Detector } from "@dannadori/voice-changer-client-js";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
import { useGuiState } from "../001_GuiStateProvider";
import { NOISE_PROFILES, currentNoiseProfile, noiseProfileValues } from "../noiseProfiles";

export type QualityAreaProps = {
    detectors: string[];
};

export const QualityArea = (props: QualityAreaProps) => {
    const { setVoiceChangerClientSetting, serverSetting, setting, webEdition } = useAppState();
    const { appGuiSettingState } = useAppRoot();
    const edition = appGuiSettingState.edition;

    const { stateControls } = useGuiState();
    const qualityArea = useMemo(() => {
        if (!serverSetting.updateServerSettings || !setVoiceChangerClientSetting || !serverSetting.serverSetting || !setting) {
            return <></>;
        }

        const generateF0DetOptions = () => {
            if (edition.indexOf("onnxdirectML-cuda") >= 0) {
                const recommended = ["crepe_tiny", "rmvpe_onnx"];
                return Object.values(props.detectors).map((x) => {
                    if (recommended.includes(x)) {
                        return (
                            <option key={x} value={x}>
                                {x}
                            </option>
                        );
                    } else {
                        return (
                            <option key={x} value={x} disabled>
                                {x}(N/A)
                            </option>
                        );
                    }
                });
            } else {
                return Object.values(props.detectors).map((x) => {
                    return (
                        <option key={x} value={x}>
                            {x}
                        </option>
                    );
                });
            }
        };
        const f0DetOptions = generateF0DetOptions();

        const f0Det = webEdition ? (
            <></>
        ) : (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">F0 Det.:</div>
                <div className="config-sub-area-control-field">
                    <select
                        className="body-select"
                        value={serverSetting.serverSetting.f0Detector}
                        onChange={(e) => {
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, f0Detector: e.target.value as F0Detector });
                        }}
                    >
                        {f0DetOptions}
                    </select>
                </div>
            </div>
        );

        const threshold = webEdition ? (
            <></>
        ) : (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">S.Thresh.:</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-slider-control">
                        <span className="config-sub-area-slider-control-kind"></span>
                        <span className="config-sub-area-slider-control-slider">
                            <input
                                type="range"
                                className="config-sub-area-slider-control-slider"
                                min="0.00000"
                                max="0.001"
                                step="0.00001"
                                value={serverSetting.serverSetting.silentThreshold || 0}
                                onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, silentThreshold: Number(e.target.value) });
                                }}
                            ></input>
                        </span>
                        <span className="config-sub-area-slider-control-val">{serverSetting.serverSetting.silentThreshold}</span>
                    </div>
                </div>
            </div>
        );

        // server-side noise robustness (RVC). Works in server-audio mode too, unlike Sup1/Sup2.
        const slider = (title: string, enabled: number, onToggle: (v: number) => void, value: number, min: number, max: number, step: number, onValue: (v: number) => void) => (
            <div className="config-sub-area-control">
                <div className="config-sub-area-control-title">{title}</div>
                <div className="config-sub-area-control-field">
                    <div className="config-sub-area-slider-control">
                        <span className="config-sub-area-slider-control-kind">
                            <input type="checkbox" checked={enabled == 1} onChange={(e) => onToggle(e.target.checked ? 1 : 0)} />
                        </span>
                        <span className="config-sub-area-slider-control-slider">
                            <input type="range" className="config-sub-area-slider-control-slider" min={min} max={max} step={step} value={value} disabled={enabled != 1} onChange={(e) => onValue(Number(e.target.value))}></input>
                        </span>
                        <span className="config-sub-area-slider-control-val">{value}</span>
                    </div>
                </div>
            </div>
        );
        const s = serverSetting.serverSetting;
        const update = serverSetting.updateServerSettings;
        const currentProfile = currentNoiseProfile(s);
        const robust = webEdition ? (
            <></>
        ) : (
            <>
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">Perfil:</div>
                    <div className="config-sub-area-control-field">
                        <select
                            className="body-select"
                            value={currentProfile}
                            onChange={(e) => {
                                const values = noiseProfileValues(Number(e.target.value));
                                if (values) update({ ...s, ...values });
                            }}
                        >
                            {currentProfile == -1 ? <option value={-1}>Personalizado</option> : <></>}
                            {NOISE_PROFILES.map((p, i) => (
                                <option key={p.name} value={i}>
                                    {p.name}
                                </option>
                            ))}
                        </select>
                        <div className="config-sub-area-buttons">
                            <div onClick={() => stateControls.showCalibrationCheckbox.updateState(true)} className="config-sub-area-button">
                                Asistente
                            </div>
                        </div>
                    </div>
                </div>
                {slider(
                    "Denoise:",
                    s.denoise,
                    (v) => update({ ...s, denoise: v }),
                    s.denoiseStrength,
                    0,
                    1,
                    0.05,
                    (v) => update({ ...s, denoiseStrength: v }),
                )}
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">Strong:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-noise-container">
                            <div className="config-sub-area-noise-checkbox-container">
                                <input type="checkbox" checked={s.denoiseStrong == 1} disabled={s.denoise != 1} onChange={(e) => update({ ...s, denoiseStrong: e.target.checked ? 1 : 0 })} /> <span>DeepFilterNet3</span>
                            </div>
                        </div>
                    </div>
                </div>
                {slider(
                    "VAD:",
                    s.vad,
                    (v) => update({ ...s, vad: v }),
                    s.vadThreshold,
                    0.1,
                    0.9,
                    0.05,
                    (v) => update({ ...s, vadThreshold: v }),
                )}
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">F0 Fix:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-noise-container">
                            <div className="config-sub-area-noise-checkbox-container">
                                <input type="checkbox" checked={s.f0Stabilize == 1} onChange={(e) => update({ ...s, f0Stabilize: e.target.checked ? 1 : 0 })} /> <span>on</span>
                            </div>
                        </div>
                    </div>
                </div>
            </>
        );

        return (
            <div className="config-sub-area">
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">NOISE:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-noise-container">
                            <div className="config-sub-area-noise-checkbox-container">
                                <input
                                    type="checkbox"
                                    disabled={serverSetting.serverSetting.enableServerAudio != 0}
                                    checked={setting.voiceChangerClientSetting.echoCancel}
                                    onChange={(e) => {
                                        try {
                                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, echoCancel: e.target.checked });
                                        } catch (e) {
                                            console.error(e);
                                        }
                                    }}
                                />{" "}
                                <span>Echo</span>
                            </div>
                            <div className="config-sub-area-noise-checkbox-container">
                                <input
                                    type="checkbox"
                                    disabled={serverSetting.serverSetting.enableServerAudio != 0}
                                    checked={setting.voiceChangerClientSetting.noiseSuppression}
                                    onChange={(e) => {
                                        try {
                                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, noiseSuppression: e.target.checked });
                                        } catch (e) {
                                            console.error(e);
                                        }
                                    }}
                                />{" "}
                                <span>Sup1</span>
                            </div>
                            <div className="config-sub-area-noise-checkbox-container">
                                <input
                                    type="checkbox"
                                    disabled={serverSetting.serverSetting.enableServerAudio != 0}
                                    checked={setting.voiceChangerClientSetting.noiseSuppression2}
                                    onChange={(e) => {
                                        try {
                                            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, noiseSuppression2: e.target.checked });
                                        } catch (e) {
                                            console.error(e);
                                        }
                                    }}
                                />{" "}
                                <span>Sup2</span>
                            </div>
                        </div>
                    </div>
                </div>
                {f0Det}
                {threshold}
                {robust}
            </div>
        );
    }, [serverSetting.serverSetting, setting, serverSetting.updateServerSettings, setVoiceChangerClientSetting]);

    return qualityArea;
};
