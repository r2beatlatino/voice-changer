import React from "react";
import { GuiStateProvider } from "./001_GuiStateProvider";
import { Dialogs } from "./900_Dialogs";
import { ModelSlotControl } from "./b00_ModelSlotControl";
import { Dialogs2 } from "./910_Dialogs2";
import { SimpleScreen } from "./920_SimpleScreen";
import { getUiMode } from "./noiseProfiles";

export const Demo = () => {
    const simple = getUiMode() == "simple";
    return (
        <GuiStateProvider>
            <div className="main-body">
                <Dialogs2 />
                <Dialogs />
                {simple ? <SimpleScreen></SimpleScreen> : <ModelSlotControl></ModelSlotControl>}
            </div>
        </GuiStateProvider>
    );
};
