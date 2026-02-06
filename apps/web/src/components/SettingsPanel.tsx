import { UserSettings } from "../types";

interface SettingsPanelProps {
  isOpen: boolean;
  settings: UserSettings;
  onClose: () => void;
  onChange: (next: UserSettings) => void;
}

const updateSettings = (
  settings: UserSettings,
  onChange: (next: UserSettings) => void,
  patch: Partial<UserSettings>
) => onChange({ ...settings, ...patch });

export default function SettingsPanel({
  isOpen,
  settings,
  onClose,
  onChange
}: SettingsPanelProps) {
  return (
    <aside className={`settings-panel ${isOpen ? "open" : ""}`}>
      <div className="panel-head">
        <h2>Learning Settings</h2>
        <button onClick={onClose} aria-label="Close settings">
          Close
        </button>
      </div>

      <label>
        Tutor Style
        <select
          value={settings.tutorStyle}
          onChange={(event) =>
            updateSettings(settings, onChange, {
              tutorStyle: event.target.value as UserSettings["tutorStyle"]
            })
          }
        >
          <option value="casual">Casual conversation</option>
          <option value="balanced">Conversation + correction</option>
          <option value="strict">Strict teacher mode</option>
        </select>
      </label>

      <label>
        Reply Language
        <select
          value={settings.replyLanguage}
          onChange={(event) =>
            updateSettings(settings, onChange, {
              replyLanguage: event.target.value as UserSettings["replyLanguage"]
            })
          }
        >
          <option value="jp">Japanese only</option>
          <option value="jp_en">Japanese + short English support</option>
          <option value="en">English first</option>
        </select>
      </label>

      <label>
        Correction Intensity
        <select
          value={settings.correctionIntensity}
          onChange={(event) =>
            updateSettings(settings, onChange, {
              correctionIntensity: event.target
                .value as UserSettings["correctionIntensity"]
            })
          }
        >
          <option value="light">Light</option>
          <option value="medium">Medium</option>
          <option value="heavy">Heavy</option>
        </select>
      </label>

      <label>
        Response Length
        <select
          value={settings.responseLength}
          onChange={(event) =>
            updateSettings(settings, onChange, {
              responseLength: event.target.value as UserSettings["responseLength"]
            })
          }
        >
          <option value="very_short">Very short (1-2 sentences)</option>
          <option value="short">Short (2-4 sentences)</option>
          <option value="detailed">Detailed explanation</option>
        </select>
      </label>

      <label className="checkbox-line">
        <input
          type="checkbox"
          checked={settings.showLiveTranscript}
          onChange={(event) =>
            updateSettings(settings, onChange, {
              showLiveTranscript: event.target.checked
            })
          }
        />
        Show live transcript while recording
      </label>

      <label className="checkbox-line">
        <input
          type="checkbox"
          checked={settings.autoPlayAssistantVoice}
          onChange={(event) =>
            updateSettings(settings, onChange, {
              autoPlayAssistantVoice: event.target.checked
            })
          }
        />
        Auto-play assistant voice
      </label>
    </aside>
  );
}
