export type TutorStyle = "casual" | "balanced" | "strict";
export type ReplyLanguage = "jp" | "jp_en" | "en";
export type CorrectionIntensity = "light" | "medium" | "heavy";
export type ResponseLength = "very_short" | "short" | "detailed";

export interface UserSettings {
  tutorStyle: TutorStyle;
  replyLanguage: ReplyLanguage;
  correctionIntensity: CorrectionIntensity;
  responseLength: ResponseLength;
  showLiveTranscript: boolean;
  autoPlayAssistantVoice: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: string;
  kind?: "text" | "voice";
  sttText?: string;
  audioUrl?: string;
  audioDurationSec?: number;
}
