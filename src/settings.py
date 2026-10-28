from dataclasses import dataclass, fields
import os


@dataclass
class Settings:
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    deepgram_api_key: str
    openai_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    public_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(**{f.name: os.environ[f.name.upper()] for f in fields(cls)})
