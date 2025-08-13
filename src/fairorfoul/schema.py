from pydantic import BaseModel, Field
from typing import Optional

class CallEvent(BaseModel):
    match_id: str
    sport: str
    date: str
    referee_id: str
    referee_county: Optional[str] = None
    team_a: str
    team_a_county: Optional[str] = None
    team_a_ses: Optional[str] = None
    team_b: str
    team_b_county: Optional[str] = None
    team_b_ses: Optional[str] = None
    call_timestamp: str = Field(alias="Call Timestamp (MM:SS)")
    call_type: str = Field(alias="Call Type")
    call_against_team: str = Field(alias="Call Against Team")
    player_number: Optional[str] = Field(default=None, alias="Player Number")
    score_at_call: Optional[str] = Field(default=None, alias="Score at Call")
    location_zone: Optional[str] = Field(default=None, alias="Location on Pitch/Court")
