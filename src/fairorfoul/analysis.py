import pandas as pd

def team_call_rates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["Sport","Referee ID","Call Against Team"]).size().reset_index(name="calls")
    t = g.groupby(["Sport","Referee ID"])["calls"].transform("sum")
    g["rate"] = g["calls"] / t
    return g.sort_values(["Sport","Referee ID","rate"], ascending=[True,True,False])

def county_alignment_bias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ref_team_alignment"] = (df["Referee County"] == df["Team A County"]).where(df["Call Against Team"]==df["Team A"], False) | \
                               (df["Referee County"] == df["Team B County"]).where(df["Call Against Team"]==df["Team B"], False)
    out = df.groupby(["Sport","Referee ID","ref_team_alignment"]).size().reset_index(name="calls")
    return out
