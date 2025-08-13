import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import typer
from fairorfoul.io import load_calls_csv, save_processed
from fairorfoul.analysis import team_call_rates, county_alignment_bias
from fairorfoul.tagger import tag_video

app = typer.Typer()


@app.command()
def rates(csv_path: str, out_csv: str = "data/processed/team_rates.csv"):
    df = load_calls_csv(csv_path)
    out = team_call_rates(df)
    save_processed(out, out_csv)
    typer.echo(out_csv)


@app.command()
def alignment(csv_path: str, out_csv: str = "data/processed/county_alignment.csv"):
    df = load_calls_csv(csv_path)
    out = county_alignment_bias(df)
    save_processed(out, out_csv)
    typer.echo(out_csv)


@app.command()
def tag(
    video: str,
    sport: str,
    match_id: str,
    referee_id: str,
    team_a: str,
    team_b: str,
    out_csv: str = "data/processed/tags.csv",
):
    tag_video(video, sport, match_id, referee_id, team_a, team_b, out_csv)


if __name__ == "__main__":
    app()
