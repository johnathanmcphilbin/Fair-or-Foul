import typer
from fairorfoul.io import load_calls_csv, save_processed
from fairorfoul.analysis import team_call_rates, county_alignment_bias

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

if __name__ == "__main__":
    app()
