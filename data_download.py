import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from tqdm import tqdm

def download_seasons(seasons):
    frames = []
    for season in seasons:
        print(f"Fetching season {season}")
        lg = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = lg.get_data_frames()[0]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print ("test")
    return out

if __name__ == "__main__":
    seasons = ["2024-25", "2023-24"]
    df = download_seasons(seasons)
    df.to_csv("data/raw/games_raw.csv", index=False)
    print("Saved to data/raw/games_raw.csv")
