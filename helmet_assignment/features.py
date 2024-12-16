import pandas as pd

def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    # Create a unique identifier for game and play
    tracks["game_play"] = tracks["gameKey"].astype(str) + "_" + tracks["playID"].astype(str).str.zfill(6)
    tracks["time"] = pd.to_datetime(tracks["time"])
    
    # Map the snap time for each game_play
    snap_dict = tracks.query('event == "ball_snap"').groupby("game_play")["time"].first().to_dict()
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    
    # Map team names
    tracks["team"] = tracks["player"].str[0].replace({"H": "Home", "V": "Away"})
    
    # Calculate time offset from the snap
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).dt.total_seconds()
    
    # Calculate the estimated video frame
    tracks["est_frame"] = ((tracks["snap_offset"] * fps) + snap_frame).round().astype(int)
    
    return tracks
