import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import joblib
import numpy as np
import sys, os

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

OUTLIER_IMG_PATH = resource_path("Images/iso_outliers.png")
TOP20_PATH = resource_path("Data/top20_iso_outliers.csv")
TEAM_SEASON_PATH = resource_path("Data/teams_season_cleaned.txt")
ISO_PATH = resource_path("Models/isoForest_model.pkl")
STACK_PATH = resource_path("Models/stack_model.pkl")
SCALER_PATH = resource_path("Models/scaler.pkl")
TEAMS_PATH = resource_path("Data/teams.txt")

top20_df = pd.read_csv(TOP20_PATH)

# Load prediction models
isoForest = joblib.load(ISO_PATH)
stack_model = joblib.load(STACK_PATH)
scaler = joblib.load(SCALER_PATH)

player_data = pd.read_csv(TEAMS_PATH)

teams = sorted(player_data["team"].unique())

team_season = pd.read_csv(TEAM_SEASON_PATH)

num_cols = [c for c in team_season.columns if c not in ['team','year','leag']]
team_season[num_cols] = team_season[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Compute win percentage
team_season['win_pct'] = team_season['won'] / (team_season['won'] + team_season['lost']).replace(0, np.nan)
team_season['win_pct'] = team_season['win_pct'].fillna(0)

# Main window
root = tk.Tk()
root.title("NBA Prediction")
root.geometry("500x500")

def open_outlier_window():
    out_win = tk.Toplevel()
    out_win.title("Outlier Prediction")
    out_win.geometry("900x800")

    # Load and display image
    img = Image.open(OUTLIER_IMG_PATH)
    img = img.resize((750, 500))
    photo = ImageTk.PhotoImage(img)

    img_label = tk.Label(out_win, image=photo)
    img_label.image = photo
    img_label.pack(pady=10)

    # Show top 20 players
    text = tk.Text(out_win, height=12, width=90)
    text.insert(tk.END, top20_df.to_string(index=False))
    text.pack()

    tk.Button(out_win, text="Back", command=out_win.destroy).pack(pady=10)

def open_prediction_window(team_a, team_b):
    if team_a == "" or team_b == "":
        tk.messagebox.showwarning("Input Error", "Please select both teams!")
        return
    if team_a == team_b:
        tk.messagebox.showwarning("Input Error", "Please select different teams!")
        return
    
    try:
        team_a = name_to_team[team_a]
        team_b = name_to_team[team_b]
    except KeyError:
        tk.messagebox.showerror("Error", "Invalid team selected.")
        return

    pred_win = tk.Toplevel()
    pred_win.title("Game Outcome Prediction")
    pred_win.geometry("400x250")

    # Get all rows for the selected teams across all seasons
    ra_df = team_season[team_season['team'] == team_a]
    rb_df = team_season[team_season['team'] == team_b]

    # Check if any data exists
    if ra_df.empty or rb_df.empty:
        tk.messagebox.showerror("Error", "Selected team data not available.")
        return

    # Take the average of features over all seasons
    feature_cols = [c for c in team_season.columns if c not in ['team','year','leag','won','lost','win_pct']]
    ra_avg = ra_df[feature_cols].mean()
    rb_avg = rb_df[feature_cols].mean()

    # Compute diff features
    diff = (ra_avg - rb_avg).to_frame().T

    # Add 'diff_' prefix to match scaler columns
    diff.columns = ['diff_' + str(c) for c in diff.columns]

    # Scale
    diff_scaled = scaler.transform(diff)

    # Predict probability
    gb_pred = stack_model.predict_proba(diff_scaled)[0][1]
    winner = team_a if gb_pred > 0.5 else team_b
    team_a_prob = gb_pred
    team_b_prob = 1 - gb_pred

    team_to_name = {v: k for k, v in name_to_team.items()}

    tk.Label(pred_win, text="Game Outcome Prediction", font=("Arial", 16)).pack(pady=10)
    tk.Label(pred_win, text=f"{team_to_name[team_a]} vs {team_to_name[team_b]}", font=("Arial", 14)).pack()
    tk.Label(pred_win, text=f"\nWinner: {team_to_name[winner]}", font=("Arial", 16)).pack()
    tk.Label(pred_win, text=f"{team_to_name[team_a]} Win Probability: {team_a_prob*100:.2f}%").pack(pady=2)
    tk.Label(pred_win, text=f"{team_to_name[team_b]} Win Probability: {team_b_prob*100:.2f}%").pack(pady=2)
    tk.Button(pred_win, text="Back", command=pred_win.destroy).pack(pady=15)

tk.Label(root, text="NBA Prediction", font=("Arial", 22)).pack(pady=20)

# Outlier Section
frame1 = tk.Frame(root, bd=2, relief=tk.RIDGE, padx=20, pady=20)
frame1.pack(pady=15)

tk.Label(frame1, text="Outliers", font=("Arial", 16)).pack()
tk.Button(frame1, text="View", command=open_outlier_window).pack(pady=5)

# Game Prediction Section
frame2 = tk.Frame(root, bd=2, relief=tk.RIDGE, padx=20, pady=20)
frame2.pack(pady=15)

tk.Label(frame2, text="Game Outcome Prediction", font=("Arial", 16)).pack()

name_to_team = dict(zip(player_data['name'], player_data['team']))
team_names = sorted(player_data['name'].dropna().unique())

team_a_var = tk.StringVar()
team_b_var = tk.StringVar()

ttk.Combobox(frame2, textvariable=team_a_var, values=team_names).pack(pady=5)
ttk.Combobox(frame2, textvariable=team_b_var, values=team_names).pack(pady=5)

tk.Button(
    frame2, text="GO",
    command=lambda: open_prediction_window(
        team_a_var.get(),
        team_b_var.get()
    )
).pack(pady=5)

root.mainloop()