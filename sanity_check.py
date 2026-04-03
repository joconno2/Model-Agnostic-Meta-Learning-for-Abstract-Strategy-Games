import sqlite3
import pandas as pd

DB_PATH = "/home/melanie/Downloads/ChessNet/lichess_2016-03_elo1800_base300.sqlite"
con = sqlite3.connect(DB_PATH)

#Print tables
print (pd.read_sql_query("SELECT name FROM sqlite_master WHERE type ='table';", con))

print(pd.read_sql_query("PRAGMA table_info(positions);", con))

print (pd.read_sql_query("SELECT id, game_id, ply, turn, fen, move_uci, action_id, z FROM POSITIONS LIMIT 5;", con))

#Quantity of data
print(pd.read_sql_query("SELECT COUNT(*) AS n_positions FROM positions;", con))
print(pd.read_sql_query("SELECT COUNT(DISTINCT action_id) AS n_unique_actions FROM positions;", con))
print(pd.read_sql_query("SELECT MIN(z) as zmin, MAX(z) as zmax FROM positions;", con))
print(pd.read_sql_query("SELECT COUNT(DISTINCT fen) FROM positions;",con))

print(pd.read_sql_query("SELECT COUNT (*) AS total_rows, COUNT (DISTINCT fen) AS unique_fens,COUNT (*) * 1.0 / COUNT(DISTINCT fen) AS avg_rows_per_fen FROM positions;", con))



