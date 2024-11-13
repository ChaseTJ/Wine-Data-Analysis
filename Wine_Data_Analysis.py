'''
Wine Data Analysis

Chase Johnson
'''

# %%
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# %% Importing wine rating from google sheet

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file("wine-analysis-441522-ec24032b1416.json", scopes=scopes)

gc = gspread.authorize(creds)

spreadsheet = gc.open("Wine Master Sheet")

worksheet = spreadsheet.worksheet("Wine Stats")

data = worksheet.get_all_records(head=2)

df = pd.DataFrame(data)
# %%
