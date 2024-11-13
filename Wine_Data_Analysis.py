'''
Wine Data Analysis

Chase Johnson
'''

# %%
import gspread
import pandas as pd
import numpy as np
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

rating_data = pd.DataFrame(data)

rating_data.replace("", np.nan, inplace=True)

rating_data = rating_data.iloc[:, :-1]


# %% Importing wine rating data from a downloaded spreadsheet instead

# rating_data = pd.read_excel('Wine Master Sheet.xlsx', skiprows=1)

# rating_data = rating_data.iloc[:, :-1]

# # Fixes a problem where a 0.5 was rounding down, makes it so it matches data from google sheet
# rating_data['Average'] = np.round(rating_data['Average'] + 1e-9, 2)

# %% Checking if the methods of bringing in data work exactly the same

# diff = rating_data.compare(rating_data2)

# if diff.empty:
#     print("The DataFrames are identical.")
# else:
#     print("Differences found:")
#     print(diff)

# %% Reading in large wine data set

wine_data = pd.read_csv('WineDataset.csv')

# %% Finding who brings the best wine

average_rating_provider = rating_data.groupby('Provider')['Average'].mean().sort_values(ascending=False)

print('Average ratings for the wine brought by each person:')
for provider, average_rating in average_rating_provider.items():
    print(f'{provider}: {average_rating:.2f}')

# %%
