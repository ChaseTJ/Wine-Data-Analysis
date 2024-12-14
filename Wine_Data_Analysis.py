'''
Wine Data Analysis

Chase Johnson
'''

# %%
import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
import seaborn as sns

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

# %% Ratings data cleaning

# dropping anyone who has rated less than 5 wines

# Identify the range of rater columns
start_col = rating_data.columns.get_loc('Reviewed On') + 1
end_col = rating_data.columns.get_loc('Average')

# Extract rater columns
rater_columns = rating_data.columns[start_col:end_col]

# Count the number of non-null ratings for each rater
rater_counts = rating_data[rater_columns].notnull().sum()

# Identify raters with at least 5 ratings
valid_raters = rater_counts[rater_counts >= 5].index

# Create a list of columns to keep (non-rater columns + valid rater columns)
columns_to_keep = list(rating_data.columns[:start_col]) + list(valid_raters) + list(rating_data.columns[end_col:])

# Filter the DataFrame to keep only the valid raters and non-rater columns
filtered_rating_data = rating_data[columns_to_keep]

# Check the results
print(f"Original number of raters: {len(rater_columns)}")
print(f"Filtered number of raters: {len(valid_raters)}")

# Optionally, reassign filtered data back to the main DataFrame
rating_data = filtered_rating_data

# %% Reading in large wine data set

wine_data = pd.read_csv('WineDataset.csv')

# Extract the numeric part of the ABV column and convert to float
wine_data['ABV'] = wine_data['ABV'].str.extract(r'([\d\.]+)').astype(float)

# Verify the cleaned ABV column
# print("Cleaned ABV column:")
# print(wine_data['ABV'].head())


# %% Creating a dataframe for data specifically on the raters

# Dynamically generate the list of valid raters based on the filtered columns
valid_raters_list = list(valid_raters)  # This comes from the filtering step

# Create the DataFrame for stats about each rater
raters_data = pd.DataFrame({'Raters': valid_raters_list})

# Display the new raters_data DataFrame
# print(raters_data)


# %% Finding who brings the best wine

average_rating_provider = rating_data.groupby('Provider')['Average'].mean().sort_values(ascending=False)

raters_data['Average Provided Wine Rating'] = raters_data['Raters'].map(average_rating_provider)

print('Average ratings for the wine brought by each person:')
for provider, average_rating in average_rating_provider.items():
    print(f'{provider}: {average_rating:.2f}')
    
# %% Finding if there is a bias towards rating the wines you brought higher than others

# # may need to revisit with normalized ratings, people who consistenty rate higher will show as biased here

# rating_bias = rating_data.groupby('Provider')['Average'].mean()

# %% rater specific data

# Identify common grapes (at least 3 wines)
common_grapes = rating_data['Type'].value_counts()
common_grapes = common_grapes[common_grapes >= 3].index

# Function to calculate stats for each rater
def calculate_rater_stats(rater_name, data, common_grapes):
    rater_column = rater_name  # Column corresponding to the rater's name
    rater_data = data[['Name', 'Red or White', 'Type', rater_column]].dropna(subset=[rater_column])  # Filter wines rated by this rater
    
    # Average rating by the rater
    avg_rating = rater_data[rater_column].mean()
    
    # Favorite wine type (red/white) based on average ratings
    favorite_red_white = rater_data.groupby('Red or White')[rater_column].mean().idxmax()
    
    # Favorite grape variety based on average ratings
    favorite_grape = rater_data.groupby('Type')[rater_column].mean().idxmax()
    
    # Favorite common grape variety
    common_grape_data = rater_data[rater_data['Type'].isin(common_grapes)]
    if not common_grape_data.empty:
        favorite_common_grape = common_grape_data.groupby('Type')[rater_column].mean().idxmax()
    else:
        favorite_common_grape = None  # No common grapes rated
    
    # Favorite wine overall
    favorite_wine = rater_data.loc[rater_data[rater_column].idxmax(), 'Name']
    
    # Least favorite wine overall
    least_favorite_wine = rater_data.loc[rater_data[rater_column].idxmin(), 'Name']
    
    # Average ratings for red and white wines
    red_rating = rater_data[rater_data['Red or White'] == 'Red'][rater_column].mean()
    white_rating = rater_data[rater_data['Red or White'] == 'White'][rater_column].mean()
    
    return avg_rating, favorite_red_white, favorite_grape, favorite_common_grape, favorite_wine, least_favorite_wine, red_rating, white_rating

# Populate the raters_data DataFrame with stats
raters_data[['Average Rating', 'Favorite Red or White', 'Favorite Grape', 
             'Favorite Common Grape', 'Favorite Wine', 'Least Favorite Wine', 
             'Average Red Wine Rating', 'Average White Wine Rating']] = raters_data['Raters'].apply(
    lambda rater: pd.Series(calculate_rater_stats(rater, rating_data, common_grapes))
)

# Reorder the columns logically
raters_data = raters_data[
    ['Raters', 'Average Rating', 'Average Red Wine Rating', 'Average White Wine Rating',
     'Favorite Red or White', 'Favorite Grape', 'Favorite Common Grape',
     'Favorite Wine', 'Least Favorite Wine', 'Average Provided Wine Rating']
]

# Display the updated DataFrame
print(raters_data)

# %% Basic data visualization

# Ensure numeric data is properly cast
rating_data['Year'] = pd.to_numeric(rating_data['Year'], errors='coerce')
rating_data['ABV (%)'] = pd.to_numeric(rating_data['ABV (%)'], errors='coerce')

# Plot correlation between ratings and Type (grape variety)
plt.figure(figsize=(10, 6))
sns.boxplot(data=rating_data, x='Type', y='Average')
plt.title("Correlation Between Grape Type and Average Rating")
plt.xticks(rotation=45)
plt.xlabel("Grape Type")
plt.ylabel("Average Rating")
plt.show()

# Plot correlation between ratings and Red or White
plt.figure(figsize=(8, 6))
sns.boxplot(data=rating_data, x='Red or White', y='Average')
plt.title("Correlation Between Red/White and Average Rating")
plt.xlabel("Red or White")
plt.ylabel("Average Rating")
plt.show()

# Scatterplot for Year vs Average Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(data=rating_data, x='Year', y='Average')
plt.title("Correlation Between Year and Average Rating")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.show()

# Barplot for Region vs Average Rating
plt.figure(figsize=(12, 6))
sns.barplot(data=rating_data, x='Region', y='Average', ci=None)
plt.title("Correlation Between Region and Average Rating")
plt.xticks(rotation=45)
plt.xlabel("Region")
plt.ylabel("Average Rating")
plt.show()

# Scatterplot for ABV (%) vs Average Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(data=rating_data, x='ABV (%)', y='Average')
plt.title("Correlation Between ABV (%) and Average Rating")
plt.xlabel("ABV (%)")
plt.ylabel("Average Rating")
plt.show()


# %% Basic rating correlations

# mainly looking at type, red or white, year, region, abv, ratings

# starting with numerical values for ease

rating_data[['Year', 'Average']].corr()

rating_data[['ABV (%)', 'Average']].corr()

rating_data.corr()

# %%

# Select relevant columns for the analysis
columns_to_keep = ['Average', 'Year', 'ABV (%)', 'Type', 'Red or White', 'Region'] + valid_raters_list

# Filter the dataset to include only these columns
rating_data_subset = rating_data[columns_to_keep]

# Create dummy variables for categorical columns
rating_data_dummies = pd.get_dummies(rating_data_subset, columns=['Type', 'Red or White', 'Region'], drop_first=True)

# Verify the columns in the updated DataFrame
print("Columns after adding dummies:")
# print(rating_data_dummies.columns)

import statsmodels.api as sm

# Features (X) and target (y) for the average rating
X = rating_data_dummies.drop(columns=['Average'] + valid_raters_list)  # Exclude target and individual rater columns
y = rating_data_dummies['Average']

# Impute missing values in X with the column mean
X = X.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col.fillna(col.mode().iloc[0]))

# Impute missing values in y with the mean
y = y.fillna(y.mean())

X['Year'] = X['Year'].fillna(X['Year'].mean())
X['ABV (%)'] = X['ABV (%)'].fillna(X['ABV (%)'].mean())


# Re-add constant after cleaning
X = sm.add_constant(X)

# Fit regression model
model = sm.OLS(y, X).fit()

# Display summary
print(model.summary())

# Add predicted ratings for average to the original DataFrame
rating_data['Predicted_Average'] = model.predict(X)

# Collect R-squared for visualization
regression_results = []
average_r_squared = model.rsquared
regression_results.append({'Rater': 'Average', 'R-squared': average_r_squared, 'Coefficients': model.params})

for rater in valid_raters_list:
    print(f"\nRegression for {rater}:")
    
    # Target variable for the current rater
    y_rater = rating_data_dummies[rater]

    # Ensure no missing values in the rater-specific target variable
    y_rater = y_rater.fillna(y_rater.mean())
    
    # Add constant to the features
    X = sm.add_constant(X, has_constant='add')
    
    # Fit the regression model
    model = sm.OLS(y_rater, X).fit()
    
    # Display the regression summary
    print(model.summary())

    # Store R-squared value for visualization
    regression_results.append({'Rater': rater, 'R-squared': model.rsquared, 'Coefficients': model.params})

    # Add predicted ratings for the rater to the original DataFrame
    rating_data[f'Predicted_{rater}'] = model.predict(X)
    
# Step 6: Visualize Regression Accuracies
results_df = pd.DataFrame(regression_results)

# Bar plot for R-squared values
plt.figure(figsize=(10, 6))
plt.bar(results_df['Rater'], results_df['R-squared'], color='skyblue')
plt.title('Regression R-squared Values for Each Rater')
plt.xlabel('Rater')
plt.ylabel('R-squared Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter plot for comparison of individual R-squared values vs. average
plt.figure(figsize=(8, 6))
plt.scatter(results_df['Rater'], results_df['R-squared'], color='orange', label='Individual Raters')
plt.axhline(average_r_squared, color='blue', linestyle='--', label='Average R-squared')
plt.title('Comparison of R-squared Values')
plt.xlabel('Rater')
plt.ylabel('R-squared Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# %%
coefficients_df = pd.DataFrame({
    result['Rater']: result['Coefficients'].drop('const', errors='ignore')
    for result in regression_results if result['Rater'] != 'Average'
}).T

plt.figure(figsize=(12, 8))
for rater in coefficients_df.index:
    plt.scatter(coefficients_df.columns, coefficients_df.loc[rater], label=rater)

plt.title('Overlayed Coefficients for Predictors by Rater (Points Only)')
plt.xlabel('Predictor')
plt.ylabel('Coefficient Value')
plt.legend(title='Rater', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(coefficients_df.columns, coefficients_df.columns, rotation=45, ha='right')
plt.tight_layout()
plt.show()
# %%
# Prepare DataFrame for boxplot
coefficients_df = pd.DataFrame({
    result['Rater']: result['Coefficients'].drop('const', errors='ignore')
    for result in regression_results if result['Rater'] != 'Average'
}).T

# Melt the DataFrame for easier plotting with Seaborn
coefficients_melted = coefficients_df.reset_index().melt(id_vars='index', var_name='Predictor', value_name='Coefficient')

# Create the boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(data=coefficients_melted, x='Predictor', y='Coefficient')
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Add horizontal line at y=0

plt.title('Boxplot of Coefficients for Predictors Across Raters')
plt.xlabel('Predictor')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Compare Predicted vs Actual for Average Ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rating_data['Average'], y=rating_data['Predicted_Average'], alpha=0.6, color='blue')
plt.plot([rating_data['Average'].min(), rating_data['Average'].max()],
         [rating_data['Average'].min(), rating_data['Average'].max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Ratings (Average)')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.legend()
plt.tight_layout()
plt.show()

# Compare Predicted vs Actual for Individual Raters
for rater in valid_raters_list:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=rating_data[rater], y=rating_data[f'Predicted_{rater}'], alpha=0.6)
    plt.plot([rating_data[rater].min(), rating_data[rater].max()],
             [rating_data[rater].min(), rating_data[rater].max()],
             color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title(f'Actual vs Predicted Ratings ({rater})')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
# Replicate original preprocessing steps so that wine_data matches the model expectations.

# 1. Convert Vintage to Year if originally the model used 'Year'
wine_data['Year'] = wine_data['Vintage']
mean_year = X['Year'].mean()  # If you have access to the training set (X)
wine_data['Year'] = wine_data['Year'].replace('NV', mean_year)
# Convert to numeric if not already
wine_data['Year'] = pd.to_numeric(wine_data['Year'], errors='coerce')


# 2. Rename ABV to ABV (%) if the model expects it
wine_data.rename(columns={'ABV': 'ABV (%)'}, inplace=True)

# 3. Recreate Region column from Country and Region as the model expects 'Region_*' columns
def determine_region(row):
    if row['Country'] == 'USA':
        return row['Region']  # For US wines, region might be the state
    else:
        return row['Country']  # For non-US wines, region might be the country

wine_data['Region'] = wine_data.apply(determine_region, axis=1)

# 5. Rename 'Type' to 'Red or White'
wine_data.rename(columns={'Type': 'Red or White'}, inplace=True)

# 4. Rename 'Grape' to 'Type'
wine_data.rename(columns={'Grape': 'Type'}, inplace=True)

# %%
# 6. Now create dummy variables for 'Type', 'Red or White', and 'Region'
wine_data_dummies = pd.get_dummies(wine_data, columns=['Type', 'Red or White', 'Region'], drop_first=False)

# 7. Align wine_data_dummies with X.
missing_cols = set(X.columns) - set(wine_data_dummies.columns)
extra_cols = set(wine_data_dummies.columns) - set(X.columns)

for col in missing_cols:
    wine_data_dummies[col] = 0.0

if extra_cols:
    wine_data_dummies.drop(list(extra_cols), axis=1, inplace=True)

wine_data_dummies = wine_data_dummies[X.columns]

# If the model had a const and it's not present, add it
if 'const' in regression_results[0]['Coefficients'].index and 'const' not in wine_data_dummies.columns:
    wine_data_dummies['const'] = 1.0
    # If const is expected to be the first column, reorder if needed
    if 'const' in X.columns and X.columns[0] == 'const':
        cols = ['const'] + [c for c in X.columns if c != 'const']
        wine_data_dummies = wine_data_dummies[cols]

# 8. Use the stored coefficients from regression_results to predict ratings
for result in regression_results:
    rater = result['Rater']
    coefs = result['Coefficients']

    # Get the intersection of columns
    aligned_cols = coefs.index.intersection(wine_data_dummies.columns)
    # Sort them to ensure consistent order
    aligned_cols = aligned_cols.sort_values()

    # Reindex both coefs and data
    aligned_coefs = coefs.reindex(aligned_cols)
    aligned_data = wine_data_dummies[aligned_cols]

    # Verify no missing values remain
    aligned_coefs = aligned_coefs.fillna(0.0)

    # Convert to numpy arrays
    X_values = aligned_data.values      # shape: (n_samples, n_features)
    coef_values = aligned_coefs.values  # shape: (n_features,)

    # Ensure shapes align
    if X_values.shape[1] != coef_values.shape[0]:
        raise ValueError(f"Mismatch in shapes after alignment: X_values has {X_values.shape[1]} columns, but coef_values has {coef_values.shape[0]} entries.")

    # Compute predictions using dot product
    predictions = X_values @ coef_values  # shape: (n_samples,)

    if rater == 'Average':
        wine_data['Predicted_Average'] = predictions
    else:
        wine_data[f'Predicted_{rater}'] = predictions

# 10. Display results
print(wine_data[['Red or White', 'Type', 'Region', 'Country', 'Vintage', 'ABV (%)', 'Predicted_Average'] 
                 + [f'Predicted_{rater}' for rater in valid_raters_list if f'Predicted_{rater}' in wine_data.columns]])

# %%
