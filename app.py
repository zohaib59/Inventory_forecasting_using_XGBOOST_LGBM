#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from prophet import Prophet

os.chdir('C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk')

# Read the dataset
df = pd.read_csv('inventory.csv')

# Parse and sort by date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date')

df = df.dropna(subset=['Date'])

target = 'Units Sold'


# === DROP UNNEEDED COLUMNS ===
df.drop(columns=['Store ID', 'Product ID', 'Region', 'Category'], inplace=True, errors='ignore')


# === FEATURE ENGINEERING === 
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Sales_Lag_1'] = df[target].shift(1)
df['Sales_Lag_7'] = df[target].shift(7)
df['Rolling_7'] = df[target].rolling(window=7, min_periods=1).mean()
df['Rolling_14'] = df[target].rolling(window=14, min_periods=1).mean()
df.dropna(inplace=True)


# === DEFINE FEATURES & TARGET === 
features = ['Sales_Lag_1', 'Sales_Lag_7', 'Rolling_7', 'Rolling_14',
            'Price', 'Discount', 'DayOfWeek', 'Month']

X = df[features]
y = df[target]


# === SPLIT DATA === #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === LIGHTGBM MODEL === 
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)

# === XGBOOST MODEL === #
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# === EVALUATION FUNCTION === 
def evaluate(model_name, y_true, y_pred):
    print(f"\n📊 {model_name} Performance:")
    print("MAE :", round(mean_absolute_error(y_true, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("MAPE:", round(mean_absolute_percentage_error(y_true, y_pred)*100, 2), "%")
    print("R²   :", round(r2_score(y_true, y_pred), 4))

# === MODEL EVALUATIONS === 
evaluate("LightGBM", y_test, y_pred_lgbm)
evaluate("XGBoost", y_test, y_pred_xgb)


# === PLOTS === #
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', color='black')
plt.plot(y_pred_lgbm[:100], label='LGBM', linestyle='--')
plt.plot(y_pred_xgb[:100], label='XGBoost', linestyle=':')
plt.title("Actual vs Predicted - First 100 Samples")
plt.legend()
plt.show()


# === FEATURE IMPORTANCE (LGBM) ===
importances = pd.Series(lgbm.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("LGBM Feature Importance")
plt.show()

# === PROPHET FORECAST === 
df_prophet = df[['Date', target]].rename(columns={'Date': 'ds', target: 'y'})
prophet = Prophet()
prophet.fit(df_prophet)

# Forecast next 30 days
future = prophet.make_future_dataframe(periods=30)
forecast = prophet.predict(future)


# === PLOT PROPHET FORECAST ===
fig1 = prophet.plot(forecast)
plt.title("Prophet Forecast - Next 30 Days")
plt.show()

# === LAST 30-DAY FORECAST === #
print("\n📅 Prophet - Last 30 Days Forecast:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).round(2))


#Universal code for data insights
# === IMPORTS === #
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === SET WORKING DIRECTORY & LOAD DATA === #
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
file_name = "inventory.csv"
df = pd.read_csv(file_name)
df.columns = df.columns.str.strip()

# === DATE CONVERSION (DD-MM-YYYY Format Handling) === #
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"\n⚠️ {invalid_dates} rows have invalid or unparseable date values.")

# === BASIC STRUCTURED SUMMARY === #
print("\n📋 BASIC DATA SUMMARY")
print("-" * 60)
print("📦 Dataset Shape:", df.shape)
print("\n🧾 Column Types:\n", df.dtypes)
print("\n🧼 Missing Values:\n", df.isnull().sum())
print("\n📊 Descriptive Stats:\n", df.describe(include='all').T)

# === CORRELATION HEATMAP === #
num_cols = df.select_dtypes(include=np.number).columns
correlation_matrix = df[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔗 Feature Correlation Matrix")
plt.show()

# === TIME SERIES TREND (Units Sold Over Time) === #
if 'Date' in df.columns and 'Units Sold' in df.columns:
    print("\n📈 TIME SERIES: Daily Units Sold")
    daily_sales = df.groupby('Date')['Units Sold'].sum()
    daily_sales.plot(figsize=(12, 5), title="📈 Daily Units Sold Over Time", grid=True)
    plt.ylabel("Units Sold")
    plt.show()

# === SEASONALITY: Month vs Units Sold + Summary === #
if 'Date' in df.columns:
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.day_name()
    month_summary = df.groupby('Month')['Units Sold'].sum()
    print("\n📅 MONTHLY SALES SUMMARY (Units Sold):\n", month_summary)
    sns.boxplot(x='Month', y='Units Sold', data=df)
    plt.title("📅 Monthly Sales Seasonality")
    plt.show()

# === WEATHER IMPACT: Visual + Data === #
if {'Weather Condition', 'Units Sold'}.issubset(df.columns):
    print("\n🌦️ WEATHER IMPACT ON SALES (Units Sold per Weather Condition):")
    weather_summary = df.groupby('Weather Condition')['Units Sold'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
    print(weather_summary)
    sns.boxplot(x='Weather Condition', y='Units Sold', data=df)
    plt.title("🌦️ Weather vs Units Sold")
    plt.xticks(rotation=45)
    plt.show()

# === LOGISTICS DELAY IMPACT === #
if 'Logistics Delay' in df.columns and 'Units Sold' in df.columns:
    print("\n🚚 LOGISTICS DELAY IMPACT:")
    sns.boxplot(x='Logistics Delay', y='Units Sold', data=df)
    plt.title("🚚 Logistics Delay vs Units Sold")
    plt.show()

# === REGIONAL SALES PERFORMANCE === #
if 'Region' in df.columns and 'Units Sold' in df.columns:
    region_sales = df.groupby('Region')['Units Sold'].sum().sort_values(ascending=False)
    print("\n📍 REGIONAL SALES PERFORMANCE:\n", region_sales)
    region_sales.plot(kind='bar', title="📍 Regional Sales Performance", figsize=(10, 4))
    plt.ylabel("Total Units Sold")
    plt.xticks(rotation=45)
    plt.show()

# === STORE-WISE REVENUE === #
if {'Store', 'Price', 'Units Sold'}.issubset(df.columns):
    df['Revenue'] = df['Price'] * df['Units Sold']
    store_revenue = df.groupby('Store')['Revenue'].sum().sort_values()
    print("\n🏪 STORE REVENUE SUMMARY")
    print("\nTop 5 Stores:\n", store_revenue.tail(5))
    print("\nBottom 5 Stores:\n", store_revenue.head(5))
    store_revenue.plot(kind='bar', title="💰 Revenue by Store", figsize=(12, 5))
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.show()

# === TOP ARTICLE BY UNITS SOLD === #
if 'Article' in df.columns and 'Units Sold' in df.columns:
    top_article = df.groupby('Article')['Units Sold'].sum().sort_values(ascending=False).head(1)
    print(f"\n🏆 Top Selling Article: {top_article.index[0]} with {top_article.values[0]} units sold")

# === TOP CATEGORY: Total Units, Orders === #
if 'Category' in df.columns and 'Units Sold' in df.columns:
    category_summary = df.groupby('Category').agg(
        Total_Units_Sold=('Units Sold', 'sum'),
        Total_Orders=('Units Sold', 'count')
    ).sort_values('Total_Units_Sold', ascending=False)
    print("\n📦 CATEGORY-WISE SUMMARY:\n", category_summary)
    top_cat = category_summary.head(1)
    print(f"\n🏆 Top Category: {top_cat.index[0]} | Units Sold: {top_cat['Total_Units_Sold'].values[0]} | Orders: {top_cat['Total_Orders'].values[0]}")

# === YEARLY REVENUE SUMMARY === #
if 'Date' in df.columns and 'Revenue' in df.columns:
    df['Year'] = df['Date'].dt.year
    revenue_yearly = df.groupby('Year')['Revenue'].sum()
    print("\n📅 TOTAL REVENUE GENERATED YEAR-WISE:\n", revenue_yearly)
    revenue_yearly.plot(kind='bar', title="💸 Revenue by Year", figsize=(10, 5))
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45)
    plt.show()

# === KEY INSIGHTS === #
insights = []
print("\n✅ KEY INSIGHTS")
print("-" * 60)
if 'Inventory Level' in df.columns:
    stockouts = (df['Inventory Level'] == 0).mean()
    insights.append(f"🟥 Stockout Rate: {stockouts*100:.2f}%")
if {'Units Sold', 'Inventory Level'}.issubset(df.columns):
    overstock = (df['Inventory Level'] > 2 * df['Units Sold'].mean()).mean()
    insights.append(f"🟦 Overstock Rate (>2x avg sales): {overstock*100:.2f}%")
if {'Discount', 'Units Sold'}.issubset(df.columns):
    high_disc = df[df['Discount'] > df['Discount'].median()]
    low_disc = df[df['Discount'] <= df['Discount'].median()]
    diff = high_disc['Units Sold'].mean() - low_disc['Units Sold'].mean()
    insights.append(f"💸 Discounts lead to ~{diff:.2f} more units sold")
if {'Price', 'Units Sold'}.issubset(df.columns):
    corr_price_sales = df['Price'].corr(df['Units Sold'])
    insights.append(f"💰 Price correlation with sales: {corr_price_sales:.2f}")
if {'Holiday/Promotion', 'Units Sold'}.issubset(df.columns):
    promo = df[df['Holiday/Promotion'] == 1]
    non_promo = df[df['Holiday/Promotion'] == 0]
    promo_diff = promo['Units Sold'].mean() - non_promo['Units Sold'].mean()
    insights.append(f"🎯 Promotions boost sales by ~{promo_diff:.2f} units")

for line in insights:
    print(line)

# === RECOMMENDATIONS === #
print("\n📌 RECOMMENDATIONS")
print("-" * 60)
if 'stockouts' in locals() and stockouts > 0.05:
    print("🔁 Reduce stockouts with improved reorder logic or safety stock.")
if 'overstock' in locals() and overstock > 0.10:
    print("📦 Overstocking detected — reduce excess inventory.")
if 'corr_price_sales' in locals() and abs(corr_price_sales) > 0.3:
    print("📉 Price elasticity detected — consider price optimization.")
if 'Weather Condition' in df.columns:
    print("🌧️ Weather is influencing sales — integrate with forecasting.")
if 'Competitor Pricing' in df.columns:
    print("🏷️ Add competitor pricing to enhance pricing strategy.")
