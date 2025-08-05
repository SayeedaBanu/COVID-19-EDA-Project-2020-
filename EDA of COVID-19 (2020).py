# COVID-19 Exploratory Data Analysis for 2020
# This script performs comprehensive EDA on COVID-19 data from 2020

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("COVID-19 2020 Exploratory Data Analysis")
print("="*50)

# 1. DATA LOADING AND PREPARATION
print("\n1. LOADING AND PREPARATION DATA")
print("-"* 30)

# Since we don't have access to the actual dataset, Lets create a realistic dataset
# Based on actually COVID-19 pattern from 2020
np.random.seeed(42)

# Create date range for 2020
dates = pd.date_range(start='2020-01-22', end='2020-12-31', freq='D')
countries = ['US', 'Italy', 'Spain', 'Germany', 'France', 'UK', 'China', 'Iran', 'South Korea', 'Brazil', 'India', 'Russia', 'Turkey', 'Belgium', 'Netherlands']

# Generate realistic COVID-19 Data
data = []
for country in countries:
    # Different countries had different outbreak patterns
    if country == 'China':
        # Early outbreak, then controlled
        peak_day = 30
        max_cases = 80000
    elif country in ['Italy', 'Spain']:
        # European first wave
        peak_day = 90
        max_cases = 200000    
    elif country == 'US':
        # Multiple waves
        peak_day = 120
        max_cases = 20000000
    elif country in ['Brazil', 'India']:
        # Late outbreak, high cases
        peak_day = 200
        max_cases = 5000000
    else:
        # General pattern 
        peak_day = 100
        max_case = 500000

        for i, data in enumerate(dates):
            # Create realistic exponential growth then decline pattern 
            if i < peak_day:
                # Growth phase
                base_case = max_cases * (1 - np.exp(-i/30)) * np.random.uniform(0.8, 1.2)
            else:
                # Decline phase
                base_case = max_cases * np.exp(-(i-peak_day)/60) * np.random.uniform(0.9, 1.1)

                # Add noise and ensure non-negative cases
                confirmed = max(0, int(base_case + np.random.normal(0, base_case * 0.1)))
                deaths = max(0, int(confirmed * np.random.uniform(0.01, 0.05)))
                recovered = max(0, int(confirmed * np.random.uniform(0.7, 0.95)))

                data.append({
                    'Date': data,
                    'Country': country,
                    'Confirmed': confirmed,
                    'Deaths': deaths,
                    'Recovered': recovered,
                    'Active': max(0, confirmed - deaths - recovered)
                })

 # Create DataFrame
df = pd.DataFrame(data)   
df['Date']  = pd.to_datetime(df['Date'])

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Countries: {len(df['Country'].unique())}")

# 2.DATA OVERVIEW
print("\n2. DATA OVERVIEW")
print("-"* 20)

print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. GLOBAL TRENDS ANALYSIS
print("\n3. GLOBAL TRENDS ANALYSIS")
print("-"* 25)

# Calculate global daily totals
global_daily = df.groupby('Date').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'Active': 'sum'
}).reset_index()

# Calculate new cases daily
global_daily['New_Cases'] = global_daily['Confirmaed'].diff()
global_daily['New_Deaths'] = global_daily['Deaths'].diff()

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Global COVID-19 Trends in 2020', fontsize=16, fontweight='bold')

# Cumulative Cases
axes[0, 0].plot(global_daily['Date'], global_daily['Confirmed'], 'b-', linewidth=2, label='Confirmed')
axes[0, 0].plot(global_daily['Date'], global_daily['Deaths'], 'r-', linewidth=2, label='Deaths')
axes[0, 0].plot(global_daily['Date'], global_daily['Recovered'], 'g-', linewidth=2, label='Recovered')
axes[0, 0].set_title('Cumulative Cases Worldwide')
axes[0,0].set_ylabel('Number of Cases')
axes[0, 0].set_xlabel('Date')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Daily New Cases
axes[0,1].plot(global_daily['Date'], global_daily['New_Cases'], 'orange', linewidth=2)
axes[0, 1].set_title('Daily New Cases Worldwide')
axes[0, 1].set_ylabel('New Cases')
axes[0, 1].set_xlabel('Date')
axes[0, 1].grid(True, alpha=0.3)

# Case Fatality Rate
cfr = (global_daily['Deaths'] / global_daily['Confirmed'] * 100).fillna(0)
axes[1, 0].plot(global_daily['Date'], cfr, 'red', linewidth=2)
axes[1, 0].set_title('Case Fatality Rate (%)')
axes[1, 0].set_ylabel('CFR (%)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].grid(True, alpha=0.3)

# Active Cases
axes[1, 1].plot(global_daily['Date'], global_daily['Active'], 'purple', linewidth=2)
axes[1, 1].set_title('Active Cases Worldwide')
axes[1, 1].set_ylabel('Active Cases')
axes[1, 1].set_xlabel('Date')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print key statistics
print("\nKey Global Statistics for 2020:")
print(f"Total Confirmed Cases: {global_daily['Confirmed'].iloc[-1]:,}")
print(f"Total Deaths: {global_daily['Deaths'].iloc[-1]:,}")
print(f"Total Recovered: {global_daily['Recovered'].iloc[-1]:,}")
print(f"Peak Daily New Cases: {global_daily['New_Cases'].max():,.0f} ")

# 4. COUNTRY-WISE ANALYSIS
print("\n4. COUNTRY-WISE ANALYSIS")
print("-"* 25)

# Get Final numbers for each country
country_totals = df[df['Date'] == df['Date'].max()].copy()
country_totals = country_totals.sort_values('Confirmed', ascending=False)

print("\nTop 10 Countries by Total Confirmed Cases (End of 2020):")
print(country_totals[['Country', 'Confirmed', 'Deaths', 'Recovered']].head(10))

# Visualize top countries
fig, axes = plt.subplot(1, 2, figsize=(15, 12))
fig.suptitle('COVID-19 Cases by Country at (End of 2020)', fontsize=16, fontweight='bold')

# Top 10 Countries by Confirmed Cases
top_10 = country_totals.head(10)
axes[0,0].barh(top_10['Country'], top_10['Confirmed'])
axes[0, 0].set_title('Top 10 Countries by Confirmed Cases')
axes[0, 0].set_xlabel('Total Confirmed Cases') 

# Top 10 Countries by Deaths
top_10_deaths = country_totals.sort_values('Deaths', ascending=False).head(10)
axes[0, 1].barh(top_10_deaths['Country'], top_10_deaths['Deaths'])
axes[0, 1].set_title('Top 10 Countries - Deaths')
axes[0, 1].set_xlabel('Total Deaths')

# Case Fatality rate by country 
country_totals['CFR'] = (country_totals['Deaths'] / country_totals['Confirmed'] * 100).fillna(0)
top_cfr = country_totals.sort_values('CFR', ascending=False).head(10)
axes[1, 0].barh(top_cfr['Country'], top_cfr['CFR'])
axes[1, 0].set_title('Top 10 Countries - Case Fatality Rate (%)')
axes[1, 0].set_xlabel('CFR (%)')

# Reccovery Rate by Country
country_totals['Recovery_Rate'] = (country_totals['Recovered'] / country_totals['Confirmed'] * 100).fillna(0)
top_recovery = country_totals.sort_values('Recovery_Rate', ascending=False).head(10)
axes[1, 1].barh(top_recovery['Country'], top_recovery['Recovery_Rate'])
axes[1, 1].set_title('Top 10 Countries - Recovery Rate (%)')
axes[1, 1].set_xlabel('Recovery Rate (%)')

plt.tight_layout()
plt.show()

# 5. TEMPORAL PATTERNS
print("\n5. TEMPORAL PATTERNS")
print("-"* 30)

# Monthly Trends
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()

monthly_cases = df.groupby(['Month', 'Month_Name'])['Confirmed'].sum().reset_index()
monthly_cases = monthly_cases.sort_values('Month')

plt.figure(figsize=(12,6))
plt.plot(monthly_cases['Month_Name'], monthly_cases['Confirmed'], 'bo-', linewidth=2, markersize=8)
plt.title('Monthly COVID-19 Cases in 2020', fonysize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Confirmed Cases')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Weekly patterns
df['Weekday'] = df['Date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_avg = df.groupby('Weekday')['Confirmed'].mean().reindex(weekday_order)

plt.figure(figsize=(10, 6))
plt.bar(weekly_avg.index, weekly_avg.values, color='skyblue', alpha=0.7)
plt.title('Average COVID-19 Cases by Day of Week', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week')
plt.ylabel('Average Confirmed Cases')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. CORRELATIONS ANALYSIS
print("\n6. CORRELATION ANALYSIS")
print("-" *25)

# Calculate correlations
correlation_data = country_totals[['Confirmed', 'Deaths', 'Recovered', 'Active', 'CFR', 'Recovery_Rate']]
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, camp='coolwarm', center=0, square=True, linewidth=0.5)
plt.title('Correlation Matrix - COVID-19 Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# 7. GROWTH RATE ANALYSIS
print("\n7. GROWTH RATE ANALYSIS")
print("-" * 25)

#Calculate growth rates for selected Countries
selected_countries = ['US', 'Italy', 'Spain', 'Germany', 'China']
growth_analysis = []

for country in selected_countries:
    country_data = df[df['Country'] == country].sort_values('Date')
    country_data['Growth_Rate'] = country_data['Confirmed'].pct_change() * 100
    
    # Calculate average growth rate (excluding infinite values)
    avg_growth = country_data['Growth_Rate'].replace([np.inf, -np.inf], np.nan).mean()
    growth_analysis.append({'Country': country, 'Ang_Growth_Rate': avg_growth})

growth_df = pd.DataFrame(growth_analysis)
print("\nAverage Daily Growth Rates By Country:")
print(growth_df.round(2))

#Visualize growth rates over time
plt.figure(figsize=(12,8))
for country in selected_countries:
    country_data = df[df['Country'] == country].sort_values('Date')
    country_data['Growth_Rate'] = country_data['Confirmed'].pct_change() * 100
    # Smooth the data with rolling average
    smoothed = country_data['Growth_Rate'].rolling(windows=7).mean()
    plt.plot(country_data['Date'], smoothed, label=country, linewidth=2)

plt.title('COVID-19 Growth Rates Over Time (7-days Moving Average)', fontsize=14, fontweight='bold') 
plt.xlabel('Date')
plt.ylabel('Growth Rate (%)')
plt.legend()
plt.grid(True, alpha=0.3) 
plt.tight_layout()
plt.show()

# 8. SUMMARY AND KEY INSIGHTS
print("\n8. KEY INSIGHTS AND SUMMARY")
print("-" * 35)

print("\n📊 KEY FINDINGS:")
print("=" * 50)

print(f"\n🌍 GLOBAL IMPACT:")
print(f"• Total confirmed cases in 2020: {global_daily['Confirmed'].iloc[-1]:,}")
print(f"• Total deaths: {global_daily['Deaths'].iloc[-1]:,}")
print(f"• Overall case fatality rate: {cfr.iloc[-1]:.2f}%")
print(f"• Peak daily new cases: {global_daily['New_Cases'].max():,.0f}")

print(f"\n🏆 MOST AFFECTED COUNTRIES:")
top_3 = country_totals.head(3)
for i, (_, country) in enumerate(top_3.iterrows(), 1):
    print(f"• #{i}: {country['Country']} - {country['Confirmed']:,} cases")

print(f"\n📈 TEMPORAL PATTERNS:")
peak_month = monthly_cases.loc[monthly_cases['Confirmed'].idxmax(), 'Month_Name']
print(f"• Peak month: {peak_month}")
print(f"• Most active day of week: {weekly_avg.idxmax()}")

print(f"\n🔄 RECOVERY INSIGHTS:")
avg_recovery_rate = country_totals['Recovery_Rate'].mean()
print(f"• Average global recovery rate: {avg_recovery_rate:.1f}%")
best_recovery = country_totals.loc[country_totals['Recovery_Rate'].idxmax(), 'Country']
print(f"• Best recovery rate: {best_recovery}")

print(f"\n📉 GROWTH PATTERNS:")
fastest_growing = growth_df.loc[growth_df['Avg_Growth_Rate'].idxmax(), 'Country']
print(f"• Fastest average growth: {fastest_growing}")
print(f"• Growth rates varied significantly across countries and time periods")

print(f"\n🎯 METHODOLOGY NOTES:")
print("• Analysis based on synthetic data following realistic COVID-19 patterns")
print("• Data covers full year 2020 (Jan 22 - Dec 31)")
print("• Includes 15 major countries affected by COVID-19")
print("• Metrics: confirmed cases, deaths, recovered, active cases, CFR, growth rates")

print("\n" + "="*50)
print("Analysis Complete! 📊✅")