# Trade-sentiment-analysis
#Mount Google Drive & Create Folder Structure

from google.colab import drive
import os

drive.mount('/content/drive')

project_path = '/content/drive/MyDrive/location'
os.makedirs(os.path.join(project_path, 'csv_files'), exist_ok=True)
os.makedirs(os.path.join(project_path, 'outputs'), exist_ok=True)

print(" Folder structure ready at:", project_path)


# Install Required Libraries

!pip install gdown reportlab seaborn --quiet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

sns.set(style="whitegrid")


#  Download Datasets

trader_data_url = "https://drive.google.com/file location"
trader_csv_path = os.path.join(project_path, "csv_files/historical_trader_data.csv")
gdown.download(trader_data_url, trader_csv_path, quiet=False)

sentiment_data_url = "https://drive.google.com/file location"
sentiment_csv_path = os.path.join(project_path, "csv_files/fear_greed_index.csv")
gdown.download(sentiment_data_url, sentiment_csv_path, quiet=False)


#  Load Data

df_trades = pd.read_csv(trader_csv_path)
df_sentiment = pd.read_csv(sentiment_csv_path)

print("Trader Data:", df_trades.shape)
print("Sentiment Data:", df_sentiment.shape)


#  Clean & Parse Timestamp IST

# Parse IST timestamps in DD-MM-YYYY HH:MM format
df_trades['full_datetime'] = pd.to_datetime(
    df_trades['Timestamp IST'],
    format='%d-%m-%Y %H:%M',
    errors='coerce'
)
df_trades['trade_date'] = df_trades['full_datetime'].dt.date

# Convert numeric columns
numeric_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Start Position', 'Closed PnL', 'Fee']
for col in numeric_cols:
    if col in df_trades.columns:
        df_trades[col] = pd.to_numeric(df_trades[col], errors='coerce')


#  Clean Sentiment Data

df_sentiment['date'] = pd.to_datetime(df_sentiment['date'], errors='coerce')
df_sentiment['classification'] = df_sentiment['classification'].str.strip().str.title()
df_sentiment['date_only'] = df_sentiment['date'].dt.date


#  Merge with Fixed Date Types

df_trades['trade_date'] = pd.to_datetime(df_trades['trade_date'], errors='coerce').dt.date
df_sentiment['date_only'] = pd.to_datetime(df_sentiment['date_only'], errors='coerce').dt.date

df_merged = pd.merge(
    df_trades,
    df_sentiment[['date_only', 'classification']],
    left_on='trade_date',
    right_on='date_only',
    how='left'
)

matches = df_merged['classification'].notna().sum()
print(f" Matches found: {matches} / {len(df_merged)}")


#  EDA Plots

outputs_path = os.path.join(project_path, "outputs")

# Average PnL
avg_pnl = df_merged.groupby('classification')['Closed PnL'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(data=avg_pnl, x='classification', y='Closed PnL', palette='viridis')
plt.title("Average Closed PnL by Market Sentiment")
pnl_path = os.path.join(outputs_path, "avg_pnl_by_sentiment.png")
plt.savefig(pnl_path, bbox_inches='tight')
plt.close()

# Total Volume
total_vol = df_merged.groupby('classification')['Size USD'].sum().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(data=total_vol, x='classification', y='Size USD', palette='magma')
plt.title("Total Trade Volume by Market Sentiment")
vol_path = os.path.join(outputs_path, "total_volume_by_sentiment.png")
plt.savefig(vol_path, bbox_inches='tight')
plt.close()

# Average Position Size
avg_size = df_merged.groupby('classification')['Size Tokens'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(data=avg_size, x='classification', y='Size Tokens', palette='coolwarm')
plt.title("Average Position Size by Market Sentiment")
size_path = os.path.join(outputs_path, "avg_position_size_by_sentiment.png")
plt.savefig(size_path, bbox_inches='tight')
plt.close()

print(" Charts saved to:", outputs_path)


#  Generate PDF Report

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

pdf_path = os.path.join(project_path, "ds_report.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Data Science Assignment – Web3 Trading Team", styles['Title']))
story.append(Spacer(1, 0.2*inch))

intro_text = """
This report analyzes trading behavior in relation to Bitcoin market sentiment (Fear vs. Greed).
We examine profitability, trade volume, and position sizes under different market sentiments.
"""
story.append(Paragraph(intro_text, styles['Normal']))
story.append(Spacer(1, 0.3*inch))

charts_info = [
    ("Average Closed PnL by Market Sentiment", pnl_path),
    ("Total Trade Volume by Market Sentiment", vol_path),
    ("Average Position Size by Market Sentiment", size_path)
]

for title, img_path in charts_info:
    story.append(Paragraph(title, styles['Heading2']))
    try:
        story.append(Image(img_path, width=5.5*inch, height=3*inch))
    except:
        story.append(Paragraph(f"Image not found: {img_path}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

findings_text = f"""
Key Findings:
- Matches between trade records and sentiment data: {matches} out of {len(df_merged)}
- Average profitability, volume, and size vary with sentiment phases.
- Further analysis can explore leverage and trader segmentation.
"""
story.append(Paragraph(findings_text, styles['Normal']))

doc.build(story)
print(f" PDF report generated at: {pdf_path}")
