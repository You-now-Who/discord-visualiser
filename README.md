# Discord Messages Visualizer 💬

A comprehensive Streamlit web application for visualizing and analyzing your Discord message data with beautiful interactive charts and statistics.

## Features

### 🌍 Timezone Support
- **Timezone Selection**: Choose from common timezones or browse all available timezones
- **Auto-detection**: Attempt to automatically detect your local timezone
- **Real-time Conversion**: All timestamps are converted to your selected timezone
- **Current Time Display**: See the current time in your selected timezone

### 📅 Temporal Analysis
- **Messages Over Time**: Line charts showing daily message counts with rolling averages
- **Hourly Activity**: Heatmaps showing when you're most active throughout the week
- **Weekly Patterns**: Bar charts of activity by day of week
- **Monthly Trends**: Seasonal patterns in your messaging behavior

### 📝 Message Content Analysis
- **Length Analysis**: Distribution of message lengths and word counts
- **Content Patterns**: Emoji usage, ALL CAPS messages, questions, and attachments
- **Word Cloud**: Visual representation of your most used words
- **Top Words**: Bar chart of your most frequent words

### 🔍 Advanced Analytics
- **Streak Analysis**: Consecutive days with messages
- **Peak Activity**: Busiest days and peak hours
- **Correlation Analysis**: Relationships between message features

### 📊 Interactive Features
- **Timezone Selection**: Convert all timestamps to your local timezone
- Date range filtering
- Message length filtering
- Downloadable filtered data
- Responsive design with beautiful styling
- Real-time statistics and metrics

## Setup & Installation

### Prerequisites
- Python 3.7 or higher
- Your Discord messages data in JSON format

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   
   **Option A: Command Line**
   ```bash
   streamlit run main.py
   ```
   
   **Option B: Python Runner**
   ```bash
   python run_app.py
   ```
   
   **Option C: PowerShell (Windows)**
   ```powershell
   .\run_app.ps1
   ```

3. **Open your browser** to the URL shown in the terminal (usually http://localhost:8501)

## Data Format

Your `data/messages.json` file should contain an array of message objects with the following structure:

```json
[
  {
    "ID": 1401248510522822748,
    "Timestamp": "2025-08-02 17:01:18",
    "Contents": "your message content here",
    "Attachments": "attachment_url_or_empty_string"
  }
]
```

## Usage Tips

1. **Timezone Setup**: First select your timezone from the sidebar to see all times in your local time
2. **Filtering**: Use the sidebar filters to focus on specific time periods or message lengths
3. **Interactive Charts**: Hover over data points for detailed information
4. **Export Data**: Download your filtered data as CSV from the Raw Data tab
5. **Performance**: For large datasets (>50k messages), consider filtering by date range for better performance

## Timezone Support

The app supports comprehensive timezone handling:

- **Common Timezones**: Quick selection from frequently used timezones
- **All Timezones**: Browse through all available IANA timezone identifiers
- **Auto-detection**: Attempts to detect your system timezone (may require manual verification)
- **Real-time Display**: Shows current time in your selected timezone
- **Data Conversion**: All message timestamps are automatically converted to your chosen timezone

### Supported Timezone Examples:
- `US/Eastern`, `US/Pacific`, `US/Central`, `US/Mountain`
- `Europe/London`, `Europe/Paris`, `Europe/Berlin`
- `Asia/Tokyo`, `Asia/Shanghai`, `Asia/Kolkata`
- `Australia/Sydney`, `Australia/Melbourne`
- And many more...

## Visualizations Included

- 📈 Line charts for trends over time
- 🔥 Heatmaps for activity patterns
- 📊 Bar charts for categorical data
- 📏 Histograms for distributions
- ☁️ Word clouds for text analysis
- 🔗 Correlation matrices for relationships
- 📦 Box plots for statistical summaries

## Metrics Calculated

- Total message count
- Active days
- Average messages per day
- Average message length
- Longest messaging streak
- Peak activity periods
- Emoji usage statistics
- Question frequency
- Attachment frequency

## Troubleshooting

1. **Import Errors**: Make sure all packages are installed with `pip install -r requirements.txt`
2. **Data Not Loading**: Check that your `data/messages.json` file exists and is properly formatted
3. **Performance Issues**: Try filtering to a smaller date range
4. **Browser Issues**: Try refreshing the page or clearing browser cache

## Technical Details

- Built with Streamlit for the web interface
- Uses Plotly for interactive visualizations
- Pandas for data manipulation
- WordCloud for text visualization
- Matplotlib and Seaborn for additional charts

Enjoy exploring your Discord messaging patterns! 🚀
