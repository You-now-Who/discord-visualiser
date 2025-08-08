import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import re
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

# Set page config
st.set_page_config(
    page_title="Discord Messages Visualizer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(timezone_str='UTC'):
    """Load and preprocess the Discord messages data"""
    try:
        with open(f'data/{user}/messages.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime and handle timezone
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # If timezone is not UTC, convert from UTC to the specified timezone
        if timezone_str != 'UTC':
            try:
                target_tz = pytz.timezone(timezone_str)
                # Assume input timestamps are in UTC and convert to target timezone
                df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC').dt.tz_convert(target_tz)
            except pytz.exceptions.UnknownTimeZoneError:
                st.warning(f"Unknown timezone '{timezone_str}', using UTC instead.")
                timezone_str = 'UTC'
        
        df['Date'] = df['Timestamp'].dt.date
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['Month'] = df['Timestamp'].dt.month_name()
        df['Year'] = df['Timestamp'].dt.year
        
        # Calculate message length
        df['MessageLength'] = df['Contents'].str.len()
        
        # Count words
        df['WordCount'] = df['Contents'].str.split().str.len()
        
        # Check for attachments
        df['HasAttachment'] = df['Attachments'].str.len() > 0
        
        # Extract emojis and special characters
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002500-\U00002BEF"  # chinese char
                                 u"\U00002702-\U000027B0"
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 u"\U0001f926-\U0001f937"
                                 u"\U00010000-\U0010ffff"
                                 u"\u2640-\u2642"
                                 u"\u2600-\u2B55"
                                 u"\u200d"
                                 u"\u23cf"
                                 u"\u23e9"
                                 u"\u231a"
                                 u"\ufe0f"
                                 "]+", flags=re.UNICODE)
        
        df['EmojiCount'] = df['Contents'].apply(lambda x: len(emoji_pattern.findall(x)))
        
        # Check for all caps messages
        df['IsAllCaps'] = df['Contents'].apply(lambda x: x.isupper() and len(x) > 2)
        
        # Check for questions
        df['IsQuestion'] = df['Contents'].str.contains(r'\?', na=False)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_common_timezones():
    """Get a list of common timezones for the selectbox"""
    common_timezones = [
        'UTC',
        'US/Eastern',
        'US/Central', 
        'US/Mountain',
        'US/Pacific',
        'Europe/London',
        'Europe/Paris',
        'Europe/Berlin',
        'Europe/Rome',
        'Europe/Madrid',
        'Europe/Moscow',
        'Asia/Tokyo',
        'Asia/Shanghai',
        'Asia/Kolkata',
        'Asia/Dubai',
        'Asia/Seoul',
        'Asia/Bangkok',
        'Asia/Singapore',
        'Australia/Sydney',
        'Australia/Melbourne',
        'Australia/Perth',
        'America/New_York',
        'America/Chicago',
        'America/Denver',
        'America/Los_Angeles',
        'America/Toronto',
        'America/Vancouver',
        'America/Mexico_City',
        'America/Sao_Paulo',
        'America/Argentina/Buenos_Aires',
        'Africa/Cairo',
        'Africa/Johannesburg',
        'Pacific/Auckland',
        'Pacific/Honolulu'
    ]
    
    # Add all available timezones for advanced users
    all_timezones = sorted(pytz.all_timezones)
    
    return common_timezones, all_timezones

def create_overview_metrics(df):
    """Create overview metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Messages", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        date_range = (df['Timestamp'].max() - df['Timestamp'].min()).days
        st.metric("Days Active", f"{date_range:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_per_day = len(df) / max(date_range, 1)
        st.metric("Avg Messages/Day", f"{avg_per_day:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_length = df['MessageLength'].mean()
        st.metric("Avg Message Length", f"{avg_length:.1f} chars")
        st.markdown('</div>', unsafe_allow_html=True)

def create_temporal_analysis(df):
    """Create temporal analysis visualizations"""
    st.header("📅 Temporal Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Messages Over Time", "Hourly Activity", "Weekly Patterns", "Monthly Trends"])
    
    with tab1:
        # Daily message count
        daily_counts = df.groupby('Date').size().reset_index(name='MessageCount')
        daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
        
        fig = px.line(daily_counts, x='Date', y='MessageCount', 
                     title='Messages Per Day Over Time',
                     labels={'MessageCount': 'Number of Messages'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling average
        daily_counts['RollingAvg'] = daily_counts['MessageCount'].rolling(window=7).mean()
        fig2 = px.line(daily_counts, x='Date', y=['MessageCount', 'RollingAvg'],
                      title='Daily Messages with 7-Day Rolling Average')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Hourly activity heatmap
        hourly_activity = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='MessageCount')
        hourly_pivot = hourly_activity.pivot(index='DayOfWeek', columns='Hour', values='MessageCount').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_pivot = hourly_pivot.reindex(day_order)
        
        fig = px.imshow(hourly_pivot, 
                       title='Activity Heatmap: Messages by Hour and Day of Week',
                       labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak hours
        hour_counts = df.groupby('Hour').size()
        fig2 = px.bar(x=hour_counts.index, y=hour_counts.values,
                     title='Messages by Hour of Day',
                     labels={'x': 'Hour of Day', 'y': 'Number of Messages'})
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Day of week analysis
        day_counts = df.groupby('DayOfWeek').size().reindex(day_order)
        fig = px.bar(x=day_counts.index, y=day_counts.values,
                    title='Messages by Day of Week',
                    labels={'x': 'Day of Week', 'y': 'Number of Messages'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Monthly trends
        monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='MessageCount')
        monthly_counts['YearMonth'] = monthly_counts['Year'].astype(str) + '-' + monthly_counts['Month']
        
        fig = px.bar(monthly_counts, x='YearMonth', y='MessageCount',
                    title='Messages by Month',
                    labels={'MessageCount': 'Number of Messages'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def create_message_analysis(df):
    """Create message content analysis"""
    st.header("📝 Message Content Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Length Analysis", "Content Patterns", "Word Cloud"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Message length distribution
            fig = px.histogram(df, x='MessageLength', nbins=50,
                             title='Distribution of Message Lengths')
            st.plotly_chart(fig, use_container_width=True)
            
            # Word count distribution
            fig2 = px.histogram(df, x='WordCount', nbins=50,
                              title='Distribution of Word Counts')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Length over time
            length_over_time = df.groupby('Date')['MessageLength'].mean().reset_index()
            length_over_time['Date'] = pd.to_datetime(length_over_time['Date'])
            
            fig3 = px.line(length_over_time, x='Date', y='MessageLength',
                          title='Average Message Length Over Time')
            st.plotly_chart(fig3, use_container_width=True)
            
            # Box plot by hour
            fig4 = px.box(df, x='Hour', y='MessageLength',
                         title='Message Length by Hour of Day')
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Emoji usage
            emoji_stats = {
                'Messages with Emojis': (df['EmojiCount'] > 0).sum(),
                'Total Emojis': df['EmojiCount'].sum(),
                'Avg Emojis per Message': df['EmojiCount'].mean()
            }
            
            st.subheader("🔥 Emoji Statistics")
            for key, value in emoji_stats.items():
                st.metric(key, f"{value:.2f}" if isinstance(value, float) else value)
            
            # All caps messages
            caps_count = df['IsAllCaps'].sum()
            caps_percentage = (caps_count / len(df)) * 100
            st.metric("ALL CAPS Messages", f"{caps_count} ({caps_percentage:.1f}%)")
        
        with col2:
            # Questions
            question_count = df['IsQuestion'].sum()
            question_percentage = (question_count / len(df)) * 100
            st.metric("Question Messages", f"{question_count} ({question_percentage:.1f}%)")
            
            # Attachments
            attachment_count = df['HasAttachment'].sum()
            attachment_percentage = (attachment_count / len(df)) * 100
            st.metric("Messages with Attachments", f"{attachment_count} ({attachment_percentage:.1f}%)")
    
    with tab3:
        # Word cloud
        st.subheader("☁️ Word Cloud")
        
        # Combine all message contents
        all_text = ' '.join(df['Contents'].dropna().astype(str))
        
        # Remove common words and clean text
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
        
        # Clean and filter words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        filtered_words = [word for word in words if word not in common_words]
        
        if filtered_words:
            word_freq = Counter(filtered_words)
            
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                 background_color='white',
                                 colormap='viridis',
                                 max_words=100).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Top words
            st.subheader("🔤 Most Frequent Words")
            top_words = word_freq.most_common(20)
            word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            
            fig = px.bar(word_df, x='Frequency', y='Word', orientation='h',
                        title='Top 20 Most Frequent Words')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

def create_advanced_analytics(df):
    """Create advanced analytics"""
    st.header("🔍 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Streak Analysis", "Peak Activity", "Correlation Analysis"])
    
    with tab1:
        # Consecutive days with messages
        daily_activity = df.groupby('Date').size().sort_index()
        dates = pd.date_range(start=daily_activity.index.min(), end=daily_activity.index.max())
        daily_activity = daily_activity.reindex(dates, fill_value=0)
        
        # Calculate streaks
        streaks = []
        current_streak = 0
        
        for count in daily_activity:
            if count > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        if streaks:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Longest Streak", f"{max(streaks)} days")
            with col2:
                st.metric("Average Streak", f"{np.mean(streaks):.1f} days")
            with col3:
                st.metric("Total Streaks", len(streaks))
            
            # Streak distribution
            fig = px.histogram(x=streaks, nbins=20, title='Distribution of Message Streaks')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Peak activity analysis
        st.subheader("🚀 Peak Activity Periods")
        
        # Find busiest days
        daily_counts = df.groupby('Date').size().sort_values(ascending=False)
        top_days = daily_counts.head(10)
        
        st.write("**Top 10 Busiest Days:**")
        for date, count in top_days.items():
            st.write(f"📅 {date}: {count} messages")
        
        # Peak hours by day of week
        peak_hours = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Count')
        peak_by_day = peak_hours.loc[peak_hours.groupby('DayOfWeek')['Count'].idxmax()]
        
        st.write("**Peak Hour by Day of Week:**")
        for _, row in peak_by_day.iterrows():
            st.write(f"📊 {row['DayOfWeek']}: {row['Hour']}:00 ({row['Count']} messages)")
    
    with tab3:
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Create correlation matrix
        numeric_cols = ['MessageLength', 'WordCount', 'EmojiCount', 'Hour']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Correlation Matrix of Message Features',
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='MessageLength', y='WordCount',
                           title='Message Length vs Word Count',
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='Hour', y='MessageLength',
                           title='Hour vs Message Length',
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">💬 Discord Messages Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.sidebar.markdown("### 📊 Dashboard Navigation")
    st.sidebar.markdown("Explore your Discord messaging patterns and statistics through interactive visualizations.")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Timezone Selection
    st.sidebar.header("🌍 Timezone Settings")
    
    common_timezones, all_timezones = get_common_timezones()
    
    # Timezone selection with common timezones
    timezone_option = st.sidebar.radio(
        "Select timezone option:",
        ["Common Timezones", "All Timezones", "Auto-detect"]
    )
    
    if timezone_option == "Auto-detect":
        # Try to auto-detect user's timezone
        try:
            import time
            local_tz_name = time.tzname[0]
            # This is a simple approach, might not work perfectly for all systems
            selected_timezone = 'UTC'  # Fallback
            st.sidebar.info("Auto-detection may not be accurate. Consider manually selecting your timezone.")
        except:
            selected_timezone = 'UTC'
            st.sidebar.warning("Could not auto-detect timezone. Using UTC.")
    elif timezone_option == "Common Timezones":
        selected_timezone = st.sidebar.selectbox(
            "Choose your timezone:",
            common_timezones,
            index=0,
            help="Select your local timezone to see all times converted properly"
        )
    else:  # All Timezones
        selected_timezone = st.sidebar.selectbox(
            "Choose your timezone:",
            all_timezones,
            index=all_timezones.index('UTC'),
            help="Select from all available timezones"
        )
    
    # Display selected timezone info
    if selected_timezone != 'UTC':
        try:
            tz = pytz.timezone(selected_timezone)
            current_time = datetime.now(tz)
            st.sidebar.info(f"🕐 Current time in {selected_timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except:
            st.sidebar.error(f"Invalid timezone: {selected_timezone}")
            selected_timezone = 'UTC'
    
    # Load data with timezone
    with st.spinner(f"Loading your Discord messages (timezone: {selected_timezone})..."):
        df = load_data(selected_timezone)
    
    if df is None:
        st.error("Failed to load data. Please check your data/messages.json file.")
        return
    
    # Filter options in sidebar
    st.sidebar.header("🔧 Filters")
    
    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Message length filter
    min_length = st.sidebar.slider(
        "Minimum Message Length",
        min_value=0,
        max_value=int(df['MessageLength'].max()),
        value=0
    )
    
    df = df[df['MessageLength'] >= min_length]
    
    # Display filtered data info
    st.sidebar.info(f"Showing {len(df):,} messages from {len(df['Date'].unique())} days")
    
    # Show timezone info in main area
    if selected_timezone != 'UTC':
        st.info(f"🌍 All times are displayed in **{selected_timezone}** timezone")
    
    # Overview metrics
    create_overview_metrics(df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Temporal Analysis", "📝 Message Analysis", "🔍 Advanced Analytics", "📊 Raw Data"])
    
    with tab1:
        create_temporal_analysis(df)
    
    with tab2:
        create_message_analysis(df)
    
    with tab3:
        create_advanced_analytics(df)
    
    with tab4:
        st.header("📊 Raw Data Explorer")
        
        # Display sample data
        st.subheader("Sample Messages")
        sample_size = st.selectbox("Sample size", [10, 25, 50, 100], index=1)
        st.dataframe(df.head(sample_size)[['Timestamp', 'Contents', 'MessageLength', 'WordCount', 'EmojiCount']])
        
        # Download filtered data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"discord_messages_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Statistics
        st.subheader("📈 Statistical Summary")
        st.dataframe(df[['MessageLength', 'WordCount', 'EmojiCount']].describe())

if __name__ == "__main__":
    main()