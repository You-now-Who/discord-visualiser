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
    page_title="Discord Conversation Analyzer",
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
    .conversation-card {
        background: linear-gradient(45deg, #e8f5e8, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
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
def load_data(_uploaded_file, timezone_str='UTC'):
    """Load and preprocess the Discord messages data"""
    try:
        if _uploaded_file is not None:
            # Read from uploaded file
            data = json.load(_uploaded_file)
        else:
            return None
        
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

def analyze_conversations(df, gap_minutes=10):
    """Analyze messages and group them into conversations based on time gaps"""
    # Sort messages by timestamp
    df_sorted = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Calculate time differences between consecutive messages
    df_sorted['TimeDiff'] = df_sorted['Timestamp'].diff()
    
    # Create conversation groups
    df_sorted['NewConversation'] = (df_sorted['TimeDiff'] > timedelta(minutes=gap_minutes)) | (df_sorted['TimeDiff'].isna())
    df_sorted['ConversationID'] = df_sorted['NewConversation'].cumsum()
    
    # Calculate conversation statistics
    conversation_stats = df_sorted.groupby('ConversationID').agg({
        'Timestamp': ['min', 'max', 'count'],
        'MessageLength': ['sum', 'mean'],
        'WordCount': 'sum',
        'EmojiCount': 'sum'
    }).round(2)
    
    # Flatten column names
    conversation_stats.columns = ['StartTime', 'EndTime', 'MessageCount', 'TotalChars', 'AvgMessageLength', 'TotalWords', 'TotalEmojis']
    
    # Calculate conversation duration
    conversation_stats['Duration'] = (conversation_stats['EndTime'] - conversation_stats['StartTime']).dt.total_seconds() / 60  # in minutes
    
    # Add conversation metadata
    conversation_stats['Date'] = conversation_stats['StartTime'].dt.date
    conversation_stats['StartHour'] = conversation_stats['StartTime'].dt.hour
    conversation_stats['DayOfWeek'] = conversation_stats['StartTime'].dt.day_name()
    conversation_stats['Month'] = conversation_stats['StartTime'].dt.month_name()
    conversation_stats['Year'] = conversation_stats['StartTime'].dt.year
    
    # Calculate messages per minute for active conversations
    conversation_stats['MessagesPerMinute'] = conversation_stats.apply(
        lambda row: row['MessageCount'] / max(row['Duration'], 1), axis=1
    ).round(3)
    
    # Filter out single-message "conversations" for some analyses
    conversation_stats['IsRealConversation'] = conversation_stats['MessageCount'] > 1
    
    # Calculate response time (time between messages within conversations)
    df_sorted['ResponseTime'] = df_sorted.groupby('ConversationID')['TimeDiff'].transform(lambda x: x.dt.total_seconds() / 60)
    df_sorted['ResponseTime'] = df_sorted['ResponseTime'].fillna(0)
    
    return df_sorted, conversation_stats

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

def create_conversation_overview(conversation_stats):
    """Create conversation overview metrics"""
    st.header("💬 Conversation Overview")
    
    real_conversations = conversation_stats[conversation_stats['IsRealConversation']]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
        st.metric("Total Conversations", f"{len(conversation_stats):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
        st.metric("Multi-Message Conversations", f"{len(real_conversations):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
        avg_duration = real_conversations['Duration'].mean()
        st.metric("Avg Conversation Duration", f"{avg_duration:.1f} min")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="conversation-card">', unsafe_allow_html=True)
        total_time = real_conversations['Duration'].sum() / 60  # hours
        st.metric("Total Time Talking", f"{total_time:.1f} hrs")
        st.markdown('</div>', unsafe_allow_html=True)

def create_time_tracking_analysis(df_with_conversations, conversation_stats):
    """Create detailed time tracking and conversation analysis"""
    st.header("⏱️ Time Spent Talking - Deep Dive")
    
    real_conversations = conversation_stats[conversation_stats['IsRealConversation']]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Daily Time Tracking", "Conversation Intensity", "Response Patterns", "Long-term Trends"])
    
    with tab1:
        st.subheader("📅 Daily Conversation Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily total conversation time
            daily_time = real_conversations.groupby('Date')['Duration'].agg(['sum', 'count', 'mean']).reset_index()
            daily_time['Date'] = pd.to_datetime(daily_time['Date'])
            daily_time['TotalHours'] = daily_time['sum'] / 60
            daily_time['AvgConversationMins'] = daily_time['mean']
            
            fig = px.line(daily_time, x='Date', y='TotalHours',
                         title='Daily Time Spent in Conversations',
                         labels={'TotalHours': 'Hours Spent Talking'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily conversation count vs time
            fig2 = px.scatter(daily_time, x='count', y='TotalHours',
                            title='Daily Conversations vs Time Spent',
                            labels={'count': 'Number of Conversations', 'TotalHours': 'Hours Spent'})
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Weekly time patterns
            weekly_time = real_conversations.groupby('DayOfWeek')['Duration'].agg(['sum', 'mean', 'count']).reset_index()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_time['DayOfWeek'] = pd.Categorical(weekly_time['DayOfWeek'], categories=day_order, ordered=True)
            weekly_time = weekly_time.sort_values('DayOfWeek')
            weekly_time['TotalHours'] = weekly_time['sum'] / 60
            
            fig3 = px.bar(weekly_time, x='DayOfWeek', y='TotalHours',
                         title='Weekly Time Distribution',
                         labels={'DayOfWeek': 'Day of Week', 'TotalHours': 'Total Hours'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Average conversation duration by day
            fig4 = px.bar(weekly_time, x='DayOfWeek', y='mean',
                         title='Average Conversation Duration by Day',
                         labels={'DayOfWeek': 'Day of Week', 'mean': 'Avg Duration (minutes)'})
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        st.subheader("🚀 Conversation Intensity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Messages per minute distribution
            fig = px.histogram(real_conversations, x='MessagesPerMinute', nbins=50,
                             title='Conversation Intensity Distribution',
                             labels={'MessagesPerMinute': 'Messages per Minute'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Intensity vs duration scatter
            fig2 = px.scatter(real_conversations, x='Duration', y='MessagesPerMinute',
                            title='Duration vs Intensity',
                            labels={'Duration': 'Duration (minutes)', 'MessagesPerMinute': 'Messages/Minute'},
                            opacity=0.6)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Hourly intensity patterns
            hourly_intensity = real_conversations.groupby('StartHour')['MessagesPerMinute'].mean().reset_index()
            fig3 = px.line(hourly_intensity, x='StartHour', y='MessagesPerMinute',
                          title='Message Intensity by Hour of Day',
                          labels={'StartHour': 'Hour', 'MessagesPerMinute': 'Avg Messages/Minute'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Conversation length categories
            real_conversations['LengthCategory'] = pd.cut(
                real_conversations['Duration'], 
                bins=[0, 5, 15, 30, 60, float('inf')], 
                labels=['Quick (0-5min)', 'Short (5-15min)', 'Medium (15-30min)', 'Long (30-60min)', 'Extended (60min+)']
            )
            
            length_dist = real_conversations['LengthCategory'].value_counts()
            fig4 = px.pie(values=length_dist.values, names=length_dist.index,
                         title='Conversation Length Categories')
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.subheader("⚡ Response Time Patterns")
        
        # Filter out the first message of each conversation (no response time)
        response_data = df_with_conversations[df_with_conversations['ResponseTime'] > 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution (capped for readability)
            response_capped = response_data[response_data['ResponseTime'] <= 30]  # Cap at 30 minutes
            fig = px.histogram(response_capped, x='ResponseTime', nbins=50,
                             title='Response Time Distribution (≤30 min)',
                             labels={'ResponseTime': 'Response Time (minutes)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Average response time by hour
            hourly_response = response_data.groupby('Hour')['ResponseTime'].mean().reset_index()
            fig2 = px.bar(hourly_response, x='Hour', y='ResponseTime',
                         title='Average Response Time by Hour',
                         labels={'Hour': 'Hour of Day', 'ResponseTime': 'Avg Response Time (min)'})
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Response time stats
            st.subheader("📊 Response Time Statistics")
            response_stats = {
                'Median Response Time': f"{response_data['ResponseTime'].median():.1f} min",
                'Average Response Time': f"{response_data['ResponseTime'].mean():.1f} min",
                '90th Percentile': f"{response_data['ResponseTime'].quantile(0.9):.1f} min",
                'Quick Responses (<1 min)': f"{(response_data['ResponseTime'] < 1).sum():,} ({(response_data['ResponseTime'] < 1).mean()*100:.1f}%)"
            }
            
            for key, value in response_stats.items():
                st.metric(key, value)
            
            # Response time by day of week
            daily_response = response_data.groupby('DayOfWeek')['ResponseTime'].mean().reset_index()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_response['DayOfWeek'] = pd.Categorical(daily_response['DayOfWeek'], categories=day_order, ordered=True)
            daily_response = daily_response.sort_values('DayOfWeek')
            
            fig3 = px.bar(daily_response, x='DayOfWeek', y='ResponseTime',
                         title='Average Response Time by Day',
                         labels={'DayOfWeek': 'Day', 'ResponseTime': 'Avg Response Time (min)'})
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.subheader("📈 Long-term Conversation Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly conversation time trends
            monthly_trends = real_conversations.groupby(['Year', 'Month']).agg({
                'Duration': ['sum', 'count', 'mean']
            }).round(2)
            monthly_trends.columns = ['TotalMinutes', 'ConversationCount', 'AvgDuration']
            monthly_trends = monthly_trends.reset_index()
            monthly_trends['YearMonth'] = monthly_trends['Year'].astype(str) + '-' + monthly_trends['Month']
            monthly_trends['TotalHours'] = monthly_trends['TotalMinutes'] / 60
            
            fig = px.line(monthly_trends, x='YearMonth', y='TotalHours',
                         title='Monthly Total Conversation Time',
                         labels={'YearMonth': 'Month', 'TotalHours': 'Total Hours'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling averages
            if len(daily_time) > 7:
                daily_time_copy = daily_time.copy()
                daily_time_copy['7DayAvg'] = daily_time_copy['TotalHours'].rolling(window=7).mean()
                daily_time_copy['30DayAvg'] = daily_time_copy['TotalHours'].rolling(window=30).mean()
                
                fig2 = px.line(daily_time_copy, x='Date', y=['TotalHours', '7DayAvg', '30DayAvg'],
                              title='Daily Conversation Time with Moving Averages')
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Conversation efficiency over time (messages per minute trend)
            monthly_efficiency = real_conversations.groupby(['Year', 'Month'])['MessagesPerMinute'].mean().reset_index()
            monthly_efficiency['YearMonth'] = monthly_efficiency['Year'].astype(str) + '-' + monthly_efficiency['Month']
            
            fig3 = px.line(monthly_efficiency, x='YearMonth', y='MessagesPerMinute',
                          title='Monthly Average Conversation Intensity',
                          labels={'YearMonth': 'Month', 'MessagesPerMinute': 'Avg Messages/Minute'})
            fig3.update_xaxes(tickangle=45)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Longest conversations over time
            top_conversations = real_conversations.nlargest(20, 'Duration')[['StartTime', 'Duration', 'MessageCount']]
            top_conversations['Date'] = top_conversations['StartTime'].dt.date
            
            fig4 = px.scatter(top_conversations, x='Date', y='Duration', size='MessageCount',
                             title='Longest Conversations Over Time',
                             labels={'Date': 'Date', 'Duration': 'Duration (minutes)', 'MessageCount': 'Messages'})
            st.plotly_chart(fig4, use_container_width=True)

def create_conversation_deep_dive(df_with_conversations, conversation_stats):
    """Create detailed conversation analysis"""
    st.header("🔍 Conversation Deep Dive")
    
    real_conversations = conversation_stats[conversation_stats['IsRealConversation']]
    
    tab1, tab2, tab3 = st.tabs(["Top Conversations", "Conversation Heatmaps", "Quality Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Longest Conversations")
            top_duration = real_conversations.nlargest(15, 'Duration')[
                ['StartTime', 'Duration', 'MessageCount', 'TotalWords', 'MessagesPerMinute']
            ].copy()
            top_duration['StartTime'] = top_duration['StartTime'].dt.strftime('%Y-%m-%d %H:%M')
            top_duration['Duration'] = top_duration['Duration'].round(1)
            st.dataframe(top_duration, use_container_width=True)
        
        with col2:
            st.subheader("🔥 Most Active Conversations")
            top_messages = real_conversations.nlargest(15, 'MessageCount')[
                ['StartTime', 'MessageCount', 'Duration', 'TotalWords', 'MessagesPerMinute']
            ].copy()
            top_messages['StartTime'] = top_messages['StartTime'].dt.strftime('%Y-%m-%d %H:%M')
            top_messages['Duration'] = top_messages['Duration'].round(1)
            st.dataframe(top_messages, use_container_width=True)
    
    with tab2:
        # Conversation start heatmap
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        conversation_activity = real_conversations.groupby(['DayOfWeek', 'StartHour']).size().reset_index(name='ConversationCount')
        conversation_pivot = conversation_activity.pivot(index='DayOfWeek', columns='StartHour', values='ConversationCount').fillna(0)
        conversation_pivot = conversation_pivot.reindex(day_order)
        
        fig = px.imshow(conversation_pivot,
                       title='Conversation Start Times Heatmap',
                       labels=dict(x="Hour of Day", y="Day of Week", color="Conversations Started"),
                       aspect="auto",
                       color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Duration heatmap
        duration_activity = real_conversations.groupby(['DayOfWeek', 'StartHour'])['Duration'].mean().reset_index()
        duration_pivot = duration_activity.pivot(index='DayOfWeek', columns='StartHour', values='Duration').fillna(0)
        duration_pivot = duration_pivot.reindex(day_order)
        
        fig2 = px.imshow(duration_pivot,
                        title='Average Conversation Duration Heatmap',
                        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Duration (min)"),
                        aspect="auto",
                        color_continuous_scale="Plasma")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Word density analysis
            real_conversations['WordsPerMinute'] = real_conversations['TotalWords'] / real_conversations['Duration'].replace(0, 1)
            
            fig = px.scatter(real_conversations, x='Duration', y='WordsPerMinute',
                           title='Conversation Duration vs Words per Minute',
                           labels={'Duration': 'Duration (minutes)', 'WordsPerMinute': 'Words/Minute'},
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
            # Message length vs conversation length
            fig2 = px.scatter(real_conversations, x='MessageCount', y='AvgMessageLength',
                            title='Message Count vs Average Message Length',
                            labels={'MessageCount': 'Messages in Conversation', 'AvgMessageLength': 'Avg Message Length'},
                            opacity=0.6)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Conversation quality metrics
            st.subheader("💎 Conversation Quality Metrics")
            
            quality_metrics = {
                'Most Efficient (Words/Min)': f"{real_conversations['WordsPerMinute'].max():.1f}",
                'Most Intense (Msgs/Min)': f"{real_conversations['MessagesPerMinute'].max():.2f}",
                'Longest Single Conversation': f"{real_conversations['Duration'].max():.1f} min",
                'Most Messages in One Conversation': f"{int(real_conversations['MessageCount'].max())}",
                'Average Words per Conversation': f"{real_conversations['TotalWords'].mean():.0f}"
            }
            
            for key, value in quality_metrics.items():
                st.metric(key, value)
            
            # Emoji usage in conversations
            emoji_conversations = real_conversations[real_conversations['TotalEmojis'] > 0]
            if len(emoji_conversations) > 0:
                fig3 = px.scatter(emoji_conversations, x='Duration', y='TotalEmojis',
                                title='Emoji Usage vs Conversation Duration',
                                labels={'Duration': 'Duration (minutes)', 'TotalEmojis': 'Total Emojis'},
                                opacity=0.6)
                st.plotly_chart(fig3, use_container_width=True)

def main():
    """Main Streamlit app for conversation analysis"""
    
    # Header
    st.markdown('<h1 class="main-header">💬 Discord Conversation Time Analyzer</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Your Discord Messages Data")
    uploaded_file = st.file_uploader(
        "Choose your messages.json file",
        type=['json'],
        help="Upload the JSON file containing your Discord messages data."
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload your messages.json file to start analyzing your conversation patterns!")
        st.markdown("""
        ### 🎯 What This App Analyzes
        - **Conversation Clustering**: Groups messages into conversations based on time gaps
        - **Time Tracking**: Shows exactly how much time you spend talking
        - **Conversation Patterns**: Analyzes when, how long, and how intensely you chat
        - **Response Times**: Tracks how quickly you respond in conversations
        - **Quality Metrics**: Finds your most engaging and productive conversations
        """)
        return
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.sidebar.markdown("### 💬 Conversation Analysis")
    st.sidebar.markdown("Deep dive into your Discord conversation patterns and time spent talking.")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Timezone Selection
    st.sidebar.header("🌍 Timezone Settings")
    common_timezones, all_timezones = get_common_timezones()
    
    timezone_option = st.sidebar.radio(
        "Select timezone option:",
        ["Common Timezones", "All Timezones"]
    )
    
    if timezone_option == "Common Timezones":
        selected_timezone = st.sidebar.selectbox(
            "Choose your timezone:",
            common_timezones,
            index=0,
            help="Select your local timezone to see all times converted properly"
        )
    else:
        selected_timezone = st.sidebar.selectbox(
            "Choose your timezone:",
            all_timezones,
            index=all_timezones.index('UTC'),
            help="Select from all available timezones"
        )
    
    # Conversation settings
    st.sidebar.header("💬 Conversation Detection")
    gap_minutes = st.sidebar.slider(
        "Conversation Gap Threshold (minutes)",
        min_value=1,
        max_value=60,
        value=10,
        help="Messages separated by more than this time will be considered separate conversations"
    )
    
    # Load data
    with st.spinner(f"Loading and analyzing conversations (timezone: {selected_timezone})..."):
        df = load_data(uploaded_file, selected_timezone)
    
    if df is None:
        st.error("Failed to load data. Please check that your uploaded file is properly formatted.")
        return
    
    # Show file info
    st.success(f"✅ Successfully loaded {len(df):,} messages!")
    
    # Analyze conversations
    with st.spinner("Clustering messages into conversations..."):
        df_with_conversations, conversation_stats = analyze_conversations(df, gap_minutes)
    
    # Show conversation detection info
    real_conversations = conversation_stats[conversation_stats['IsRealConversation']]
    st.info(f"🔍 **Conversation Detection**: Found {len(conversation_stats):,} total conversations ({len(real_conversations):,} multi-message) using {gap_minutes}-minute gap threshold")
    
    # Show timezone info
    if selected_timezone != 'UTC':
        st.info(f"🌍 All times are displayed in **{selected_timezone}** timezone")
    
    # Create overview
    create_conversation_overview(conversation_stats)
    
    # Main analysis sections
    create_time_tracking_analysis(df_with_conversations, conversation_stats)
    create_conversation_deep_dive(df_with_conversations, conversation_stats)
    
    # Export options
    st.header("📊 Export Conversation Data")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export conversation statistics
        csv_conversations = conversation_stats.to_csv(index=False)
        st.download_button(
            label="Download Conversation Statistics CSV",
            data=csv_conversations,
            file_name=f"conversation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export messages with conversation IDs
        csv_messages = df_with_conversations[['Timestamp', 'Contents', 'ConversationID', 'ResponseTime']].to_csv(index=False)
        st.download_button(
            label="Download Messages with Conversation IDs CSV",
            data=csv_messages,
            file_name=f"messages_with_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
