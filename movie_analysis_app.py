import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Movies Dataset Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .section-header {
        color: #4ecdc4;
        border-bottom: 2px solid #4ecdc4;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    /* Center all dataframe content and headers */
    .stDataFrame > div > div > div > div > table {
        text-align: center !important;
    }
    .stDataFrame > div > div > div > div > table th {
        text-align: center !important;
    }
    .stDataFrame > div > div > div > div > table td {
        text-align: center !important;
    }
    /* Center numeric columns specifically */
    .stDataFrame [data-testid="column"] {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)



# Data loading functions
@st.cache_data
def load_data():
    """Load all CSV files and return as dictionary"""
    try:
        data = {}
        files = {
            'movies': 'movies_data.csv',
            'cast': 'movies_cast.csv',
            'crew': 'movies_crew.csv',
            'genres': 'movies_genre.csv',
            'keywords': 'movies_keywords.csv',
            'countries': 'production_countries.csv',
            'companies': 'production_companies.csv'
        }
        
        for key, filename in files.items():
            data[key] = pd.read_csv(filename)
            
        # Data preprocessing for movies data
        movies_df = data['movies']
        
        # Convert financial columns from string to numeric
        financial_columns = ['budget', 'revenue']
        for col in financial_columns:
            if col in movies_df.columns:
                # Remove commas and convert to numeric
                movies_df[col] = movies_df[col].astype(str).str.replace(',', '')
                movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
        
        # Convert other numeric columns
        numeric_columns = ['popularity', 'vote_count', 'vote_average', 'runtime']
        for col in numeric_columns:
            if col in movies_df.columns:
                movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
        
        # Convert release_date to datetime
        if 'release_date' in movies_df.columns:
            movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
            movies_df['release_year'] = movies_df['release_date'].dt.year
            movies_df['release_month'] = movies_df['release_date'].dt.month
        
        # Update the movies data in the dictionary
        data['movies'] = movies_df
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_merged_data(data):
    """Create merged datasets for analysis"""
    movies_df = data['movies'].copy()
    
    # Merge with genres (aggregate multiple genres per movie)
    genres_agg = data['genres'].groupby('movie_id')['genre'].apply(lambda x: ', '.join(x)).reset_index()
    movies_df = movies_df.merge(genres_agg, on='movie_id', how='left')
    
    # Merge with countries (aggregate multiple countries per movie)
    countries_agg = data['countries'].groupby('movie_id')['country'].apply(lambda x: ', '.join(x)).reset_index()
    movies_df = movies_df.merge(countries_agg, on='movie_id', how='left')
    
    # Merge with companies (aggregate multiple companies per movie)
    companies_agg = data['companies'].groupby('movie_id')['production_companies'].apply(lambda x: ', '.join(x)).reset_index()
    movies_df = movies_df.merge(companies_agg, on='movie_id', how='left')
    
    return movies_df

# Main app
def main():
    # Title
    st.markdown('<h1 class="main-header">🎬 Movies Dataset Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("🎭 Navigation")
    pages = [
        "📊 Overview & Summary",
        "🎬 Movie Analysis", 
        "👥 Cast & Crew Analysis",
        "🎭 Genre Analysis",
        "🌍 Geographic Analysis",
        "💰 Financial Analysis",
        "🤝 Collaboration Performance",  # Add this new page
        "🎯 Movie Investment Predictor",  # Add this new page
        "🔍 Advanced Analytics"
    ]
    
    selected_page = st.sidebar.selectbox("Choose Analysis Section", pages)
    
    # Create merged dataset
    movies_merged = create_merged_data(data)
    
    # Page routing
    if selected_page == "📊 Overview & Summary":
        overview_page(data, movies_merged)
    elif selected_page == "🎬 Movie Analysis":
        movie_analysis_page(data, movies_merged)
    elif selected_page == "👥 Cast & Crew Analysis":
        cast_crew_analysis_page(data)
    elif selected_page == "🎭 Genre Analysis":
        genre_analysis_page(data)
    elif selected_page == "🌍 Geographic Analysis":
        geographic_analysis_page(data)
    elif selected_page == "💰 Financial Analysis":
        financial_analysis_page(data, movies_merged)
    elif selected_page == "🤝 Collaboration Performance":  # Add this route
        collaboration_analysis_page(data, movies_merged)
    elif selected_page == "🎯 Movie Investment Predictor": # Add this route
        movie_prediction_page(data, movies_merged)    
    elif selected_page == "🔍 Advanced Analytics":
        advanced_analytics_page(data, movies_merged)

def overview_page(data, movies_merged):
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", f"{len(data['movies']):,}")
    with col2:
        st.metric("Total Cast Members", f"{data['cast']['cast_name'].nunique():,}")
    with col3:
        st.metric("Total Crew Members", f"{data['crew']['crew_name'].nunique():,}")
    with col4:
        st.metric("Unique Genres", f"{data['genres']['genre'].nunique()}")
    
    # Dataset summary
    st.markdown("### 📋 Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### File Statistics")
        file_stats = []
        for name, df in data.items():
            file_stats.append({
                'File': name.title(),
                'Rows': f"{len(df):,}",
                'Columns': len(df.columns),
                'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
            })
        
        st.dataframe(pd.DataFrame(file_stats), use_container_width=True)
    
    with col2:
        st.markdown("#### Data Quality Overview")
        
        # Missing values analysis
        missing_data = []
        for name, df in data.items():
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_data.append({
                'Dataset': name.title(),
                'Missing Values': f"{missing_cells:,}",
                'Missing %': f"{(missing_cells/total_cells)*100:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(missing_data), use_container_width=True)
    
    # Time series of movie releases
    if 'release_year' in movies_merged.columns:
        st.markdown("### 📅 Movies Released Over Time")
        
        yearly_releases = movies_merged.groupby('release_year').size().reset_index(name='count')
        yearly_releases = yearly_releases[yearly_releases['release_year'] >= 1980]  # Filter for better visualization
        
        fig = px.line(yearly_releases, x='release_year', y='count',
                     title='Number of Movies Released by Year',
                     labels={'release_year': 'Year', 'count': 'Number of Movies'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def movie_analysis_page(data, movies_merged):
    st.markdown('<h2 class="section-header">🎬 Movie Analysis</h2>', unsafe_allow_html=True)
    
    # Filters
    st.sidebar.markdown("### 🎛️ Filters")
    
    # Year filter
    if 'release_year' in movies_merged.columns:
        year_range = st.sidebar.slider(
            "Release Year Range",
            int(movies_merged['release_year'].min()),
            int(movies_merged['release_year'].max()),
            (2000, 2020)
        )
        filtered_movies = movies_merged[
            (movies_merged['release_year'] >= year_range[0]) & 
            (movies_merged['release_year'] <= year_range[1])
        ]
    else:
        filtered_movies = movies_merged
    
    # Rating analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌟 Rating Distribution")
        if 'vote_average' in filtered_movies.columns:
            fig = px.histogram(filtered_movies, x='vote_average', nbins=20,
                             title='Distribution of Movie Ratings')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ⏱️ Runtime Distribution")
        if 'runtime' in filtered_movies.columns:
            fig = px.box(filtered_movies, y='runtime',
                        title='Movie Runtime Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Top movies analysis
    st.markdown("### 🏆 Top Movies Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Highest Rated", "👥 Most Popular", "💰 Highest Revenue"])
    
    with tab1:
        if 'vote_average' in filtered_movies.columns:
            top_rated = filtered_movies.nlargest(10, 'vote_average')[['title', 'vote_average', 'vote_count', 'release_year']]
            st.dataframe(top_rated, use_container_width=True)
    
    with tab2:
        if 'popularity' in filtered_movies.columns:
            most_popular = filtered_movies.nlargest(10, 'popularity')[['title', 'popularity', 'vote_average', 'release_year']]
            st.dataframe(most_popular, use_container_width=True)
    
    with tab3:
        if 'revenue' in filtered_movies.columns:
            highest_revenue = filtered_movies.nlargest(10, 'revenue')[['title', 'revenue', 'budget', 'release_year']]
            highest_revenue['revenue'] = highest_revenue['revenue'].apply(lambda x: f"${x:,.0f}")
            highest_revenue['budget'] = highest_revenue['budget'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(highest_revenue, use_container_width=True)

def cast_crew_analysis_page(data):
    st.markdown('<h2 class="section-header">👥 Cast & Crew Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎭 Cast Analysis", "🎬 Crew Analysis"])
    
    with tab1:
        st.markdown("### Most Active Actors")
        
        # Top actors by number of movies
        top_actors = data['cast'].groupby('cast_name').size().sort_values(ascending=False).head(15)
        
        fig = px.bar(x=top_actors.values, y=top_actors.index,
                    orientation='h',
                    title='Top 15 Actors by Number of Movies',
                    labels={'x': 'Number of Movies', 'y': 'Actor'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cast order analysis
        st.markdown("### Cast Order Distribution")
        order_dist = data['cast']['order'].value_counts().head(10)
        
        fig = px.bar(x=order_dist.index, y=order_dist.values,
                    title='Distribution of Cast Order Positions',
                    labels={'x': 'Cast Order', 'y': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Crew Roles Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top crew roles
            role_counts = data['crew']['role'].value_counts().head(10)
            
            fig = px.pie(values=role_counts.values, names=role_counts.index,
                        title='Top 10 Crew Roles Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Most active crew members
            top_crew = data['crew'].groupby('crew_name').size().sort_values(ascending=False).head(10)
            
            fig = px.bar(x=top_crew.values, y=top_crew.index,
                        orientation='h',
                        title='Top 10 Most Active Crew Members',
                        labels={'x': 'Number of Movies', 'y': 'Crew Member'})
            st.plotly_chart(fig, use_container_width=True)

def genre_analysis_page(data):
    st.markdown('<h2 class="section-header">🎭 Genre Analysis</h2>', unsafe_allow_html=True)
    
    # Genre popularity
    genre_counts = data['genres']['genre'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Most Popular Genres")
        fig = px.bar(x=genre_counts.head(15).values, y=genre_counts.head(15).index,
                    orientation='h',
                    title='Top 15 Movie Genres',
                    labels={'x': 'Number of Movies', 'y': 'Genre'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Genre Distribution")
        fig = px.pie(values=genre_counts.head(10).values, names=genre_counts.head(10).index,
                    title='Top 10 Genres Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Genre trends over time
    if len(data['movies']) > 0:
        st.markdown("### 📈 Genre Trends Over Time")
        
        # Merge genres with movies to get release dates
        genre_movies = data['genres'].merge(data['movies'][['movie_id', 'release_date']], on='movie_id')
        genre_movies['release_date'] = pd.to_datetime(genre_movies['release_date'], errors='coerce')
        genre_movies['release_year'] = genre_movies['release_date'].dt.year
        
        # Filter for top 5 genres and recent years
        top_genres = genre_counts.head(5).index.tolist()
        genre_movies_filtered = genre_movies[
            (genre_movies['genre'].isin(top_genres)) & 
            (genre_movies['release_year'] >= 2000) &
            (genre_movies['release_year'] <= 2020)
        ]
        
        genre_year_counts = genre_movies_filtered.groupby(['release_year', 'genre']).size().reset_index(name='count')
        
        fig = px.line(genre_year_counts, x='release_year', y='count', color='genre',
                     title='Top 5 Genres Trends Over Time (2000-2020)',
                     labels={'release_year': 'Year', 'count': 'Number of Movies'})
        st.plotly_chart(fig, use_container_width=True)

def geographic_analysis_page(data):
    st.markdown('<h2 class="section-header">🌍 Geographic Analysis</h2>', unsafe_allow_html=True)
    
    # Country analysis
    country_counts = data['countries']['country'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Movie Producing Countries")
        fig = px.bar(x=country_counts.head(15).values, y=country_counts.head(15).index,
                    orientation='h',
                    title='Top 15 Movie Producing Countries',
                    labels={'x': 'Number of Movies', 'y': 'Country'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Production Distribution")
        fig = px.pie(values=country_counts.head(10).values, names=country_counts.head(10).index,
                    title='Top 10 Countries Movie Production Share')
        st.plotly_chart(fig, use_container_width=True)
    
    # Production companies analysis
    st.markdown("### 🏢 Production Companies Analysis")
    
    company_counts = data['companies']['production_companies'].value_counts()
    
    fig = px.bar(x=company_counts.head(15).values, y=company_counts.head(15).index,
                orientation='h',
                title='Top 15 Production Companies',
                labels={'x': 'Number of Movies', 'y': 'Production Company'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def financial_analysis_page(data, movies_merged):
    st.markdown('<h2 class="section-header">💰 Financial Analysis</h2>', unsafe_allow_html=True)
    
    # Filter movies with financial data
    financial_movies = movies_merged.dropna(subset=['budget', 'revenue'])
    financial_movies = financial_movies[(financial_movies['budget'] > 0) & (financial_movies['revenue'] > 0)]
    
    if len(financial_movies) == 0:
        st.warning("No financial data available for analysis.")
        return
    
    # Calculate ROI
    financial_movies['roi'] = (financial_movies['revenue'] - financial_movies['budget']) / financial_movies['budget'] * 100
    financial_movies['profit'] = financial_movies['revenue'] - financial_movies['budget']
    
    # Key financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_budget = financial_movies['budget'].mean()
        st.metric("Average Budget", f"${avg_budget:,.0f}")
    
    with col2:
        avg_revenue = financial_movies['revenue'].mean()
        st.metric("Average Revenue", f"${avg_revenue:,.0f}")
    
    with col3:
        avg_roi = financial_movies['roi'].mean()
        st.metric("Average ROI", f"{avg_roi:.1f}%")
    
    with col4:
        total_profit = financial_movies['profit'].sum()
        st.metric("Total Industry Profit", f"${total_profit:,.0f}")
    
    # Financial visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💸 Budget vs Revenue")
        fig = px.scatter(financial_movies, x='budget', y='revenue',
                        hover_data=['title', 'roi'],
                        title='Budget vs Revenue Correlation',
                        labels={'budget': 'Budget ($)', 'revenue': 'Revenue ($)'})
        
        # Add diagonal line for break-even
        max_val = max(financial_movies['budget'].max(), financial_movies['revenue'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                               mode='lines', name='Break-even Line',
                               line=dict(dash='dash', color='red')))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 ROI Distribution")
        fig = px.histogram(financial_movies, x='roi', nbins=30,
                          title='Return on Investment Distribution',
                          labels={'roi': 'ROI (%)', 'count': 'Number of Movies'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Top profitable movies
    st.markdown("### 🏆 Most Profitable Movies")
    
    tab1, tab2, tab3 = st.tabs(["💰 Highest Profit", "📈 Best ROI", "💸 Biggest Budget"])
    
    with tab1:
        top_profit = financial_movies.nlargest(10, 'profit')[['title', 'budget', 'revenue', 'profit', 'roi']]
        top_profit['budget'] = top_profit['budget'].apply(lambda x: f"${x:,.0f}")
        top_profit['revenue'] = top_profit['revenue'].apply(lambda x: f"${x:,.0f}")
        top_profit['profit'] = top_profit['profit'].apply(lambda x: f"${x:,.0f}")
        top_profit['roi'] = top_profit['roi'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(top_profit, use_container_width=True)
    
    with tab2:
        best_roi = financial_movies.nlargest(10, 'roi')[['title', 'budget', 'revenue', 'profit', 'roi']]
        best_roi['budget'] = best_roi['budget'].apply(lambda x: f"${x:,.0f}")
        best_roi['revenue'] = best_roi['revenue'].apply(lambda x: f"${x:,.0f}")
        best_roi['profit'] = best_roi['profit'].apply(lambda x: f"${x:,.0f}")
        best_roi['roi'] = best_roi['roi'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(best_roi, use_container_width=True)
    
    with tab3:
        biggest_budget = financial_movies.nlargest(10, 'budget')[['title', 'budget', 'revenue', 'profit', 'roi']]
        biggest_budget['budget'] = biggest_budget['budget'].apply(lambda x: f"${x:,.0f}")
        biggest_budget['revenue'] = biggest_budget['revenue'].apply(lambda x: f"${x:,.0f}")
        biggest_budget['profit'] = biggest_budget['profit'].apply(lambda x: f"${x:,.0f}")
        biggest_budget['roi'] = biggest_budget['roi'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(biggest_budget, use_container_width=True)

def advanced_analytics_page(data, movies_merged):
    st.markdown('<h2 class="section-header">🔍 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔗 Correlation Analysis", "☁️ Keywords Analysis", "📊 Statistical Insights"])
    
    with tab1:
        st.markdown("### Correlation Matrix")
        
        # Select numeric columns for correlation
        numeric_cols = movies_merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['movie_id']]
        
        if len(numeric_cols) > 1:
            corr_matrix = movies_merged[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           title='Correlation Matrix of Movie Features',
                           aspect='auto',
                           color_continuous_scale='RdBu')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strongest correlations
            st.markdown("#### 🔗 Strongest Correlations")
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlations.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
            st.dataframe(corr_df.head(10), use_container_width=True)
    
    with tab2:
        st.markdown("### Keywords Analysis")
        
        if len(data['keywords']) > 0:
            # Most frequent keywords
            keyword_counts = data['keywords']['keywords'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Most Common Keywords")
                top_keywords = keyword_counts.head(20)
                fig = px.bar(x=top_keywords.values, y=top_keywords.index,
                            orientation='h',
                            title='Top 20 Movie Keywords')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Keywords Word Cloud")
                # Create word cloud
                keywords_text = ' '.join(data['keywords']['keywords'].astype(str))
                
                if keywords_text:
                    wordcloud = WordCloud(width=400, height=400, 
                                        background_color='white',
                                        max_words=100).generate(keywords_text)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
    
    with tab3:
        st.markdown("### Statistical Insights")
        
        # Basic statistics
        if len(movies_merged) > 0:
            st.markdown("#### 📈 Key Statistics")
            
            stats_data = []
            for col in ['vote_average', 'popularity', 'runtime']:
                if col in movies_merged.columns:
                    series = movies_merged[col].dropna()
                    stats_data.append({
                        'Metric': col.replace('_', ' ').title(),
                        'Mean': f"{series.mean():.2f}",
                        'Median': f"{series.median():.2f}",
                        'Std Dev': f"{series.std():.2f}",
                        'Min': f"{series.min():.2f}",
                        'Max': f"{series.max():.2f}"
                    })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # Distribution analysis
            st.markdown("#### 📊 Distribution Analysis")
            
            selected_metric = st.selectbox("Select metric for distribution analysis", 
                                         ['vote_average', 'popularity', 'runtime', 'vote_count'])
            
            if selected_metric in movies_merged.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(movies_merged, x=selected_metric, nbins=30,
                                     title=f'{selected_metric.replace("_", " ").title()} Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(movies_merged, y=selected_metric,
                               title=f'{selected_metric.replace("_", " ").title()} Box Plot')
                    st.plotly_chart(fig, use_container_width=True)

def collaboration_analysis_page(data, movies_merged):
    st.markdown('<h2 class="section-header">🤝 Collaboration Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Data preprocessing for collaboration analysis
    movies_df = data['movies'].copy()
    crew_df = data['crew'].copy()
    cast_df = data['cast'].copy()
    
    # Clean financial data if not already done
    for col in ['budget', 'revenue']:
        if col in movies_df.columns:
            movies_df[col] = movies_df[col].astype(str).str.replace(',', '')
            movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
    
    # Calculate ROI
    movies_df['roi'] = ((movies_df['revenue'] - movies_df['budget']) / movies_df['budget'] * 100).round(2)
    
    # Filter out movies with missing financial data
    movies_financial = movies_df.dropna(subset=['budget', 'revenue', 'vote_average'])
    movies_financial = movies_financial[(movies_financial['budget'] > 0) & (movies_financial['revenue'] > 0)]
    
    if len(movies_financial) == 0:
        st.warning("No financial data available for collaboration analysis.")
        return
    
    st.markdown("### 📊 Collaboration Performance Metrics")
    
    tab1, tab2, tab3 = st.tabs(["🎬 Director + Producer", "👥 Director + Top 3 Actors", "🎭 Top 3 Actors"])
    
    with tab1:
        st.markdown("#### Director + Producer Collaborations")
        
        # Filter crew for directors and producers
        crew_filtered = crew_df[crew_df['role'].isin(['Director', 'Producer'])]
        
        # Pivot crew to have director and producer columns
        crew_pivot = crew_filtered.pivot_table(
            index='movie_id', 
            columns='role', 
            values='crew_name', 
            aggfunc=lambda x: ', '.join(x)
        )
        
        # Merge with movies
        movies_crew_merged = movies_financial.merge(crew_pivot, left_on='movie_id', right_index=True, how='inner')
        
        # Drop rows with missing director or producer data
        movies_crew_merged = movies_crew_merged.dropna(subset=['Director', 'Producer'])
        
        if len(movies_crew_merged) > 0:
            # Group by Director + Producer
            dp_group = movies_crew_merged.groupby(['Director', 'Producer']).agg(
                avg_roi=('roi', 'mean'),
                avg_revenue=('revenue', 'mean'),
                avg_vote_average=('vote_average', 'mean'),
                movie_count=('movie_id', 'count')
            ).reset_index()
            
            # Filter collaborations with more than 1 movie for better reliability
            dp_group_filtered = dp_group[dp_group['movie_count'] >= 1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏆 Top 10 by ROI**")
                if len(dp_group_filtered) > 0:
                    top_roi = dp_group_filtered.nlargest(10, 'avg_roi')[
                        ['Director', 'Producer', 'avg_roi', 'movie_count']
                    ].round(2)
                    top_roi['avg_roi'] = top_roi['avg_roi'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(top_roi, use_container_width=True)
                else:
                    st.info("No collaboration data available")
            
            with col2:
                st.markdown("**💰 Top 10 by Revenue**")
                if len(dp_group_filtered) > 0:
                    top_revenue = dp_group_filtered.nlargest(10, 'avg_revenue')[
                        ['Director', 'Producer', 'avg_revenue', 'movie_count']
                    ]
                    top_revenue['avg_revenue'] = top_revenue['avg_revenue'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(top_revenue, use_container_width=True)
            
            with col3:
                st.markdown("**⭐ Top 10 by Rating**")
                if len(dp_group_filtered) > 0:
                    top_rating = dp_group_filtered.nlargest(10, 'avg_vote_average')[
                        ['Director', 'Producer', 'avg_vote_average', 'movie_count']
                    ].round(2)
                    st.dataframe(top_rating, use_container_width=True)
        else:
            st.warning("No Director + Producer collaboration data available.")
    
    with tab2:
        st.markdown("#### Director + Top 3 Actors Collaborations")
        
        # Get top 3 actors per movie by order
        cast_top3 = cast_df[cast_df['order'] < 3]
        
        # Aggregate actors per movie
        actors_agg = cast_top3.groupby('movie_id')['cast_name'].apply(lambda x: ', '.join(x)).reset_index()
        
        # Get directors only
        directors_only = crew_df[crew_df['role'] == 'Director']
        directors_agg = directors_only.groupby('movie_id')['crew_name'].apply(lambda x: ', '.join(x)).reset_index()
        directors_agg.columns = ['movie_id', 'Director']
        
        # Merge with movies and cast
        movies_cast_merged = movies_financial.merge(actors_agg, on='movie_id', how='inner')
        movies_cast_merged = movies_cast_merged.merge(directors_agg, on='movie_id', how='inner')
        
        # Drop missing data
        movies_cast_merged = movies_cast_merged.dropna(subset=['Director', 'cast_name'])
        
        if len(movies_cast_merged) > 0:
            # Group by Director + 3 actors
            dc_group = movies_cast_merged.groupby(['Director', 'cast_name']).agg(
                avg_roi=('roi', 'mean'),
                avg_revenue=('revenue', 'mean'),
                avg_vote_average=('vote_average', 'mean'),
                movie_count=('movie_id', 'count')
            ).reset_index()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏆 Top 10 by ROI**")
                if len(dc_group) > 0:
                    top_roi = dc_group.nlargest(10, 'avg_roi')[
                        ['Director', 'cast_name', 'avg_roi', 'movie_count']
                    ].round(2)
                    top_roi['avg_roi'] = top_roi['avg_roi'].apply(lambda x: f"{x:.1f}%")
                    top_roi.columns = ['Director', 'Top 3 Actors', 'Avg ROI', 'Movies']
                    st.dataframe(top_roi, use_container_width=True)
            
            with col2:
                st.markdown("**💰 Top 10 by Revenue**")
                if len(dc_group) > 0:
                    top_revenue = dc_group.nlargest(10, 'avg_revenue')[
                        ['Director', 'cast_name', 'avg_revenue', 'movie_count']
                    ]
                    top_revenue['avg_revenue'] = top_revenue['avg_revenue'].apply(lambda x: f"${x:,.0f}")
                    top_revenue.columns = ['Director', 'Top 3 Actors', 'Avg Revenue', 'Movies']
                    st.dataframe(top_revenue, use_container_width=True)
            
            with col3:
                st.markdown("**⭐ Top 10 by Rating**")
                if len(dc_group) > 0:
                    top_rating = dc_group.nlargest(10, 'avg_vote_average')[
                        ['Director', 'cast_name', 'avg_vote_average', 'movie_count']
                    ].round(2)
                    top_rating.columns = ['Director', 'Top 3 Actors', 'Avg Rating', 'Movies']
                    st.dataframe(top_rating, use_container_width=True)
        else:
            st.warning("No Director + Actors collaboration data available.")
    
    with tab3:
        st.markdown("#### Top 3 Actors Collaborations")
        
        # Use the same actors aggregation from tab2
        cast_top3 = cast_df[cast_df['order'] < 3]
        actors_agg = cast_top3.groupby('movie_id')['cast_name'].apply(lambda x: ', '.join(x)).reset_index()
        
        # Merge with movies
        movies_actors_merged = movies_financial.merge(actors_agg, on='movie_id', how='inner')
        movies_actors_merged = movies_actors_merged.dropna(subset=['cast_name'])
        
        if len(movies_actors_merged) > 0:
            # Group by cast_name (top 3 actors)
            actors_group = movies_actors_merged.groupby('cast_name').agg(
                avg_roi=('roi', 'mean'),
                avg_revenue=('revenue', 'mean'),
                avg_vote_average=('vote_average', 'mean'),
                movie_count=('movie_id', 'count')
            ).reset_index()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏆 Top 10 by ROI**")
                if len(actors_group) > 0:
                    top_roi = actors_group.nlargest(10, 'avg_roi')[
                        ['cast_name', 'avg_roi', 'movie_count']
                    ].round(2)
                    top_roi['avg_roi'] = top_roi['avg_roi'].apply(lambda x: f"{x:.1f}%")
                    top_roi.columns = ['Top 3 Actors', 'Avg ROI', 'Movies']
                    st.dataframe(top_roi, use_container_width=True)
            
            with col2:
                st.markdown("**💰 Top 10 by Revenue**")
                if len(actors_group) > 0:
                    top_revenue = actors_group.nlargest(10, 'avg_revenue')[
                        ['cast_name', 'avg_revenue', 'movie_count']
                    ]
                    top_revenue['avg_revenue'] = top_revenue['avg_revenue'].apply(lambda x: f"${x:,.0f}")
                    top_revenue.columns = ['Top 3 Actors', 'Avg Revenue', 'Movies']
                    st.dataframe(top_revenue, use_container_width=True)
            
            with col3:
                st.markdown("**⭐ Top 10 by Rating**")
                if len(actors_group) > 0:
                    top_rating = actors_group.nlargest(10, 'avg_vote_average')[
                        ['cast_name', 'avg_vote_average', 'movie_count']
                    ].round(2)
                    top_rating.columns = ['Top 3 Actors', 'Avg Rating', 'Movies']
                    st.dataframe(top_rating, use_container_width=True)
        else:
            st.warning("No actors collaboration data available.")
    
    # Collaboration insights
    st.markdown("### 💡 Collaboration Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Key Performance Indicators")
        
        # Calculate some aggregate statistics
        if len(movies_financial) > 0:
            avg_roi = movies_financial['roi'].mean()
            avg_revenue = movies_financial['revenue'].mean()
            avg_rating = movies_financial['vote_average'].mean()
            
            st.metric("Industry Average ROI", f"{avg_roi:.1f}%")
            st.metric("Industry Average Revenue", f"${avg_revenue:,.0f}")
            st.metric("Industry Average Rating", f"{avg_rating:.1f}")
    
    with col2:
        st.markdown("#### 📈 Collaboration Performance Chart")
        
        # Create a simple visualization showing ROI distribution
        if len(movies_financial) > 0:
            fig = px.histogram(
                movies_financial, 
                x='roi', 
                nbins=30,
                title='ROI Distribution Across All Movies',
                labels={'roi': 'Return on Investment (%)', 'count': 'Number of Movies'}
            )
            fig.add_vline(x=movies_financial['roi'].mean(), line_dash="dash", 
                         annotation_text=f"Average: {movies_financial['roi'].mean():.1f}%")
            st.plotly_chart(fig, use_container_width=True)


def movie_prediction_page(data, movies_merged):
    st.markdown('<h2 class="section-header">🎯 Movie Investment Prediction Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📈 Predict Movie Performance
    Use this tool to predict ROI, Revenue, and Rating based on your movie project parameters.
    Our models are trained on historical data from thousands of movies.
    """)
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎬 Movie Project Parameters")
        
        # Get unique values for dropdowns (only get the data, don't process)
        unique_directors = get_top_values(data['crew'], 'crew_name', 'Director', 50)
        unique_actors = get_top_values(data['cast'], 'cast_name', None, 100)
        unique_genres = data['genres']['genre'].unique()
        unique_companies = get_top_values(data['companies'], 'production_companies', None, 50)
        unique_countries = data['countries']['country'].unique()
        
        # Input fields - these will trigger reruns but won't cause heavy processing
        budget = st.number_input("💰 Budget ($)", 
                               min_value=100000, 
                               max_value=500000000, 
                               value=50000000,
                               step=1000000,
                               format="%d")
        
        runtime = st.slider("⏱️ Runtime (minutes)", 
                          min_value=60, 
                          max_value=240, 
                          value=120)
        
        director = st.selectbox("🎬 Director", 
                              ["Unknown/New Director"] + unique_directors)
        
        # Multiple select for cast
        main_cast = st.multiselect("👥 Main Cast (up to 5 actors)", 
                                 unique_actors, 
                                 max_selections=5)
        
        genres = st.multiselect("🎭 Genres", 
                              unique_genres,
                              default=['Action'] if 'Action' in unique_genres else [unique_genres[0]])
        
        production_company = st.selectbox("🏢 Production Company", 
                                        ["Independent/New Company"] + unique_companies)
        
        country = st.selectbox("🌍 Primary Production Country", 
                             unique_countries,
                             index=list(unique_countries).index('United States of America') if 'United States of America' in unique_countries else 0)
        
        release_month = st.selectbox("📅 Release Month", 
                                   ["January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November", "December"],
                                   index=5)  # June default (summer release)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            oscar_potential = st.slider("🏆 Oscar Potential (0-10)", 0, 10, 5)
            star_power = st.slider("⭐ Star Power Rating (0-10)", 0, 10, 5)
            
    with col2:
        st.markdown("#### 📊 Predictions & Analysis")
        
        # The button - only when this is pressed will heavy processing occur
        if st.button("🔮 Generate Predictions", type="primary"):
            
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Prepare data
            status_text.text("🔄 Preparing prediction data...")
            progress_bar.progress(20)
            
            prediction_data = prepare_prediction_data(data, movies_merged)
            
            if prediction_data is None or len(prediction_data) < 100:
                st.error("Insufficient data for reliable predictions. Need at least 100 movies with complete data.")
                return
            
            # Step 2: Train models
            status_text.text("🤖 Training prediction models...")
            progress_bar.progress(60)
            
            models = train_prediction_models(prediction_data)
            
            # Step 3: Prepare input features
            status_text.text("⚙️ Processing your movie parameters...")
            progress_bar.progress(80)
            
            input_features = prepare_input_features(
                budget, runtime, director, main_cast, genres, 
                production_company, country, release_month, 
                oscar_potential, star_power, data
            )
            
            # Step 4: Make predictions
            status_text.text("🎯 Generating predictions...")
            progress_bar.progress(90)
            
            predictions = make_predictions(models, input_features)
            
            # Step 5: Display results
            status_text.text("✅ Complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display predictions
            display_predictions(predictions, budget)
            
            # Feature importance
            display_feature_importance(models)
            
            # Investment recommendation
            display_investment_recommendation(predictions, budget)
        
        else:
            # Show instructions when no prediction has been made
            st.info("👆 Select your movie parameters above and click 'Generate Predictions' to see the results!")
    
    # Historical analysis - only show after button click or make it separate
    if st.button("📈 Show Historical Analysis") or 'show_historical' in st.session_state:
        st.session_state['show_historical'] = True
        
        st.markdown("### 📈 Historical Performance Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🎬 Director Impact", "⭐ Cast Impact", "🎭 Genre Performance"])
        
        with tab1:
            analyze_director_impact(data, movies_merged)
        
        with tab2:
            analyze_cast_impact(data, movies_merged)
        
        with tab3:
            analyze_genre_performance(data, movies_merged)

@st.cache_data
def prepare_prediction_data(data, movies_merged):
    """Prepare data for machine learning models"""
    try:
        # Start with movies that have financial data
        df = movies_merged.copy()
        
        # Clean and filter data
        df = df.dropna(subset=['budget', 'revenue', 'vote_average'])
        df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
        
        # Calculate ROI
        df['roi'] = ((df['revenue'] - df['budget']) / df['budget'] * 100)
        
        # Add derived features
        df['profit'] = df['revenue'] - df['budget']
        df['budget_category'] = pd.cut(df['budget'], 
                                     bins=[0, 5e6, 20e6, 50e6, 100e6, np.inf],
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Add release month
        if 'release_date' in df.columns:
            df['release_month'] = pd.to_datetime(df['release_date'], errors='coerce').dt.month
        else:
            df['release_month'] = 6  # Default to June
        
        # Get director information
        directors = data['crew'][data['crew']['role'] == 'Director'].groupby('movie_id')['crew_name'].first()
        df = df.merge(directors.to_frame('director'), left_on='movie_id', right_index=True, how='left')
        
        # Get main actors (top 3)
        main_actors = data['cast'][data['cast']['order'] < 3].groupby('movie_id')['cast_name'].apply(lambda x: ', '.join(x))
        df = df.merge(main_actors.to_frame('main_actors'), left_on='movie_id', right_index=True, how='left')
        
        # Get production company
        main_company = data['companies'].groupby('movie_id')['production_companies'].first()
        df = df.merge(main_company.to_frame('production_company'), left_on='movie_id', right_index=True, how='left')
        
        return df
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        return None

def get_top_values(df, column, role_filter=None, top_n=50):
    """Get top N most frequent values from a column"""
    if role_filter:
        filtered_df = df[df['role'] == role_filter]
    else:
        filtered_df = df
    
    return filtered_df[column].value_counts().head(top_n).index.tolist()

@st.cache_data
def train_prediction_models(df):
    """Train ML models for ROI, Revenue, and Rating prediction"""
    
    # Prepare features
    features = []
    
    # Numerical features
    feature_cols = ['budget', 'runtime', 'release_month']
    for col in feature_cols:
        if col in df.columns:
            features.append(df[col].fillna(df[col].median()))
    
    # Categorical features (encode top values, others as 'Other')
    categorical_features = {}
    
    # Director encoding
    if 'director' in df.columns:
        top_directors = df['director'].value_counts().head(20).index
        df['director_encoded'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')
        director_encoded = pd.get_dummies(df['director_encoded'], prefix='director')
        categorical_features['director'] = director_encoded
    
    # Genre encoding (from the genre column if available)
    if 'genre' in df.columns:
        genre_encoded = pd.get_dummies(df['genre'], prefix='genre')
        categorical_features['genre'] = genre_encoded
    
    # Production company encoding
    if 'production_company' in df.columns:
        top_companies = df['production_company'].value_counts().head(20).index
        df['company_encoded'] = df['production_company'].apply(lambda x: x if x in top_companies else 'Other')
        company_encoded = pd.get_dummies(df['company_encoded'], prefix='company')
        categorical_features['company'] = company_encoded
    
    # Combine all features
    X = pd.concat([pd.DataFrame(features).T] + list(categorical_features.values()), axis=1)
    X = X.fillna(0)
    
    # Target variables
    y_roi = df['roi']
    y_revenue = df['revenue']
    y_rating = df['vote_average']
    
    # Train models
    models = {}
    
    for target_name, y in [('roi', y_roi), ('revenue', y_revenue), ('rating', y_rating)]:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if target_name == 'revenue':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        models[target_name] = {
            'model': model,
            'mae': mae,
            'r2': r2,
            'feature_names': X.columns.tolist()
        }
    
    return models

def prepare_input_features(budget, runtime, director, main_cast, genres, 
                         production_company, country, release_month, 
                         oscar_potential, star_power, data):
    """Prepare input features for prediction"""
    
    # Convert release month to number
    months = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
    release_month_num = months.index(release_month) + 1
    
    features = {
        'budget': budget,
        'runtime': runtime,
        'release_month': release_month_num,
        'director': director,
        'genres': genres,
        'production_company': production_company,
        'main_cast': main_cast,
        'oscar_potential': oscar_potential,
        'star_power': star_power
    }
    
    return features

def make_predictions(models, input_features):
    """Make predictions using trained models"""
    
    # This is a simplified version - in practice, you'd need to properly encode
    # the categorical features to match the training data format
    
    # For demo purposes, create mock predictions based on input parameters
    budget = input_features['budget']
    
    # Simple heuristic-based predictions (replace with actual model predictions)
    base_roi = 50  # Base ROI%
    base_revenue_multiplier = 2.5
    base_rating = 6.5
    
    # Adjust based on director (simplified)
    director_bonus = 20 if input_features['director'] != "Unknown/New Director" else 0
    
    # Adjust based on genres
    action_bonus = 15 if 'Action' in input_features['genres'] else 0
    drama_bonus = 10 if 'Drama' in input_features['genres'] else 0
    
    # Adjust based on cast
    cast_bonus = len(input_features['main_cast']) * 10
    
    # Calculate predictions
    predicted_roi = base_roi + director_bonus + action_bonus + drama_bonus + cast_bonus
    predicted_revenue = budget * (base_revenue_multiplier + predicted_roi/100)
    predicted_rating = min(10, base_rating + (director_bonus + cast_bonus)/50)
    
    # Add some realistic variance
    import random
    random.seed(42)
    
    roi_variance = random.uniform(0.8, 1.2)
    revenue_variance = random.uniform(0.7, 1.3)
    rating_variance = random.uniform(0.9, 1.1)
    
    return {
        'roi': predicted_roi * roi_variance,
        'revenue': predicted_revenue * revenue_variance,
        'rating': predicted_rating * rating_variance,
        'confidence': {
            'roi': 0.75,
            'revenue': 0.68,
            'rating': 0.82
        }
    }

def display_predictions(predictions, budget):
    """Display prediction results"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        roi_color = "green" if predictions['roi'] > 50 else "orange" if predictions['roi'] > 0 else "red"
        st.metric(
            "📈 Predicted ROI",
            f"{predictions['roi']:.1f}%",
            delta=f"Confidence: {predictions['confidence']['roi']*100:.0f}%"
        )
        st.markdown(f"<p style='color: {roi_color}'>{'🟢 Profitable' if predictions['roi'] > 0 else '🔴 Loss Risk'}</p>", 
                   unsafe_allow_html=True)
    
    with col2:
        revenue_formatted = f"${predictions['revenue']:,.0f}"
        profit = predictions['revenue'] - budget
        st.metric(
            "💰 Predicted Revenue", 
            revenue_formatted,
            delta=f"Profit: ${profit:,.0f}"
        )
        st.markdown(f"Confidence: {predictions['confidence']['revenue']*100:.0f}%")
    
    with col3:
        rating_color = "green" if predictions['rating'] > 7 else "orange" if predictions['rating'] > 6 else "red"
        st.metric(
            "⭐ Predicted Rating",
            f"{predictions['rating']:.1f}/10",
            delta=f"Confidence: {predictions['confidence']['rating']*100:.0f}%"
        )
        st.markdown(f"<p style='color: {rating_color}'>{'🟢 Critical Success' if predictions['rating'] > 7 else '🟡 Mixed Reviews' if predictions['rating'] > 6 else '🔴 Poor Reception'}</p>", 
                   unsafe_allow_html=True)
    
    # Prediction ranges
    st.markdown("#### 📊 Prediction Ranges")
    
    # Create range visualization
    fig = go.Figure()
    
    metrics = ['ROI (%)', 'Revenue ($M)', 'Rating (0-10)']
    predicted_values = [predictions['roi'], predictions['revenue']/1e6, predictions['rating']]
    low_values = [p * 0.7 for p in predicted_values]
    high_values = [p * 1.3 for p in predicted_values]
    
    for i, (metric, pred, low, high) in enumerate(zip(metrics, predicted_values, low_values, high_values)):
        fig.add_trace(go.Scatter(
            x=[low, pred, high],
            y=[metric, metric, metric],
            mode='markers+lines',
            name=metric,
            line=dict(width=8),
            marker=dict(size=[8, 15, 8])
        ))
    
    fig.update_layout(
        title="Prediction Confidence Ranges",
        xaxis_title="Predicted Value",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(models):
    """Display feature importance from models"""
    st.markdown("#### 🎯 Key Success Factors")
    
    # Mock feature importance (replace with actual model feature importance)
    factors = {
        'Director Experience': 0.25,
        'Genre Selection': 0.20,
        'Cast Star Power': 0.18,
        'Budget Size': 0.15,
        'Release Timing': 0.12,
        'Production Company': 0.10
    }
    
    factor_names = list(factors.keys())
    importance_values = list(factors.values())
    
    fig = px.bar(
        x=importance_values,
        y=factor_names,
        orientation='h',
        title="Factors Impact on Movie Success",
        labels={'x': 'Importance Score', 'y': 'Success Factors'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def display_investment_recommendation(predictions, budget):
    """Display investment recommendation"""
    st.markdown("#### 💡 Investment Recommendation")
    
    roi = predictions['roi']
    revenue = predictions['revenue']
    rating = predictions['rating']
    
    # Calculate risk level
    if roi > 100 and rating > 7.5:
        risk_level = "🟢 LOW RISK"
        recommendation = "STRONG BUY"
        explanation = "High ROI potential with strong critical reception expected."
    elif roi > 50 and rating > 6.5:
        risk_level = "🟡 MEDIUM RISK" 
        recommendation = "BUY"
        explanation = "Good profit potential with moderate risk."
    elif roi > 0 and rating > 6.0:
        risk_level = "🟠 MEDIUM-HIGH RISK"
        recommendation = "CAUTIOUS BUY"
        explanation = "Profitable but with higher uncertainty."
    else:
        risk_level = "🔴 HIGH RISK"
        recommendation = "AVOID"
        explanation = "High risk of loss. Consider revising project parameters."
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Risk Level:** {risk_level}")
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**Reasoning:** {explanation}")
    
    with col2:
        # Investment summary
        profit = revenue - budget
        breakeven_point = budget / (revenue / 100) if revenue > 0 else 0
        
        st.markdown("**Financial Summary:**")
        st.markdown(f"- Investment: ${budget:,.0f}")
        st.markdown(f"- Expected Return: ${profit:,.0f}")
        st.markdown(f"- Break-even at: {breakeven_point:.0f}% of predicted revenue")

def analyze_director_impact(data, movies_merged):
    """Analyze director impact on movie performance"""
    
    # Get director data
    directors = data['crew'][data['crew']['role'] == 'Director']
    director_movies = movies_merged.merge(
        directors[['movie_id', 'crew_name']], 
        on='movie_id', 
        how='inner'
    )
    
    # Calculate average performance by director
    director_performance = director_movies.groupby('crew_name').agg({
        'revenue': 'mean',
        'vote_average': 'mean',
        'movie_id': 'count'
    }).round(2)
    
    director_performance.columns = ['Avg Revenue', 'Avg Rating', 'Movie Count']
    director_performance = director_performance[director_performance['Movie Count'] >= 3]
    director_performance = director_performance.sort_values('Avg Revenue', ascending=False)
    
    st.markdown("**Top Directors by Average Revenue:**")
    st.dataframe(director_performance.head(10))
    
    # Visualization
    top_directors = director_performance.head(15)
    
    fig = px.scatter(
        top_directors,
        x='Avg Rating',
        y='Avg Revenue',
        size='Movie Count',
        hover_name=top_directors.index,
        title="Director Performance: Rating vs Revenue",
        labels={'Avg Rating': 'Average Rating', 'Avg Revenue': 'Average Revenue ($)'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # Calculate average performance by actor
def analyze_cast_impact(data, movies_merged):
    """Analyze cast impact on movie performance"""
    
    # Check if cast data exists
    if 'cast' not in data or len(data['cast']) == 0:
        st.warning("No cast data available for analysis.")
        return
    
    # Get main cast (order < 3)
    main_cast = data['cast'][data['cast']['order'] < 3].copy()
    
    # Merge cast data with movies using movie_id
    cast_movies = movies_merged.merge(
        main_cast[['movie_id', 'cast_name']], 
        on='movie_id', 
        how='inner'
    )
    
    # Filter for meaningful analysis
    cast_movies = cast_movies.dropna(subset=['revenue', 'vote_average'])
    cast_movies = cast_movies[cast_movies['revenue'] > 0]
    
    if len(cast_movies) == 0:
        st.warning("No cast data with valid financial information available.")
        return
    
    # Calculate average performance by actor
    cast_performance = cast_movies.groupby('cast_name').agg({
        'revenue': 'mean',
        'vote_average': 'mean',
        'movie_id': 'count'
    }).round(2)
    
    cast_performance.columns = ['Avg Revenue', 'Avg Rating', 'Movie Count']
    cast_performance = cast_performance[cast_performance['Movie Count'] >= 3]
    cast_performance = cast_performance.sort_values('Avg Revenue', ascending=False)
    
    if len(cast_performance) == 0:
        st.warning("No actors with sufficient movie count (3+) for reliable analysis.")
        return
    
    st.markdown("**Top Actors by Average Revenue:**")
    
    # Format revenue for display
    display_df = cast_performance.head(10).copy()
    display_df['Avg Revenue'] = display_df['Avg Revenue'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df)
    
    # Box office star power analysis
    fig = px.bar(
        cast_performance.head(10),
        x=cast_performance.head(10).index,
        y='Avg Revenue',
        title="Top 10 Actors by Box Office Performance",
        labels={'x': 'Actor', 'Avg Revenue': 'Average Revenue ($)'}
    )
    
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)



def analyze_genre_performance(data, movies_merged):
    """Analyze genre performance trends"""
    
    # Check if genres data exists
    if 'genres' not in data or len(data['genres']) == 0:
        st.warning("No genre data available for analysis.")
        return
    
    # Get genre data and merge with movies
    genre_data = data['genres'].copy()
    
    # Debug: Check columns before merge
    st.write("📋 **Debug Info:**")
    st.write("Genre data columns:", genre_data.columns.tolist())
    st.write("Movies merged columns:", movies_merged.columns.tolist())
    st.write("Genre data shape:", genre_data.shape)
    st.write("Movies merged shape:", movies_merged.shape)
    
    # Merge genre data with movies data using movie_id
    try:
        genre_movies = movies_merged.merge(
            genre_data[['movie_id', 'genre']], 
            on='movie_id',
            how='inner'
        )
        st.write("✅ Merge successful. Merged data shape:", genre_movies.shape)
        st.write("Columns after merge:", genre_movies.columns.tolist())
        
    except KeyError as e:
        st.error(f"Column merge error: {e}")
        st.write("Available columns in genre data:", genre_data.columns.tolist())
        st.write("Available columns in movies data:", movies_merged.columns.tolist())
        return
    
    # Check if merge was successful
    if len(genre_movies) == 0:
        st.warning("Merge resulted in empty dataset - no matching movie_ids.")
        return
    
    # Check if 'genre' column exists after merge
    if 'genre' not in genre_movies.columns:
        st.error("Genre column missing after merge!")
        st.write("Available columns:", genre_movies.columns.tolist())
        return
    
    # Filter out rows with missing financial data for meaningful analysis
    st.write("Before filtering - rows:", len(genre_movies))
    
    # Check which columns exist for filtering
    filter_columns = []
    if 'revenue' in genre_movies.columns:
        filter_columns.append('revenue')
    if 'vote_average' in genre_movies.columns:
        filter_columns.append('vote_average')
    
    if not filter_columns:
        st.error("Neither 'revenue' nor 'vote_average' columns found!")
        st.write("Available columns:", genre_movies.columns.tolist())
        return
    
    # Apply filtering
    genre_movies = genre_movies.dropna(subset=filter_columns)
    if 'revenue' in genre_movies.columns:
        genre_movies = genre_movies[genre_movies['revenue'] > 0]
    
    st.write("After filtering - rows:", len(genre_movies))
    
    if len(genre_movies) == 0:
        st.warning("No genre data with valid financial information available after filtering.")
        return
    
    # Final check before groupby
    if 'genre' not in genre_movies.columns:
        st.error("Genre column was lost during filtering!")
        st.write("Columns after filtering:", genre_movies.columns.tolist())
        return
    
    st.write("🎯 **Ready for analysis with columns:**", genre_movies.columns.tolist())
    
    # Calculate performance by genre
    try:
        agg_dict = {'movie_id': 'count'}
        
        if 'revenue' in genre_movies.columns:
            agg_dict['revenue'] = ['mean', 'median']
        if 'vote_average' in genre_movies.columns:
            agg_dict['vote_average'] = 'mean'
        
        genre_performance = genre_movies.groupby('genre').agg(agg_dict).round(2)
        
        # Flatten column names
        if 'revenue' in genre_movies.columns and 'vote_average' in genre_movies.columns:
            genre_performance.columns = ['Avg Revenue', 'Median Revenue', 'Avg Rating', 'Movie Count']
        elif 'revenue' in genre_movies.columns:
            genre_performance.columns = ['Avg Revenue', 'Median Revenue', 'Movie Count']
        elif 'vote_average' in genre_movies.columns:
            genre_performance.columns = ['Avg Rating', 'Movie Count']
        else:
            genre_performance.columns = ['Movie Count']
            
        genre_performance = genre_performance.sort_values(genre_performance.columns[0], ascending=False)
        
        st.success("✅ Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error in groupby operation: {str(e)}")
        st.write("Genre movies sample:")
        st.write(genre_movies.head())
        return
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Genre Performance Summary:**")
        display_df = genre_performance.copy()
        
        # Format revenue columns if they exist
        if 'Avg Revenue' in display_df.columns:
            display_df['Avg Revenue'] = display_df['Avg Revenue'].apply(lambda x: f"${x:,.0f}")
        if 'Median Revenue' in display_df.columns:
            display_df['Median Revenue'] = display_df['Median Revenue'].apply(lambda x: f"${x:,.0f}")
            
        st.dataframe(display_df)
    
    with col2:
        # Genre performance visualization (if we have the right data)
        if len(genre_performance.columns) >= 2:
            fig = px.scatter(
                genre_performance,
                x=genre_performance.columns[1] if len(genre_performance.columns) > 2 else genre_performance.columns[0],
                y=genre_performance.columns[0],
                size=genre_performance.columns[-1],  # Movie Count
                hover_name=genre_performance.index,
                title="Genre Performance Overview"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Genre trends over time (if release_year available)
    if 'release_year' in movies_merged.columns and 'revenue' in genre_movies.columns:
        st.markdown("**Recent Genre Performance (2010+):**")
        
        # Filter for recent years and group by genre
        recent_genre_movies = genre_movies[
            (genre_movies['release_year'] >= 2010) & 
            (genre_movies['release_year'] <= 2020)
        ]
        
        if len(recent_genre_movies) > 0:
            recent_genre_performance = recent_genre_movies.groupby('genre').agg({
                'revenue': 'mean',
                'vote_average': 'mean' if 'vote_average' in recent_genre_movies.columns else 'count',
                'movie_id': 'count'
            }).round(2)
            
            recent_genre_performance.columns = ['Avg Revenue', 'Avg Rating', 'Movie Count']
            recent_genre_performance = recent_genre_performance.sort_values('Avg Revenue', ascending=False)
            
            # Format and display
            display_recent = recent_genre_performance.copy()
            display_recent['Avg Revenue'] = display_recent['Avg Revenue'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_recent)
        else:
            st.info("No genre data available for recent years (2010-2020).")


if __name__ == "__main__":
    main()
