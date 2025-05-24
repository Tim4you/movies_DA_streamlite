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
        numeric_columns = ['popularity', 'vote_count', 'vote_average', 'runtime', 'oscar_nominations']
        for col in numeric_columns:
            if col in movies_df.columns:
                movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
        
        # Convert release_date to datetime
        if 'release_date' in movies_df.columns:
            movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format='%d/%m/%Y', errors='coerce')
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

if __name__ == "__main__":
    main()
