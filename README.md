# 🎬 Movies Dataset Analysis Dashboard

A comprehensive Streamlit web application for analyzing movies dataset across seven CSV files with advanced machine learning predictions for movie investors. This interactive dashboard provides extensive data analysis, visualizations, and investment recommendations for the movie industry.

## 📋 Table of Contents

- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard Sections](#dashboard-sections)
- [Machine Learning Features](#machine-learning-features)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔍 **Comprehensive Analysis**
- **Multi-dataset Integration**: Seamlessly analyzes 7 interconnected CSV files with 5,868 movies
- **Interactive Filtering**: Dynamic filters for year range, genres, and other criteria
- **Real-time Calculations**: Live computation of ROI, correlations, and statistical metrics
- **Professional Visualizations**: 25+ interactive charts and graphs using Plotly

### 📊 **Advanced Analytics**
- **Financial Analysis**: Budget vs Revenue correlation, ROI calculations, profitability insights
- **Statistical Insights**: Correlation matrices, distribution analysis, trend identification
- **Geographic Intelligence**: Production countries and companies analysis across 81 countries
- **Cast & Crew Analytics**: Performance metrics for 70,813+ actors and 66,616+ crew members
- **Collaboration Analysis**: Director-Producer, Director-Actor, and Actor trio performance metrics

### 🎯 **Investment Prediction Tool**
- **ROI Prediction**: Machine learning-powered return on investment forecasting
- **Revenue Forecasting**: Predict box office performance based on project parameters
- **Risk Assessment**: Color-coded investment recommendations (Low/Medium/High risk)
- **Feature Importance**: Identify key factors that drive movie success
- **Interactive Parameter Selection**: Budget, cast, crew, genre, release timing selection

### 🎨 **User Experience**
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Professional Styling**: Custom CSS with gradient themes and modern UI elements
- **Performance Optimized**: Efficient data caching and processing for 400,000+ records
- **Interactive Navigation**: Sidebar navigation with 8 distinct analysis sections

## 📁 Dataset Structure

The application analyzes seven interconnected CSV files with comprehensive movie industry data:

| File | Rows | Columns | Key Data |
|------|------|---------|----------|
| `movies_data.csv` | 5,868 | 11 | Core movie information with financial data, ratings, popularity |
| `movies_cast.csv` | 142,110 | 4 | Complete cast information with billing order for 70,813 actors |
| `movies_crew.csv` | 176,432 | 4 | Director, producer, and crew data (443 unique roles, 66,616 crew members) |
| `movies_genre.csv` | 15,290 | 2 | Movie genres (17 unique genres, multiple per movie) |
| `movies_keywords.csv` | 49,436 | 2 | Movie keywords and tags (11,524 unique keywords) |
| `production_countries.csv` | 8,253 | 3 | Production countries (81 countries with symbols) |
| `production_companies.csv` | 18,567 | 2 | Production company information (6,096 companies) |

### 📈 **Dataset Statistics**
- **Total Records**: 400,000+ individual data points
- **Time Range**: Multi-decade movie collection
- **Languages**: 31 original languages represented
- **Financial Data**: Budget and revenue information for 4,000+ movies
- **Missing Data**: Comprehensive data quality analysis included

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM recommended for optimal performance

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movies-analysis-dashboard.git
   cd movies-analysis-dashboard
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv movie_analysis_env
   
   # On macOS/Linux:
   source movie_analysis_env/bin/activate
   
   # On Windows:
   movie_analysis_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   Place all 7 CSV files in the project root directory:
   - `movies_data.csv`
   - `movies_cast.csv` 
   - `movies_crew.csv`
   - `movies_genre.csv`
   - `movies_keywords.csv`
   - `production_countries.csv`
   - `production_companies.csv`

5. **Run the application**
   ```bash
   streamlit run movie_analysis_app.py
   ```

6. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`
   - The dashboard will automatically load and process your data

## 💻 Usage

### Quick Start Guide
1. **Launch**: Run the app using installation instructions above
2. **Navigate**: Use the sidebar to explore different analysis sections
3. **Filter**: Apply interactive filters to customize your analysis view
4. **Predict**: Use the investment predictor for ROI and revenue forecasting
5. **Export**: Leverage Streamlit's built-in sharing and export features

### Navigation Guide
- **📊 Overview & Summary**: Dataset statistics and key metrics
- **🎬 Movie Analysis**: Ratings, runtime, and top movies analysis
- **👥 Cast & Crew Analysis**: Actor and crew performance insights
- **🎭 Genre Analysis**: Genre trends and popularity over time
- **🌍 Geographic Analysis**: Production countries and companies
- **💰 Financial Analysis**: Budget, revenue, and ROI calculations
- **🤝 Collaboration Performance**: Team collaboration analysis
- **🎯 Movie Investment Predictor**: ML-powered investment forecasting
- **🔍 Advanced Analytics**: Correlation analysis and statistical insights

## 🎯 Dashboard Sections

### 📊 **Overview & Summary**
- **Key Metrics**: Total movies (5,868), cast members (70,813), crew members (66,616)
- **Data Quality**: Missing values analysis and dataset statistics
- **Release Trends**: Time series analysis of movie releases over decades
- **Memory Usage**: Real-time memory consumption monitoring

### 🎬 **Movie Analysis**
- **Interactive Filters**: Year range selection for focused analysis
- **Rating Distribution**: Statistical analysis of vote averages (scale 0-10)
- **Runtime Analysis**: Movie duration patterns and trends
- **Top Movies**: Rankings by rating, popularity, and revenue
- **Language Distribution**: Analysis across 31 original languages

### 👥 **Cast & Crew Analysis**
- **Actor Insights**: Most active actors across 142,110 cast records
- **Cast Order Analysis**: Lead vs supporting role distribution patterns
- **Crew Roles**: Analysis of 443 unique crew positions
- **Collaboration Networks**: Most prolific crew members and their partnerships

### 🎭 **Genre Analysis**
- **Genre Popularity**: Comprehensive analysis of 17 movie genres
- **Trend Analysis**: Genre popularity evolution over time
- **Market Share**: Visual breakdown of genre distribution
- **Cross-Genre Analysis**: Movies spanning multiple genres

### 🌍 **Geographic Analysis**
- **Production Countries**: Global movie production analysis (81 countries)
- **Market Dominance**: Top producing nations and market share
- **Production Companies**: Analysis of 6,096 production companies
- **International Collaborations**: Cross-country production partnerships

### 💰 **Financial Analysis**
- **ROI Calculations**: Return on investment with profit margin analysis
- **Budget vs Revenue**: Scatter plot analysis with break-even visualization
- **Profitability Rankings**: Top movies by profit, ROI, and budget categories
- **Financial Trends**: Budget inflation and revenue growth over time

### 🤝 **Collaboration Performance**
- **Director-Producer Pairs**: Top 10 most successful collaborations
- **Director-Actor Teams**: Analysis of director and top 3 actor combinations
- **Actor Ensembles**: Most profitable actor trio combinations
- **Performance Metrics**: ROI, revenue, and rating analysis for each collaboration type

### 🎯 **Movie Investment Predictor**
- **ROI Prediction**: ML-powered investment return forecasting
- **Revenue Forecasting**: Box office prediction based on project parameters
- **Parameter Input**: Budget, cast, crew, genre, release timing selection
- **Risk Assessment**: Automated investment recommendation system
- **Confidence Intervals**: Prediction uncertainty quantification
- **Feature Importance**: Key success factor identification

### 🔍 **Advanced Analytics**
- **Correlation Matrix**: Relationship analysis between movie features
- **Keywords Analysis**: Word cloud and frequency analysis (11,524 keywords)
- **Statistical Distributions**: Comprehensive statistical summaries
- **Trend Analysis**: Multi-variate trend identification

## 🤖 Machine Learning Features

### **Prediction Models**
- **Random Forest Regressor**: For ROI and rating predictions
- **Gradient Boosting**: For revenue forecasting
- **Feature Engineering**: Automated categorical encoding
- **Model Validation**: Cross-validation with accuracy metrics

### **Input Parameters**
- **Financial**: Budget allocation ($100K - $500M range)
- **Talent**: Director and cast selection from database
- **Creative**: Genre combination and keyword selection
- **Production**: Company and country selection
- **Timing**: Release month optimization
- **Advanced**: Oscar potential and star power ratings

### **Output Predictions**
- **ROI Percentage**: Expected return on investment
- **Revenue Forecast**: Predicted box office performance
- **Critical Rating**: Expected review scores (0-10 scale)
- **Risk Level**: Investment risk categorization
- **Confidence Scores**: Prediction reliability metrics

## 🛠 Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Streamlit** | Web framework and UI | 1.28.0+ |
| **Pandas** | Data manipulation and analysis | 1.5.0+ |
| **Plotly** | Interactive visualizations | 5.0.0+ |
| **NumPy** | Numerical computing | 1.21.0+ |
| **Scikit-learn** | Machine learning models | 1.0.0+ |
| **Seaborn** | Statistical plotting | 0.11.0+ |
| **Matplotlib** | Additional plotting capabilities | 3.5.0+ |
| **WordCloud** | Text visualization | 1.9.0+ |

## 📂 File Structure

```
movies-analysis-dashboard/
│
├── movie_analysis_app.py          # Main Streamlit application (1,500+ lines)
├── requirements.txt               # Python dependencies
├── README.md                     # Comprehensive project documentation
│
├── data/                         # CSV data files (place your files here)
│   ├── movies_data.csv           # Core movie data (5,868 movies)
│   ├── movies_cast.csv           # Cast information (142,110 records)
│   ├── movies_crew.csv           # Crew data (176,432 records)
│   ├── movies_genre.csv          # Genre mappings (15,290 records)
│   ├── movies_keywords.csv       # Keywords (49,436 records)
│   ├── production_countries.csv  # Country data (8,253 records)
│   └── production_companies.csv  # Company info (18,567 records)
│
├── screenshots/                  # Dashboard screenshots
│   ├── dashboard-overview.png
│   ├── financial-analysis.png
│   ├── investment-predictor.png
│   └── collaboration-analysis.png
│
└── docs/                        # Additional documentation
    ├── data_dictionary.md       # Detailed data descriptions
    └── usage_guide.md           # Comprehensive usage instructions
```

## 📈 Key Insights & Analytics

### **Financial Intelligence**
- **ROI Analysis**: Industry-wide return patterns and optimization strategies
- **Budget Efficiency**: Correlation analysis between investment and success
- **Revenue Prediction**: Advanced forecasting with confidence intervals
- **Profit Margins**: Comprehensive profitability analysis

### **Market Analysis**
- **Genre Evolution**: Tracking genre popularity trends over decades
- **Global Production**: International film industry patterns and shifts
- **Studio Performance**: Production company success metrics and rankings
- **Seasonal Trends**: Release timing impact on performance

### **Collaboration Insights**
- **Dream Teams**: Most successful director-producer partnerships
- **Star Power**: Actor combination impact on box office performance
- **Career Trajectories**: Long-term success pattern analysis
- **Network Effects**: Industry collaboration network mapping

### **Predictive Intelligence**
- **Success Factors**: Data-driven identification of hit movie characteristics
- **Risk Modeling**: Investment risk quantification and mitigation strategies
- **Market Timing**: Optimal release window recommendations
- **Portfolio Optimization**: Investment diversification strategies

## 📷 Screenshots

### Dashboard Overview
![Dashboard Overview](screenshots/dashboar Analysis
![Financial Analysis](screenshots/financial-analysis.pngting

We welcome contributions to enhance this comprehensive movie analysis platform!

### **Getting Started**
1. **Fork** the repository and create a feature branch
2. **Follow** PEP 8 style guidelines for Python code
3. **Add** comprehensive docstrings and comments
4. **Test** thoroughly with sample data
5. **Update** documentation for new features
6. **Submit** pull request with detailed description

### **Contribution Areas**
- **New Visualizations**: Additional chart types and interactive features
- **ML Models**: Enhanced prediction algorithms and accuracy improvements
- **Data Sources**: Integration with additional movie databases
- **Performance**: Optimization for larger datasets
- **UI/UX**: Enhanced user interface and experience features

### **Development Guidelines**
- **Type Hints**: Use Python type annotations
- **Error Handling**: Comprehensive exception management
- **Performance**: Efficient algorithms and data structures
- **Documentation**: Clear, comprehensive documentation
- **Testing**: Unit tests for core functionality

## 📧 Contact & Support

- **Project Maintainer**: Timor Sigal
- **Email**: sigal.tim@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Tim4you/movies-analysis-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tim4you/movies-analysis-dashboard/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: Comprehensive movie industry dataset compilation
- **Streamlit Team**: Excellent framework for rapid dashboard development
- **Plotly Community**: Powerful visualization capabilities
- **Scikit-learn**: Robust machine learning library
- **Open Source Community**: Amazing tools and libraries that make this possible

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.1.0** | 2025-05-25 | Performance optimization, caching implementation, responsive UI |
| **2.0.0** | 2025-05-24 | Added ML prediction tool, collaboration analysis, enhanced UI |
| **1.1.0** | 2025-05-24 | Performance improvements, bug fixes, data validation |
| **1.0.0** | 2025-05-24 | Initial release with core analysis features |

## 🚀 Future Roadmap

### **Upcoming Features**
- **Real-time Data**: Integration with live movie databases
- **Advanced ML**: Deep learning models for enhanced predictions
- **API Integration**: RESTful API for external access
- **Mobile App**: Companion mobile application
- **Collaborative Features**: Multi-user analysis and sharing

### **Technical Enhancements**
- **Performance**: Distributed computing for large datasets
- **Scalability**: Cloud deployment and auto-scaling
- **Security**: Enterprise-grade security features
- **Integration**: Third-party service integrations

---

**⭐ Star this repository if you found it helpful!**

**🍴 Fork it to create your own movie analysis dashboard!**

**🐛 Report bugs or request features through GitHub Issues!**

**💡 Contribute to make this the best movie analysis tool available!**

---

*Built with ❤️ for the movie industry and data science community*



