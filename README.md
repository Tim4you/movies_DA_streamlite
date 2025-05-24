# 🎬 Movies Dataset Analysis Dashboard

A comprehensive Streamlit web application for analyzing movies dataset across seven CSV files. This interactive dashboard provides extensive data analysis and visualizations for movie industry insights.

## 📋 Table of Contents

- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard Sections](#dashboard-sections)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔍 **Comprehensive Analysis**
- **Multi-dataset Integration**: Seamlessly analyzes 7 interconnected CSV files
- **Interactive Filtering**: Dynamic filters for year range, genres, and other criteria
- **Real-time Calculations**: Live computation of ROI, correlations, and statistical metrics
- **Professional Visualizations**: 20+ interactive charts and graphs using Plotly

### 📊 **Advanced Analytics**
- **Financial Analysis**: Budget vs Revenue correlation, ROI calculations, profitability insights
- **Statistical Insights**: Correlation matrices, distribution analysis, trend identification
- **Geographic Intelligence**: Production countries and companies analysis
- **Cast & Crew Analytics**: Actor/director performance metrics and collaboration patterns

### 🎨 **User Experience**
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Professional Styling**: Custom CSS with gradient themes and modern UI elements
- **Performance Optimized**: Efficient data caching and processing
- **Interactive Navigation**: Sidebar navigation with 7 distinct analysis sections

## 📁 Dataset Structure

The application analyzes seven interconnected CSV files:

| File | Rows | Key Columns | Description |
|------|------|-------------|-------------|
| `movies_data.csv` | 5,868 | movie_id, title, budget, revenue, ratings | Core movie information |
| `movies_cast.csv` | 142,110 | movie_id, cast_name, cast_id, order | Actor information and roles |
| `movies_crew.csv` | 176,432 | movie_id, role, crew_name, crew_id | Director, producer, and crew data |
| `movies_genre.csv` | 15,292 | movie_id, genre | Movie genres (multiple per movie) |
| `movies_keywords.csv` | 49,436 | movie_id, keywords | Movie keywords and tags |
| `production_countries.csv` | 8,253 | movie_id, symbol, country | Production countries |
| `production_companies.csv` | 18,567 | movie_id, production_companies | Production company information |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tim4you/movies-analysis-dashboard.git
   cd movies-analysis-dashboard
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv movie_analysis_env
   source movie_analysis_env/bin/activate  # On Windows: movie_analysis_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place all 7 CSV files in the project root directory
   - Ensure files are named exactly as specified in the dataset structure

5. **Run the application**
   ```bash
   streamlit run movie_analysis_app.py
   ```

6. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Quick Start
1. Launch the app using the installation instructions above
2. Use the sidebar navigation to explore different analysis sections
3. Apply filters to customize your analysis view
4. Hover over charts for detailed information
5. Export insights using Streamlit's built-in sharing features

### Navigation Guide
- **📊 Overview & Summary**: Dataset statistics and key metrics
- **🎬 Movie Analysis**: Ratings, runtime, and top movies analysis
- **👥 Cast & Crew Analysis**: Actor and crew performance insights
- **🎭 Genre Analysis**: Genre trends and popularity over time
- **🌍 Geographic Analysis**: Production countries and companies
- **💰 Financial Analysis**: Budget, revenue, and ROI calculations
- **🔍 Advanced Analytics**: Correlation analysis and statistical insights

## 🎯 Dashboard Sections

### 📊 Overview & Summary
- **Key Metrics**: Total movies, cast members, crew members, unique genres
- **Data Quality**: Missing values analysis and dataset statistics
- **Release Trends**: Time series of movie releases over decades

### 🎬 Movie Analysis
- **Interactive Filters**: Year range selection for focused analysis
- **Rating Distribution**: Histogram of movie ratings with statistical insights
- **Runtime Analysis**: Box plots showing movie duration patterns
- **Top Movies**: Highest rated, most popular, and highest revenue films

### 👥 Cast & Crew Analysis
- **Actor Insights**: Most active actors by number of films
- **Cast Order**: Distribution analysis of lead vs supporting roles
- **Crew Roles**: Pie charts of crew position distribution
- **Collaboration Networks**: Most prolific crew members across projects

### 🎭 Genre Analysis
- **Genre Popularity**: Bar charts of most common movie genres
- **Trend Analysis**: Line charts showing genre popularity over time
- **Distribution Pie Charts**: Visual breakdown of genre market share

### 🌍 Geographic Analysis
- **Production Countries**: Global distribution of movie production
- **Market Share**: Pie charts of top producing countries
- **Production Companies**: Analysis of major studios and independents

### 💰 Financial Analysis
- **ROI Calculations**: Return on investment analysis with profit margins
- **Budget vs Revenue**: Scatter plots with break-even line analysis
- **Profitability Rankings**: Top movies by profit, ROI, and budget size
- **Financial Trends**: Correlation between budget and success metrics

### 🔍 Advanced Analytics
- **Correlation Matrix**: Heatmap of relationships between movie features
- **Keywords Analysis**: Word clouds and frequency analysis of movie tags
- **Statistical Distributions**: Box plots and histograms with statistical summaries

## 📷 Screenshots

### Dashboard Overview
![Dashboarncial Analysis
![Financial Analysis Distribution
![Geographic Analysis](screenshots/geographic.se | Version |
|------------|---------|---------|
| **Streamlit** | Web framework | 1.28.0+ |
| **Pandas** | Data manipulation | 1.5.0+ |
| **Plotly** | Interactive visualizations | 5.0.0+ |
| **NumPy** | Numerical computing | 1.21.0+ |
| **Seaborn** | Statistical plotting | 0.11.0+ |
| **Matplotlib** | Additional plotting | 3.5.0+ |
| **WordCloud** | Text visualization | 1.9.0+ |

## 📂 File Structure

```
movies-analysis-dashboard/
│
├── movie_analysis_app.py          # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
├── screenshots/                  # Dashboard screenshots
│   ├── overview.png
│   ├── financial.png
│   └── geographic.png
│
├── data/                        # CSV data files (7 files)
│   ├── movies_data.csv
│   ├── movies_cast.csv
│   ├── movies_crew.csv
│   ├── movies_genre.csv
│   ├── movies_keywords.csv
│   ├── production_countries.csv
│   └── production_companies.csv
│
└── docs/                       # Additional documentation
    ├── data_dictionary.md
    └── analysis_methodology.md
```

## 📈 Key Insights & Analytics

### Financial Insights
- **Average ROI**: Industry-wide return on investment analysis
- **Budget Efficiency**: Correlation between budget size and success
- **Profit Leaders**: Most financially successful films and studios

### Market Analysis
- **Genre Trends**: Evolution of popular genres over decades
- **Global Production**: International film production patterns
- **Studio Performance**: Major production company market share

### Quality Metrics
- **Rating Patterns**: Distribution of critical and audience scores
- **Success Factors**: Statistical correlation between various success metrics
- **Industry Standards**: Benchmarking against industry averages

## 🤝 Contributing

We welcome contributions to improve this analysis dashboard! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 style guidelines
- **Documentation**: Update README and add docstrings for new functions
- **Testing**: Ensure new features work with sample data
- **Performance**: Consider data processing efficiency for large datasets

### Areas for Contribution
- **New Visualizations**: Additional chart types and analysis methods
- **Data Processing**: Enhanced data cleaning and preprocessing
- **UI/UX Improvements**: Better styling and user experience features
- **Performance Optimization**: Faster data loading and processing
- **Documentation**: Improved guides and tutorials

## 📧 Contact & Support

- **Project Maintainer**: Timor Sigal
- **Email**: sigal.tim@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Tim4you/movies-analysis-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tim4you/movies-analysis-dashboard/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source**: Movie dataset compilation from various entertainment industry sources
- **Streamlit Community**: For excellent documentation and community support
- **Plotly Team**: For powerful and intuitive visualization capabilities
- **Open Source Contributors**: All the amazing developers who make these tools possible

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-05-24 | Initial release with core analysis features |
| 1.1.0 | TBD | Enhanced visualizations and performance improvements |

---

**⭐ Star this repository if you found it helpful!**

**🍴 Fork it to create your own movie analysis dashboard!**

**🐛 Report bugs or request features through GitHub Issues!**
