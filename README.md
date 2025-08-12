# StockLensAi

![Project Image](images/stocklensai_screenshot.png)  <!-- Replace with your image path -->

StockLensAi is a project that provides a curated list of top international stocks, helping users get quick insights into key global market players. This repository contains data and tools related to analyzing and visualizing international stock information.

## Features

- List of top 100 international stocks with ticker symbols  
- Supports multiple global markets and sectors  
- Ready for integration with AI or data analysis tools  
- Simple and easy-to-understand structure for further development  

## Machine Learning Usage

This project integrates several machine learning components for enhanced stock analysis:

- **Sentiment Analysis:** Uses a pretrained FinBERT transformer model to analyze financial news headlines and classify their sentiment (Positive, Negative, Neutral).  
- **Stock Price Forecasting:** Implements Facebook's Prophet model to forecast future stock prices based on historical data, providing estimated accuracy using Mean Absolute Percentage Error (MAPE).  
- **Performance Metrics:** Calculates and displays forecasting accuracy to help users understand model reliability.

These ML features provide insights beyond raw stock data by incorporating natural language processing and time series forecasting.

## Getting Started

### Prerequisites

- Python 3.x installed on your system

### Installation & Setup

Run the following commands in your terminal to clone the repo, create and activate a virtual environment, and install dependencies:

```bash
# Clone the repository
git clone https://github.com/Manishkumarsingh41/StockLensAi.git
cd StockLensAi

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1

# macOS/Linux
# source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt





```

Contributing
Contributions are welcome! Please fork the repo and open a pull request.

License
MIT License © 2025 Manish Kumar Singh

This project is licensed under the MIT License. See the LICENSE file for details.

Credits
Project created and maintained by Manish Kumar Singh

Sentiment analysis powered by FinBERT

Forecasting powered by Prophet

Data fetched from Yahoo Finance using yfinance Python package

Follow Me
For more projects and updates, follow me on GitHub:
https://github.com/Manishkumarsingh41
