
# 📈 Gold Price Prediction with LSTM & Feature Engineering

This project implements a comprehensive pipeline to forecast gold prices using historical data and machine learning techniques. It utilizes a **Bidirectional LSTM neural network**, detailed **feature engineering**, and additional analysis such as **Random Forest feature importance** and **correlation heatmaps**.

---

## 🔧 Features

- 📊 Data preprocessing and cleaning
- 🧠 Feature engineering (moving averages, volatility, percentage changes)
- 🔁 Sequence generation for LSTM
- 🤖 Bidirectional LSTM-based model for price forecasting
- 📉 Model evaluation (MAE, MSE, RMSE, MAPE, Accuracy)
- 🔍 Feature importance analysis using Random Forest
- 🔮 15-day future price prediction
- 📅 Multiple visualizations:
  - Historical trend by year
  - Monthly trends
  - Daily average trends
  - Year-over-year percentage change
  - Correlation heatmap

---

## 📂 Directory Structure

```
Gold_Price_Prediction/
│
├── Gold_Price_Prediction.py         # Main script
├── CSV_Files/
│   └── Final.csv                    # Input dataset
├── Gold_Forecast_Results/
│   ├── Saved_Models/                # Trained model and scaler
│   ├── Historical_Data_Visualizations/ # Yearly trend visualizations
│   ├── Feature_Importance_RF.png    # Feature importance plot
│   ├── Correlation_Heatmap.png      # Feature correlation plot
│   ├── Gold_Price_Future_Prediction.png # Main forecast visualization
│   ├── Prediction_Report.txt        # Summary report
│   └── ...                          # Other plots
```

---

## 📌 Requirements

Install all required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow joblib
```

---

## ▶️ How to Run

1. **Update the dataset path** (if different):  
   Edit the `DATASET_PATH` in `Gold_Price_Prediction.py`.

2. **Run the script**:

```bash
python Gold_Price_Prediction.py
```

3. **Outputs**:  
   - Forecast plots and results will be saved to the `Gold_Forecast_Results/` directory.  
   - A detailed `Prediction_Report.txt` will be generated.

---

## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Accuracy**

Metrics are logged in the console and written to the report.

---

## 📈 Example Visualizations

- `Gold_Price_Future_Prediction.png`: Complete timeline of actual, predicted, and future gold prices.
- `Correlation_Heatmap.png`: Feature correlation matrix.
- `Feature_Importance_RF.png`: Random Forest feature importance ranking.

---

## 🤖 Model Architecture

A **Bidirectional LSTM** model with:

- 2× Bidirectional LSTM (64 units)
- Dropout (20%) after each LSTM
- Dense layer (32 units, ReLU)
- Output layer (1 unit)

---

## 🔮 Future Prediction Strategy

- Forecasts gold price 15 days ahead using recent patterns.
- Predicts correlated features (`Oil_Rate`, `USD_Index_Rate`) based on learned relationships.

---

## 📝 Author Notes

- Code uses extensive logging for transparency.
- Visualizations are saved in high resolution (suitable for reports or presentations).

---

## 📬 Contact

- **Developer**: Asjal Abdullah Butt  
- **GitHub**: [AsjalAbdullahButt](https://github.com/AsjalAbdullahButt)
