import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, Dropout 
import joblib
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.exceptions import DataConversionWarning
import warnings
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor 

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning) 
warnings.filterwarnings(action='ignore', category=UserWarning, module='matplotlib') 

# --- Configuration ---
DATASET_PATH = '/home/kbk-soft/Desktop/Gold_Price_Prediction/CSV_Files/Final.csv'
RESULTS_DIR = "Gold_Forecast_Results"
HISTORY_DIR = os.path.join(RESULTS_DIR, "Historical_Data_Visualizations")
MODELS_DIR = os.path.join(RESULTS_DIR, "Saved_Models")
DATE_COLUMN = 'Date'
TARGET_COLUMN = 'Close' 

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Loading and Preprocessing ---
def load_data():
    """
    Loads the gold price data, performs initial cleaning, and sets the Date column as index.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Original DataFrame shape: {df.shape}")
        logger.info(f"Original DataFrame columns: {df.columns.tolist()}")
        logger.info(f"Original DataFrame head:\n{df.head()}")

        df.columns = df.columns.str.strip()

        # Drop rows where ALL values are NaN (often empty rows from CSV export)
        initial_rows = df.shape[0]
        df.dropna(how='all', inplace=True)
        if df.shape[0] < initial_rows:
            logger.warning(f"{initial_rows - df.shape[0]} rows had all NaN values and were dropped.")

        # Drop 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            logger.info("Dropped 'Unnamed: 0' column.")

        # Check for the existence of the DATE_COLUMN
        if DATE_COLUMN not in df.columns:
            logger.error(f"Date column '{DATE_COLUMN}' not found in CSV after initial cleaning. Available columns: {df.columns.tolist()}")
            return pd.DataFrame() 

        # Convert DATE_COLUMN to datetime, coercing errors to NaT 
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        df.dropna(subset=[DATE_COLUMN], inplace=True) # Drop rows where Date conversion failed
        
        # Set Date as index and sort
        df.set_index(DATE_COLUMN, inplace=True)
        df.sort_index(inplace=True)

        # Convert all other columns to numeric, handling common non-numeric characters
        for col in df.columns:
            if df[col].dtype == 'object': 
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                logger.warning(f"Column '{col}' became entirely NaN after numeric conversion. Check its original format.")

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Drop any remaining rows with NaN values (e.g., if first/last rows were NaN and couldn't be filled)
        initial_rows_after_fill = df.shape[0]
        df.dropna(inplace=True)
        if df.shape[0] < initial_rows_after_fill:
             logger.warning(f"{initial_rows_after_fill - df.shape[0]} rows still had NaN values after fill and were dropped.")

        if df.empty:
            logger.error("DataFrame became empty after comprehensive cleaning. Please inspect your original CSV formatting and contents.")
            return pd.DataFrame()

        logger.info(f"Final DataFrame shape after cleaning: {df.shape}")
        logger.info(f"Data preview after cleaning:\n{df.head()}")
        logger.info(f"Data tail after cleaning:\n{df.tail()}")
        logger.info("DataFrame info after cleaning:")
        df.info()

        return df
    except FileNotFoundError:
        logger.error(f"Error: Dataset not found at {DATASET_PATH}. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading or processing: {e}")
        return pd.DataFrame()

# --- Feature Engineering ---
def feature_engineering(df):
    """
    Creates new features based on the existing DataFrame and generates a correlation heatmap.
    """
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found for feature engineering.")
        return df

    df['Gold_MA_7'] = df[TARGET_COLUMN].rolling(window=7, min_periods=1).mean()
    df['Gold_MA_30'] = df[TARGET_COLUMN].rolling(window=30, min_periods=1).mean()
    df['Gold_Volatility_7'] = df[TARGET_COLUMN].rolling(window=7, min_periods=1).std()
    df['Gold_pct_change'] = df[TARGET_COLUMN].pct_change()
    
    if 'Oil_Rate' in df.columns:
        df['Oil_pct_change'] = df['Oil_Rate'].pct_change()
    if 'USD_Index_Rate' in df.columns:
        df['USD_pct_change'] = df['USD_Index_Rate'].pct_change()
    
    # Fill NaN values introduced by rolling/pct_change operations
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Drop any remaining rows with NaN values after feature creation (e.g., at the very beginning of the series)
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    if df.shape[0] < initial_rows:
        logger.warning(f"{initial_rows - df.shape[0]} rows dropped after feature engineering due to NaNs.")

    if df.empty:
        logger.error("DataFrame became empty after feature engineering and dropping NaNs. Cannot generate heatmap.")
        return df

    # Correlation Heatmap
    try:
        corr_fig = plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
        plt.title('Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'Correlation_Heatmap.png'))
        plt.close(corr_fig)
        logger.info("Correlation heatmap saved.")
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")

    return df

def create_sequences(data, seq_len):
    """
    Creates sequences for LSTM training.

    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0]) 
    return np.array(X), np.array(y)

# --- LSTM Model Definition ---
def build_bilstm_model(input_shape):
    """
    Builds a Bidirectional LSTM model.
    
    """
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create directories for results, historical visualizations, and models
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(HISTORY_DIR).mkdir(exist_ok=True)
    Path(MODELS_DIR).mkdir(exist_ok=True)

    # Load and preprocess data
    df = load_data()
    if df.empty:
        logger.error("Exiting due to empty DataFrame after preprocessing.")
        exit(1)

    # Perform feature engineering
    df = feature_engineering(df)
    if df.empty:
        logger.error("Exiting due to empty DataFrame after feature engineering.")
        exit(1)

    # Ensure TARGET_COLUMN is present
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found in the DataFrame after feature engineering. Exiting.")
        exit(1)

    # Reorder columns to ensure TARGET_COLUMN is at index 0 for `create_sequences` and `scaler`
    cols_ordered = [TARGET_COLUMN] + [col for col in df.columns if col != TARGET_COLUMN]
    df = df[cols_ordered]
    features = df.columns.tolist() 

    sequence_length = 60 

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # --- Train and Test Split ---
    train_size = int(len(scaled_data) * 0.8)
    
    # Check if there's enough data for training sequences
    if train_size <= sequence_length:
        logger.error(f"Not enough data for training with sequence_length={sequence_length}. "
                     f"Total data points: {len(scaled_data)}, Training size: {train_size}. "
                     f"Please reduce sequence_length or provide more data.")
        exit(1)

    train_data = scaled_data[:train_size]
    
    test_data = scaled_data[train_size - sequence_length:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    logger.info(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Final check for empty sequences
    if X_train.shape[0] == 0:
        logger.error("No training sequences could be created. Adjust data size or sequence_length.")
        exit(1)
    if X_test.shape[0] == 0:
        logger.error("No testing sequences could be created. Adjust data size or sequence_length.")
        exit(1)

    # Build and train the model
    model = build_bilstm_model((sequence_length, len(features))) # Corrected function name
    logger.info("Starting model training...")
    history = model.fit(X_train, y_train, epochs=32, batch_size=32, validation_split=0.1, verbose=1) 
    logger.info("Model training complete.")

    # --- Evaluate Model Performance ---
    predictions = model.predict(X_test, verbose=0)

    dummy_predictions = np.zeros((len(predictions), len(features)))
    dummy_predictions[:, 0] = predictions.flatten() 
    predicted_actual = scaler.inverse_transform(dummy_predictions)[:, 0]

    dummy_true_values = np.zeros((len(y_test), len(features)))
    dummy_true_values[:, 0] = y_test.flatten() 
    actual = scaler.inverse_transform(dummy_true_values)[:, 0]

    # Calculate evaluation metrics
    mae = mean_absolute_error(actual, predicted_actual)
    mse = mean_squared_error(actual, predicted_actual)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted_actual) / (actual + 1e-8))) * 100
    accuracy = 100 - mape 

    logger.info(f"Evaluation Metrics - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, Accuracy: {accuracy:.2f}%")

    # --- Feature Importance (Random Forest) ---
    try:
        # Exclude the target column and any non-numeric columns added for plotting/analysis (like Year, Month, Day)
        features_for_rf = df.drop(columns=[TARGET_COLUMN, 'Year', 'Month', 'Day'], errors='ignore')
        
        # Ensure there are features left for Random Forest
        if not features_for_rf.empty:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features_for_rf, df[TARGET_COLUMN])
            importances = rf.feature_importances_
            
            importance_df = pd.DataFrame({'Feature': features_for_rf.columns, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=True)

            importance_fig, ax_importance = plt.subplots(figsize=(10, 6))
            ax_importance.barh(importance_df['Feature'], importance_df['Importance'])
            ax_importance.set_title("Feature Importances (Random Forest)", fontsize=16)
            ax_importance.set_xlabel("Importance", fontsize=12)
            ax_importance.set_ylabel("Feature", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "Feature_Importance_RF.png"))
            plt.close(importance_fig)
            logger.info("Feature importance plot saved.")
        else:
            logger.warning("No suitable features remaining for Random Forest Feature Importance calculation. Skipping plot.")
    except Exception as e:
        logger.error(f"Error generating Feature Importance plot: {e}")

    # --- Future Forecasting ---
    num_future_days = 15
    
    # Initialize the last sequence with the actual scaled data
    last_scaled_features_sequence = scaled_data[-sequence_length:].copy()
    future_predictions = []

    oil_idx = -1
    usd_idx = -1
    try:
        oil_idx = features.index('Oil_Rate')
    except ValueError:
        logger.warning("Column 'Oil_Rate' not found in features. Correlation logic for oil will be skipped in future forecast.")
    try:
        usd_idx = features.index('USD_Index_Rate')
    except ValueError:
        logger.warning("Column 'USD_Index_Rate' not found in features. Correlation logic for USD Index will be skipped in future forecast.")

    for _ in range(num_future_days):
        # Predict the next gold price using the current sequence
        next_scaled_gold = model.predict(last_scaled_features_sequence.reshape(1, sequence_length, len(features)), verbose=0)[0, 0]

        # Create a new row based on the last row of the sequence
        new_feature_row_scaled = last_scaled_features_sequence[-1].copy()

        # Update the gold price (target) in the new row
        new_feature_row_scaled[0] = next_scaled_gold

        # --- Calculate and Apply Correlation for Other Features ---
        dummy_last_gold = np.zeros((1, len(features)))
        dummy_last_gold[0, 0] = last_scaled_features_sequence[-1, 0]
        last_gold_price_actual = scaler.inverse_transform(dummy_last_gold)[0, 0]

        # Inverse transform the predicted gold price to its actual scale
        dummy_predicted_gold = np.zeros((1, len(features)))
        dummy_predicted_gold[0, 0] = next_scaled_gold
        predicted_gold_actual = scaler.inverse_transform(dummy_predicted_gold)[0, 0]

        # Calculate the percentage change in actual gold price
        gold_price_pct_change = 0
        if last_gold_price_actual != 0: # Avoid division by zero
            gold_price_pct_change = (predicted_gold_actual - last_gold_price_actual) / last_gold_price_actual

        # Oil Price (Directly Proportional)
        if oil_idx != -1:
            last_oil_scaled = last_scaled_features_sequence[-1, oil_idx]
            dummy_last_oil = np.zeros((1, len(features)))
            dummy_last_oil[0, oil_idx] = last_oil_scaled
            last_oil_actual = scaler.inverse_transform(dummy_last_oil)[0, oil_idx]

            projected_oil_actual = last_oil_actual * (1 + gold_price_pct_change) 
            
            dummy_projected_oil = np.zeros((1, len(features)))
            dummy_projected_oil[0, oil_idx] = projected_oil_actual
            new_feature_row_scaled[oil_idx] = scaler.transform(dummy_projected_oil)[0, oil_idx]

        # USD Index (Inversely Proportional)
        if usd_idx != -1:
            last_usd_scaled = last_scaled_features_sequence[-1, usd_idx]
            dummy_last_usd = np.zeros((1, len(features)))
            dummy_last_usd[0, usd_idx] = last_usd_scaled
            last_usd_actual = scaler.inverse_transform(dummy_last_usd)[0, usd_idx]

            projected_usd_actual = last_usd_actual * (1 - gold_price_pct_change) 
            
            # Scale the projected actual USD index back to the 0-1 range
            dummy_projected_usd = np.zeros((1, len(features)))
            dummy_projected_usd[0, usd_idx] = projected_usd_actual
            new_feature_row_scaled[usd_idx] = scaler.transform(dummy_projected_usd)[0, usd_idx]

        # Append the new row to the sequence, dropping the oldest point
        last_scaled_features_sequence = np.append(last_scaled_features_sequence[1:],
                                                 new_feature_row_scaled.reshape(1, len(features)),
                                                 axis=0)
        future_predictions.append(predicted_gold_actual)

    # Create DataFrame for future predictions
    future_dates = [df.index[-1] + timedelta(days=i + 1) for i in range(num_future_days)]
    future_df = pd.DataFrame({"Predicted Price": future_predictions}, index=future_dates)
    logger.info(f"Future predictions generated. First 5:\n{future_df.head()}")
    logger.info(f"Future predictions generated. Last 5:\n{future_df.tail()}")

    # --- Plotting Configuration ---
    plt.style.use('seaborn-v0_8-darkgrid') 
    formatter = mticker.FormatStrFormatter('$%.2f') 

    # --- Gold Price Forecast Plot (Training, Actual, Predicted, Future) ---
    logger.info("Generating main forecast plot...")
    train_plot = df[TARGET_COLUMN][:train_size]

    # Get the dates corresponding to the actual and predicted test values
    # Ensure this slicing aligns with how y_test was created from test_data
    test_start_date_idx = df.index.get_loc(df.index[train_size])
    test_dates_for_plot = df.index[test_start_date_idx : test_start_date_idx + len(actual)]

    valid = pd.DataFrame(index=test_dates_for_plot)
    valid['Actual'] = actual
    valid['Predicted'] = predicted_actual

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(train_plot.index, train_plot.values, label='Training Data', color='#1f77b4', linewidth=1.5)
    ax.plot(valid.index, valid['Actual'], label='Actual Prices (Test Set)', color='#ff7f0e', linewidth=2)
    ax.plot(valid.index, valid['Predicted'], label='Predicted Prices (Test Set)', color='#2ca02c', linestyle='--', linewidth=2)
    ax.plot(future_df.index, future_df['Predicted Price'], label='Future Forecast', color='#d62728', linestyle=':', linewidth=2.5) # Corrected linestyle

    ax.set_title("Gold Price Forecast with Historical Data and Future Projections", fontsize=20, fontweight='bold')
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Price (USD/Troy Ounce)", fontsize=16)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Dynamic date formatting for X-axis
    start_date = df.index.min()
    end_date = future_df.index.max()
    date_range_days = (end_date - start_date).days

    if date_range_days > 365 * 5: 
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2)) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator()) 
    elif date_range_days > 365 * 1: 
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator()) 
    elif date_range_days > 90: 
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    else: 
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

    fig.autofmt_xdate(rotation=45, ha='right') 
    ax.yaxis.set_major_formatter(formatter) 

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Gold_Price_Future_Prediction.png"), dpi=300)
    plt.close(fig)
    logger.info(f"Main forecast plot saved to {os.path.join(RESULTS_DIR, 'Gold_Price_Future_Prediction.png')}")

    # --- Save Model and Scaler ---
    model.save(os.path.join(MODELS_DIR, "gold_price_lstm_model.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    logger.info(f"Model saved to {os.path.join(MODELS_DIR, 'gold_price_lstm_model.keras')}")
    logger.info(f"Scaler saved to {os.path.join(MODELS_DIR, 'scaler.joblib')}")

    # --- Write Prediction Report ---
    with open(os.path.join(RESULTS_DIR, "Prediction_Report.txt"), 'w') as f:
        f.write("---- Gold Price Prediction Report ----\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training size: {train_size} data points\n")
        f.write(f"Testing size: {len(actual)} data points\n")
        f.write("\n--- Evaluation Metrics ---\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"\n--- Future {num_future_days}-day Forecast ---\n")
        f.write(future_df.to_string()) # Use to_string() to write entire DataFrame
        f.write("\n\n--- Historical Data Visualizations ---\n")
        f.write("• Yearly trend graphs are saved for each individual year showing daily movements.\n")
        f.write("• A combined summary of average monthly prices per year is plotted.\n")
        f.write("• Daily trends across all years are visualized for average price changes per day of the month.\n")
        f.write("• Year-over-year percentage change in average price is shown as a bar chart.\n")
        f.write("\nThese plots offer insights into seasonal trends, annual volatility, and long-term gold price evolution.\n")
    logger.info(f"Prediction report saved to {os.path.join(RESULTS_DIR, 'Prediction_Report.txt')}")

    # --- Historical Data Visualizations ---
    logger.info("Generating historical data visualizations...")
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    years = df['Year'].unique()

    # Create subplots for yearly trends
    num_years = len(years)
    cols = 3 
    rows = (num_years + cols - 1) // cols 
    fig_yearly, axes_yearly = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False, sharey=True)
    axes_yearly = axes_yearly.flatten() # Flatten for easy iteration

    for i, year in enumerate(years):
        if i >= len(axes_yearly): # Ensure we don't try to access out of bounds if rows * cols is less than num_years
            break
        ax = axes_yearly[i]
        yearly_data = df[df['Year'] == year][TARGET_COLUMN]

        if not yearly_data.empty: 
            ax.plot(yearly_data.index, yearly_data.values, label=f"{year} Trend", color='#3182bd', linewidth=1.5)
            ax.set_title(f"Gold Price Trend - {year}", fontsize=12)
            ax.set_ylabel("Price ($)", fontsize=10)

            # Adjust x-axis formatting based on data density
            if len(yearly_data) > 1:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) 
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7)) 
            else:
                 ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                 ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

            ax.yaxis.set_major_formatter(formatter)
            ax.grid(True, linestyle=':', alpha=0.6)
            fig_yearly.autofmt_xdate(rotation=45, ha='right')
        else:
            ax.set_visible(False) # Hide empty subplots

    # Turn off any remaining unused subplots
    for j in range(num_years, len(axes_yearly)):
        fig_yearly.delaxes(axes_yearly[j])

    fig_yearly.suptitle("Gold Price Trends by Year", fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(HISTORY_DIR, "Yearly_Trends_Summary.png"), dpi=300)
    plt.close(fig_yearly)
    logger.info("Yearly trend plots saved.")


    # --- Monthly Average Plot ---
    fig_monthly, ax_monthly = plt.subplots(figsize=(14, 8))
    monthly_avg = df.groupby(['Year', 'Month'])[TARGET_COLUMN].mean().unstack()
    if not monthly_avg.empty:
        monthly_avg.T.plot(ax=ax_monthly, marker='o', linewidth=1.5) # Transpose to have month on x-axis
        ax_monthly.set_title('Average Monthly Gold Prices by Year', fontsize=18, fontweight='bold')
        ax_monthly.set_xlabel('Month', fontsize=14)
        ax_monthly.set_ylabel('Average Price (USD/Troy Ounce)', fontsize=14)
        ax_monthly.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize='13')
        ax_monthly.set_xticks(range(1, 13))
        ax_monthly.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax_monthly.yaxis.set_major_formatter(formatter)
        ax_monthly.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "Monthly_Trend_Summary.png"), dpi=300)
        plt.close(fig_monthly)
        logger.info("Monthly trend summary plot saved.")
    else:
        logger.warning("Monthly average data is empty. Skipping Monthly Trend Summary plot.")
        plt.close(fig_monthly) 

    # --- Daily Average Plot (Across all years for each day of month) ---
    fig_daily, ax_daily = plt.subplots(figsize=(16, 8))
    daily_avg_overall = df.groupby(df.index.day)[TARGET_COLUMN].mean()
    if not daily_avg_overall.empty:
        ax_daily.plot(daily_avg_overall.index, daily_avg_overall.values, marker='s', linestyle='-', color='purple', linewidth=2)
        ax_daily.set_title('Average Gold Price by Day of Month (Overall)', fontsize=18, fontweight='bold')
        ax_daily.set_xlabel('Day of Month', fontsize=14)
        ax_daily.set_ylabel('Average Price (USD/Troy Ounce)', fontsize=14)
        ax_daily.set_xticks(range(1, 32)) # Set ticks for all 31 days
        ax_daily.yaxis.set_major_formatter(formatter)
        ax_daily.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "Overall_Daily_Trend_Summary.png"), dpi=300)
        plt.close(fig_daily)
        logger.info("Overall Daily trend summary plot saved.")
    else:
        logger.warning("Daily average data is empty. Skipping Overall Daily Trend Summary plot.")
        plt.close(fig_daily)

    # --- Year-over-Year Percentage Change Bar Chart ---
    fig_pct, ax_pct = plt.subplots(figsize=(16, 8))
    pct_change = df[TARGET_COLUMN].resample('Y').mean().pct_change() * 100
    pct_change = pct_change.dropna() 

    if not pct_change.empty:
        pct_change.index = pct_change.index.year # Use year for plotting

        colors = ['green' if x >= 0 else 'red' for x in pct_change.values] # Color bars based on positive/negative change
        ax_pct.bar(pct_change.index.astype(str), pct_change.values, color=colors, alpha=0.8)
        ax_pct.axhline(0, color='black', linewidth=0.8, linestyle='--') # Add a horizontal line at 0 for reference
        ax_pct.set_title('Year-over-Year % Change in Average Gold Price', fontsize=18, fontweight='bold')
        ax_pct.set_xlabel('Year', fontsize=14)
        ax_pct.set_ylabel('% Change', fontsize=14)
        ax_pct.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability
        ax_pct.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "Yearly_Percentage_Change.png"), dpi=300)
        plt.close(fig_pct)
        logger.info("Yearly percentage change plot saved.")
    else:
        logger.warning("Year-over-year percentage change data is empty. Skipping plot.")
        plt.close(fig_pct)

    logger.info("Gold price forecasting process complete. All results and plots saved.")