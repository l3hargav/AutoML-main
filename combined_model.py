import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.tsa.stattools import adfuller, acf, kpss
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer
#import xgboost as xgb
import concurrent.futures
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import time
from datetime import timedelta
# from multiprocessing import Process, Manager, cpu_count
import multiprocessing as mp
from functools import partial
from prophet import Prophet
from colorama import init as colorama_init
from colorama import Fore, Back
from colorama import Style
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from scipy.stats import entropy
from scipy.signal import find_peaks
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pywt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import pickle
import statsmodels.api as sm
import itertools
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from model import LSTMClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("/home/bhargav/Documents/projects/AutoML/saved_models/ensemble_model.pkl", 'rb') as f:
    classifier = pickle.load(f)

with open("/home/bhargav/Documents/projects/AutoML/saved_models/nn_model.pth", 'rb') as f:
    nn_model = LSTMClassifier(input_size=7, hidden_size=128, num_classes=3).to(device)
    nn_model.load_state_dict(torch.load(f))

model = ["ARIMA", "HWES", "PROPHET"]


# TODO
# Figure out best datasets for plot and plot a few graphs

def prepare_data(df):
    """
    Prepare data by selecting only numeric columns and handling missing values.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Cleaned DataFrame with only numeric columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    df_numeric = df_numeric.ffill().bfill()

    return df_numeric

# Preprocess the data
def preprocess_input(x):
    # Return preprocessed input
    x = x.drop(['Volume', 'High', 'Low', 'Open', 'Name'], axis=1)
    x['Date'] = pd.to_datetime(x['Date'])
    x.set_index('Date', inplace=True)
    x = x.asfreq('D', method="ffill")
    x = prepare_data(x)
    return x


def preprocess_series(series):
    log_series = np.log(series)
    log_returns = log_series.diff().dropna()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(log_returns.values.reshape(-1, 1)).flatten()
    scaled_series = pd.Series(scaled_values, index=log_returns.index)

    return scaled_series

def compute_rs_rsi(window):
    df = pd.DataFrame(window)
    window_length = 5
    df['diff'] = df.diff(axis=0, periods=1)
    df['gain'] = df['diff'].clip(lower=0).round(2)
    df['loss'] = df['diff'].clip(upper=0).abs().round(2)

    # Initialize avg_gain and avg_loss columns
    df['avg_gain'] = np.nan
    df['avg_loss'] = np.nan

    # Calculate initial averages
    df.loc[window_length-1, 'avg_gain'] = df['gain'].iloc[1:window_length].mean()
    df.loc[window_length-1, 'avg_loss'] = df['loss'].iloc[1:window_length].mean()

    # Get column indices
    avg_gain_col = df.columns.get_loc('avg_gain')
    avg_loss_col = df.columns.get_loc('avg_loss')
    gain_col = df.columns.get_loc('gain')
    loss_col = df.columns.get_loc('loss')

    # Update averages using a single loop with .iat
    for i in range(window_length, len(df)):
        if i < window_length:
            continue  # Skip initial period

        # Update avg_gain
        prev_avg_gain = df.iat[i-1, avg_gain_col]
        current_gain = df.iat[i, gain_col]
        new_avg_gain = (prev_avg_gain * (window_length - 1) + current_gain) / window_length
        df.iat[i, avg_gain_col] = new_avg_gain

        # Update avg_loss
        prev_avg_loss = df.iat[i-1, avg_loss_col]
        current_loss = df.iat[i, loss_col]
        new_avg_loss = (prev_avg_loss * (window_length - 1) + current_loss) / window_length
        df.iat[i, avg_loss_col] = new_avg_loss

    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rs'] = df['rs'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['rsi'] = 100 - (100 / (1.0 + df['rs']))

    return df["rs"].tolist(), df["rsi"].tolist()

# Feature extraction function
def extract_features(window, compute_for_transformed=True):
    features = {}
    if compute_for_transformed:
        features['std'] = np.std(window)
        features['skewness'] = skew(window)
        features['kurtosis'] = kurtosis(window)
        features['first_diff'] = window.iloc[-1] - window.iloc[-2] if len(window) > 1 else 0
        features['rate_of_change'] = (window.iloc[-1] - window.iloc[0]) / window.iloc[0] if window.iloc[0] != 0 else 0
        features['rolling_std_7'] = window.rolling(window=7).std().iloc[-1]
        features['rolling_std_30'] = window.rolling(window=30).std().iloc[-1]
        features['volatility_7'] = window.rolling(window=7).std().mean() if len(window) >= 7 else 0
        features['volatility_30'] = window.rolling(window=30).std().mean() if len(window) >= 30 else 0
        fft_values = np.abs(fft(window))
        features['fft_1'] = fft_values[1] if len(fft_values) > 1 else 0
        features['fft_2'] = fft_values[2] if len(fft_values) > 2 else 0
        rs_list, rsi_list = compute_rs_rsi(window)
        for i, rs_value in enumerate(rs_list):
            features[f'rs_{i}'] = rs_value
        for i, rsi_value in enumerate(rsi_list):
            features[f'rsi_{i}'] = rsi_value
        features['entropy'] = entropy(window.value_counts(normalize=True))
        coeffs = pywt.wavedec(window, 'db1', level=2)
        features['wavelet_energy_level_1'] = np.sum(np.square(coeffs[0]))
        features['wavelet_energy_level_2'] = np.sum(np.square(coeffs[1]))
        try:
            acf_values = acf(window, nlags=5)
            for i in range(1, min(6, len(acf_values))):
                features[f'acf_lag_{i}'] = acf_values[i]
        except:
            for i in range(1, 6):
                features[f'acf_lag_{i}'] = 0
        try:
            decomposition = seasonal_decompose(window, period=7, model='additive', extrapolate_trend='freq')
            features['seasonal_strength'] = np.std(decomposition.seasonal)
            features['residual_std'] = np.std(decomposition.resid)
        except Exception as e:
            features['seasonal_strength'] = 0
            features['residual_std'] = 0

    else:
        # Basic Statistical Features
        features['mean'] = np.mean(window)
        features['max'] = np.max(window)
        features['min'] = np.min(window)
        features['range'] = features['max'] - features['min']

        # Differential Features
        features['first_diff'] = window.iloc[-1] - window.iloc[-2] if len(window) > 1 else 0
        features['rate_of_change'] = (window.iloc[-1] - window.iloc[0]) / window.iloc[0] if window.iloc[0] != 0 else 0

        # Rolling Statistics (using window sizes 7 and 30)
        features['rolling_mean_7'] = window.rolling(window=7).mean().iloc[-1]
        features['rolling_mean_30'] = window.rolling(window=30).mean().iloc[-1]

        # Momentum and Volatility
        features['momentum_5'] = window.iloc[-1] - window.iloc[-5] if len(window) >= 5 else 0
        features['momentum_10'] = window.iloc[-1] - window.iloc[-10] if len(window) >= 10 else 0

        # Trend Features
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]
        features['slope'] = slope
        poly_fit = np.polyfit(x, window, 2)
        features['trend_quadratic'] = poly_fit[0]
        features['trend_linear'] = poly_fit[1]

        # Stationarity Test Features
        adf_result = adfuller(window)
        features['adf_stat'] = adf_result[0]
        features['adf_pvalue'] = adf_result[1]

        # Peak Features
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(window)
        features['num_peaks'] = len(peaks)
        features['avg_peak_height'] = np.mean(window.iloc[peaks]) if len(peaks) > 0 else 0

        # Lag Features
        features['lag_1'] = window.iloc[-2] if len(window) >= 2 else 0
        features['lag_2'] = window.iloc[-3] if len(window) >= 3 else 0
        features['lag_3'] = window.iloc[-4] if len(window) >= 4 else 0

        # Exponential Moving Averages
        features['ema_7'] = window.ewm(span=7, adjust=False).mean().iloc[-1]
        features['ema_30'] = window.ewm(span=30, adjust=False).mean().iloc[-1]

        # Date-based Features (if available)
        if hasattr(window, 'index') and isinstance(window.index, pd.DatetimeIndex):
            features['month'] = window.index[-1].month
            features['day_of_week'] = window.index[-1].dayofweek

        # Recent Value Features
        features['last_value'] = window.iloc[-1]
        features['second_last_value'] = window.iloc[-2] if len(window) >= 2 else window.iloc[-1]
        features['third_last_value'] = window.iloc[-3] if len(window) >= 3 else window.iloc[-1]

    return features

def transform_window(window):
    # Ensure values are positive (for log transformation)
    window = window.apply(lambda x: x if x > 1e-10 else 1e-10)
    log_window = np.log(window)
    log_returns = log_window.diff().dropna()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(log_returns.values.reshape(-1, 1)).flatten()
    transformed_series = pd.Series(scaled_values, index=log_returns.index)
    return transformed_series

def extract_features_hybrid(window):
    # Extract features from the raw window using your existing function.
    raw_features = extract_features(window, compute_for_transformed=False)
    transformed_window = transform_window(window)
    trans_features = extract_features(transformed_window, compute_for_transformed=True)

    combined_features = {}
    for key, value in raw_features.items():
        combined_features[f"raw_{key}"] = value
    for key, value in trans_features.items():
        combined_features[f"trans_{key}"] = value

    return combined_features

# Return the best model given some input dataset
def combined_predictor(classifier, nn_model, features, nn_input):
    # Combine the predictions
    features = pd.DataFrame([features])
    features.insert(0, "Unnamed: 0", range(len(features)))
    nn_inputs_last = nn_input[-1]
    lengths = torch.tensor([nn_inputs_last.shape[0]], dtype=torch.int64)
    nn_proba = get_nn_predictions(nn_model, nn_inputs_last.unsqueeze(0), lengths)
    # features = pd.DataFrame(features[-1])
    # print(window)
    print(features)
    vc_proba = classifier.predict_proba(features)

    combined_proba = 0.6*vc_proba + 0.4*nn_proba  # Adjust weights based on model performance
    y_pred_combined = np.argmax(combined_proba, axis=1)
    return model[y_pred_combined[0]]

# Get predictions given the best model and input
def get_preds(model, x):
    if model == "ARIMA":
        p = range(0, 4)
        d = range(0, 3)
        q = range(0, 4)

        pdq = list(itertools.product(p, d, q))

        best_aic = float("inf")
        best_order = None
        best_model = None
        # try:
        for order in pdq:
            try:
                model = ARIMA(
                    x,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_model = results
            except Exception:
                if best_model == None:
                    best_model = ARIMA(x, order=(1,1,1)).fit()
                continue
        if best_model == None:
            best_model = ARIMA(x, order=(1,1,1)).fit()
        pred_arima = best_model.forecast(steps=90)
        return pred_arima

    elif model == "HWES":
        model_hwes = ExponentialSmoothing(x,
                                         seasonal='add',
                                         seasonal_periods=5)
        model_hwes_fit = model_hwes.fit(optimized=True)
        pred_hwes = model_hwes_fit.forecast(90)
        return pred_hwes

    elif model == "PROPHET":
        x = x["Close"]
        prophet_df = pd.DataFrame({'ds': x.index, 'y': x.values})
        prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=False
        ).add_seasonality(name="stock_weekly", period=3, fourier_order=4)
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=90)
        forecast = prophet_model.predict(future)
        pred_prophet = forecast['yhat'].iloc[-90:].values
        last_date = x.index[-1]
        pred_prophet = pred_prophet.flatten()
        forecast_horizon = len(pred_prophet)
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        print(pred_prophet)
        forecast_series = pd.Series(pred_prophet, index=forecast_dates)
        return forecast_series

    else:
        raise ValueError("Invalid model")

def encode_dates(df):
    """
    Encode dates using multiple methods for neural network input.

    Parameters:
    df: DataFrame with DateTimeIndex and values

    Returns:
    DataFrame with encoded date features and original values
    """
    # Convert string dates to datetime if needed
    if isinstance(df.index, str):
        df.index = pd.to_datetime(df.index)

    # Create a new DataFrame with the original values
    encoded_df = pd.DataFrame(df.values, columns=['value'], index=df.index)

    # Method 1: Cyclical encoding for day of week
    encoded_df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
    encoded_df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)

    # Method 2: Cyclical encoding for month
    encoded_df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
    encoded_df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)

    # Method 3: Normalized day of year
    scaler = MinMaxScaler()
    encoded_df['day_of_year'] = scaler.fit_transform(df.index.dayofyear.values.reshape(-1, 1))

    # Method 4: Time elapsed since start (normalized)
    days_elapsed = (df.index - df.index.min()).days
    encoded_df['time_elapsed'] = scaler.fit_transform(np.array(days_elapsed).reshape(-1, 1))

    return encoded_df

def create_dynamic_windows(series, min_window=50, max_window=200, step=25, volatility_threshold=0.1):
    """
    Create dynamic windows based on local volatility.

    Parameters:
    - series: Time series data (pd.Series).
    - min_window: Minimum window size.
    - max_window: Maximum window size.
    - step: Step size for moving the window.
    - volatility_threshold: Threshold for determining high volatility.

    Returns:
    - windows: List of windows (pd.Series).
    """
    windows = []
    i = 0
    while i < len(series) - min_window + 1:
        # Calculate volatility in the current region
        current_volatility = series.iloc[i:i + min_window].std()

        # Adjust window size based on volatility
        if current_volatility > volatility_threshold:
            window_size = min_window  # Smaller window for high volatility
        else:
            window_size = max_window  # Larger window for low volatility

        # Ensure the window does not exceed the series length
        window_size = min(window_size, len(series) - i)

        # Extract the window
        window = series.iloc[i:i + window_size]
        windows.append(window)

        # Move the window by the step size
        i += step

    return windows

def generate_windows_and_nn_inputs(data_series, temp):
    windows = create_dynamic_windows(data_series)
    nn_inputs = [temp.loc[window.index].values for window in windows]
    return windows, nn_inputs

# def get_nn_predictions(model, input, length):
#     nn_model.eval()
#     all_probas = []
#     with torch.inference_mode():
#         input, length = input.to(device), length.to(device)
#         outputs = nn_model(input, length)
#         probas = torch.softmax(outputs, dim=1).cpu()
#         all_probas.append(probas.numpy())
#     return np.concatenate(all_probas, axis=0)
def get_nn_predictions(model, inputs, lengths):
    model.eval()
    all_probas = []

    # Ensure inputs is a tensor
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)

    # Ensure inputs is on the correct device
    inputs = inputs.to(device)

    # Prepare lengths tensor correctly
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64)

    # Ensure lengths is 1D and on CPU
    lengths = lengths.cpu().view(-1).long()

    # Sort inputs and lengths in descending order of length
    sorted_lengths, sorted_indices = lengths.sort(0, descending=True)
    sorted_inputs = inputs[sorted_indices]

    with torch.inference_mode():
        # Pack the padded sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)

        # Run the model
        packed_out, _ = model.lstm(packed_inputs)

        # Unpack the sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = out[torch.arange(out.size(0)), sorted_lengths - 1]
        outputs = model.fc(out)

        probas = torch.softmax(outputs, dim=1).cpu()
        all_probas.append(probas.numpy())

    return np.concatenate(all_probas, axis=0)

if __name__ == "__main__":
    data = pd.read_csv('XOM.csv')
    data = preprocess_input(data)
    data_series = data["Close"]
    data_series = preprocess_series(data_series)
    temp = pd.DataFrame(data_series)
    temp = encode_dates(temp)
    windows, nn_inputs =  generate_windows_and_nn_inputs(data_series, temp)
    nn_inputs = torch.tensor(nn_inputs, dtype=torch.float32)
    print(windows[-1], nn_inputs[-1])

    print(combined_predictor(classifier, nn_model, extract_features_hybrid(windows[-1]), nn_inputs))

    predicted_model = combined_predictor(classifier, nn_model, extract_features_hybrid(windows[-1]), nn_inputs)

    forecast = get_preds(predicted_model, data)

    print(forecast)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Price', color='blue')
    plt.plot(forecast.index, forecast.values, marker='o', linestyle='--', color='red', label='Forecasted Price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Historical vs Forecasted Stock Prices")
    plt.legend()
    plt.grid(True)
    plt.show()
