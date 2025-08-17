import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.tsa.stattools import adfuller, acf, kpss
from matplotlib import pyplot
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

device = "cuda" if torch.cuda.is_available() else "cpu"

simplefilter("ignore", category=ConvergenceWarning)

start_total = time.time()

# data_diff = data_series.diff().dropna()
# result = adfuller(data_diff)
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# if result[1] > 0.05:
#     print("Data is likely non-stationary; consider differencing or transformations.")



# Load the dataset and preprocess data
def preprocess_data(data):

    data = data.drop(['Volume', 'High', 'Low', 'Open', 'Name'], axis=1)

    # window_size = 5
    # std_dev_threshold = 3

    # # Calculate rolling mean and standard deviation
    # rolling_mean = data['Close'].rolling(window=window_size, center=True).mean()
    # rolling_std = data['Close'].rolling(window=window_size, center=True).std()

    # # Detect outliers
    # outliers = (data['Close'] - rolling_mean).abs() > std_dev_threshold * rolling_std

    # # Replace outliers with interpolated values
    # data['value_corrected'] = data['Close'].copy()
    # data.loc[outliers, 'value_corrected'] = np.nan  # Set outliers to NaN
    # data['Close'] = data['value_corrected'].interpolate()
    # data = data.drop('value_corrected', axis=1)

    # # Convert the date column to datetime and set it as the index
    # data['Close'] = data['Close'].apply(lambda x: x if x > 1e-10 else 1e-10)
    # data = data[data['Close'] > 0]
    # data.plot()
    # pyplot.show()
    print("NaN values:", data['Close'].isna().sum())
    print("Infinite values:", np.isinf(data['Close']).sum())
    # data['Close'] = np.log(data['Close'] + 1).diff().dropna()


    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D', method="ffill")
    data = prepare_data(data)
    # data.plot()
    # pyplot.show()
    return data

def preprocess_series(series):

    log_series = np.log(series)

    # 2. Compute log returns (first difference of log prices)
    log_returns = log_series.diff().dropna()

    # 3. Scale the log returns with StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(log_returns.values.reshape(-1, 1)).flatten()

    # 4. Create a new Series with the scaled values (preserving the original index)
    scaled_series = pd.Series(scaled_values, index=log_returns.index)

    return scaled_series

# result = adfuller(data_transformed_series)

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

def prepare_data(df):
    """
    Prepare data by selecting only numeric columns and handling missing values.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Cleaned DataFrame with only numeric columns
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # Handle any missing values
    df_numeric = df_numeric.ffill().bfill()

    return df_numeric

def run_stationarity_tests(data, column_name='Close'):
    """
    Run ADF and KPSS tests on the time series data.

    Parameters:
    data (pd.Series or np.array): Time series data
    column_name (str): Name of the column being tested
    """
    # Ensure data is numeric and handle NaN values
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).iloc[:, 0]
    data = pd.to_numeric(data, errors='coerce')
    data = data.dropna()

    # ADF Test
    adf_result = adfuller(data)
    print(f'\nAugmented Dickey-Fuller Test Results for {column_name}:')
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')

    # KPSS Test
    kpss_result = kpss(data)
    print(f'\nKPSS Test Results for {column_name}:')
    print(f'KPSS Statistic: {kpss_result[0]:.4f}')
    print(f'p-value: {kpss_result[1]:.4f}')
    print('Critical values:')
    for key, value in kpss_result[3].items():
        print(f'\t{key}: {value:.4f}')



# Function to create rolling windows from the time series
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

# NEW
def create_fixed_windows(df, window_size=100):
    """Create fixed-length windows with padding/truncating"""
    windows = []
    n = len(df)
    for i in range(0, n, window_size//2):  # 50% overlap
        end = i + window_size
        if end > n:
            # Pad with last value
            window = df.iloc[i:n]
            pad_values = pd.DataFrame([df.iloc[-1]]*(end-n))
            window = pd.concat([window, pad_values], ignore_index=False)
        else:
            window = df.iloc[i:end]
        windows.append(window)
    return windows

# NEW
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

# NEW
def generate_windows_and_nn_inputs(data_series, temp):
    windows = create_dynamic_windows(data_series)
    nn_inputs = [temp.loc[window.index].values for window in windows]
    return windows, nn_inputs

# NEW
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

# NEW
def compute_macd(window):
    df = pd.DataFrame(window)

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

# Apply forecasting models and get the best model based on a simple error measure (e.g., RMSE)
# For simplicity, using ARIMA and HWES as examples of models; more can be added


def evaluate_models(window):
    errors = {}
    predictions = {}
    actual_values = window[-5:]
    window = window[:-5]

    # ARIMA with optimized parameters
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
                window,
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
                best_model = ARIMA(window, order=(1,1,1)).fit()
            continue
    pred_arima = best_model.forecast(steps=5)
    error_arima = np.sqrt(mean_squared_error(actual_values, pred_arima))
    errors['ARIMA'] = error_arima
    print(f"{Fore.GREEN}ARIMA Done{Style.RESET_ALL}")
    # except:
    #     errors['ARIMA'] = float('inf')
    #     print(f"{Fore.RED}ERROR HAPPENED AT ARIMA{Style.RESET_ALL}")


    # try:
    #     model_arima = ARIMA(window, order=(1, 1, 1))
    #     model_arima_fit = model_arima.fit()
    #     pred_arima = model_arima_fit.forecast(steps=1)
    #     error_arima = np.sqrt(mean_squared_error(window[-1:], pred_arima))
    #     errors['ARIMA'] = error_arima
    #     print(f"{Fore.GREEN}ARIMA Done{Style.RESET_ALL}")
    # except:
    #     errors['ARIMA'] = float('inf')
    #     print(f"{Fore.RED}ERROR HAPPENED AT ARIMA{Style.RESET_ALL}")
    # HWES with optimized parameters
    try:
        model_hwes = ExponentialSmoothing(window,
                                         seasonal='add',
                                         seasonal_periods=5) # NEW
        model_hwes_fit = model_hwes.fit(optimized=True)
        pred_hwes = model_hwes_fit.forecast(5)
        error_hwes = np.sqrt(mean_squared_error(actual_values, pred_hwes))
        errors['HWES'] = error_hwes
        predictions['HWES'] = pred_hwes
        print(f"{Fore.GREEN}HWES Done{Style.RESET_ALL}")
    except:
        errors['HWES'] = float('inf')
        predictions['HWES'] = None
        print(f"{Fore.RED}ERROR HAPPENED AT HWES{Style.RESET_ALL}")

    # Prophet with enhanced configuration
    try:
        prophet_df = pd.DataFrame({'ds': window.index, 'y': window.values})
        prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=False
        ).add_seasonality(name="stock_weekly", period=3, fourier_order=4)
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=5)
        forecast = prophet_model.predict(future)
        pred_prophet = forecast['yhat'].iloc[-5:].values
        error_prophet = np.sqrt(mean_squared_error(actual_values, pred_prophet))
        errors['Prophet'] = error_prophet
        predictions['Prophet'] = pred_prophet
        print(f"{Fore.GREEN}PROPHET Done{Style.RESET_ALL}")
    except:
        errors['Prophet'] = float('inf')
        predictions['Prophet'] = None
        print(f"{Fore.RED}ERROR HAPPENED AT PROPHET{Style.RESET_ALL}")

    # Enhanced model selection logic
    valid_models = {k: v for k, v in errors.items() if v != float('inf')}
    if not valid_models:
        return 'ARIMA'  # Default to ARIMA if all models fail

    # Weight recent performance more heavily
    best_model = min(valid_models.items(), key=lambda x: x[1])[0]
    print(f"{Fore.GREEN}ARIMA error: {error_arima}, HWES error: {error_hwes}, PROPHET error: {error_prophet}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Best Model: {best_model}{Style.RESET_ALL}")
    return best_model

def process_window(window):
    features = extract_features(window)
    best_model = evaluate_models(window)
    return features, best_model

def process_window_hybrid(window):
    features = extract_features_hybrid(window)
    best_model = evaluate_models(window)  # You may choose to use the same evaluation on the raw window or combine evaluations.
    return features, best_model


def process_single_window(window_tuple):
    """
    Process a single window and return features and best model.
    Args:
        window_tuple: Tuple of (dataset_idx, window)
    """
    _, window = window_tuple  # Unpack the tuple
    features, best_model = process_window_hybrid(window)
    return features, best_model

def flatten_windows(window_lists):
    """Flatten all windows into a single list with their dataset indices."""
    flat_windows = []
    for dataset_idx, windows in enumerate(window_lists):
        for window in windows:
            flat_windows.append((dataset_idx, window))
    return flat_windows


preprocess_start = time.time()

data1 = pd.read_csv("CSCO.csv")
print(data1)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
data1 = preprocess_data(data1)
# print(data1)
print("///////////////////////////////////////////////////////////////////")
data_series1 = data1['Close']  # Adjust as necessary
print(data_series1)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
data_series1 = preprocess_series(data_series1)  # Apply transformations and differencing
temp1 = pd.DataFrame(data_series1)
print(temp1)
print("-----------------------------------------------------------------")
temp1 = encode_dates(temp1)



# Load and preprocess Dataset 2
data2 = pd.read_csv("AXP.csv")
data2 = preprocess_data(data2)  # Apply same preprocessing steps
data_series2 = data2['Close']  # Adjust as necessary
data_series2 = preprocess_series(data_series2)
temp2 = pd.DataFrame(data_series2)
temp2 = encode_dates(temp2)
print(temp2)


# Load and preprocess Dataset 3
data3 = pd.read_csv("GOOGLE.csv")
data3 = preprocess_data(data3)  # Assumes prepare_data does necessary preprocessing steps
data_series3 = data3['Close']  # Adjust as necessary
data_series3 = preprocess_series(data_series3)  # Apply transformations and differencing
temp3 = pd.DataFrame(data_series3)
temp3 = encode_dates(temp3)


# Load and preprocess Dataset 4
data4 = pd.read_csv("IBM.csv")
data4 = preprocess_data(data4)  # Apply same preprocessing steps
data_series4 = data4['Close']  # Adjust as necessary
data_series4 = preprocess_series(data_series4)
temp4 = pd.DataFrame(data_series4)
temp4 = encode_dates(temp4)


# Load and preprocess Dataset 5
data5 = pd.read_csv("MCD.csv")
data5 = preprocess_data(data5)  # Apply same preprocessing steps
data_series5 = data5['Close']  # Adjust as necessary
data_series5 = preprocess_series(data_series5)
temp5 = pd.DataFrame(data_series5)
temp5 = encode_dates(temp5)


# Load and preprocess Dataset 6
data6 = pd.read_csv("CAT.csv")
data6 = preprocess_data(data6)  # Apply same preprocessing steps
data_series6 = data6['Close']  # Adjust as necessary
data_series6 = preprocess_series(data_series6)
temp6 = pd.DataFrame(data_series6)
temp6 = encode_dates(temp6)


# Load and preprocess Dataset 7
data7 = pd.read_csv("BA.csv")
data7 = preprocess_data(data7)  # Apply same preprocessing steps
data_series7 = data7['Close']  # Adjust as necessary
data_series7 = preprocess_series(data_series7)
temp7 = pd.DataFrame(data_series7)
temp7 = encode_dates(temp7)


# Load and preprocess Dataset 8
data8 = pd.read_csv("AMZN.csv")
data8 = preprocess_data(data8)  # Apply same preprocessing steps
data_series8 = data8['Close']  # Adjust as necessary
data_series8 = preprocess_series(data_series8)
temp8 = pd.DataFrame(data_series8)
temp8 = encode_dates(temp8)


#Generate features and labels for each window
data9 = pd.read_csv("NKE.csv")
data9 = preprocess_data(data9)  # Apply same preprocessing steps
data_series9 = data9['Close']  # Adjust as necessary
data_series9 = preprocess_series(data_series9)
temp9 = pd.DataFrame(data_series9)
temp9 = encode_dates(temp9)


data10 = pd.read_csv("JPM.csv")
data10 = preprocess_data(data10)  # Apply same preprocessing steps
data_series10 = data10['Close']  # Adjust as necessary
data_series10 = preprocess_series(data_series10)
temp10 = pd.DataFrame(data_series10)
temp10 = encode_dates(temp10)


# print(windows)

preprocess_end = time.time()
elapsed_time_preprocess = preprocess_end - preprocess_start
formatted_prepro_time = str(timedelta(seconds=elapsed_time_preprocess))


#Generate features and labels for each window
feat_gen_start = time.time()
print(f"{Fore.YELLOW}Generating windows and Neural Network inputs{Style.RESET_ALL}")
windows1, nn_inputs1 =  generate_windows_and_nn_inputs(data_series1, temp1)
windows2, nn_inputs2 =  generate_windows_and_nn_inputs(data_series2, temp2)
windows3, nn_inputs3 =  generate_windows_and_nn_inputs(data_series3, temp3)
windows4, nn_inputs4 =  generate_windows_and_nn_inputs(data_series4, temp4)
windows5, nn_inputs5 =  generate_windows_and_nn_inputs(data_series5, temp5)
windows6, nn_inputs6 =  generate_windows_and_nn_inputs(data_series6, temp6)
windows7, nn_inputs7 =  generate_windows_and_nn_inputs(data_series7, temp7)
windows8, nn_inputs8 =  generate_windows_and_nn_inputs(data_series8, temp8)
windows9, nn_inputs9 =  generate_windows_and_nn_inputs(data_series9, temp9)
windows10, nn_inputs10 =  generate_windows_and_nn_inputs(data_series10, temp10)

nn_windows = nn_inputs1 + nn_inputs2 + nn_inputs3 + nn_inputs4 + nn_inputs5 + nn_inputs6 + nn_inputs7 + nn_inputs8 + nn_inputs9 + nn_inputs10

feat_gen_end = time.time()
elapsed_time_feat_gen = feat_gen_end - feat_gen_start
formatted_feat_gen_time = str(timedelta(seconds=elapsed_time_feat_gen))

training_data = []
# Feature extraction for all datasets
i = 1
print(f"{Fore.YELLOW}Extracting Features{Style.RESET_ALL}")
feat_extr_start = time.time()
training_data1, labels1 = [], []
for window in windows1:
    features, best_model = process_window_hybrid(window)
    training_data1.append(features)
    labels1.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data2, labels2 = [], []
for window in windows2:
    features, best_model = process_window_hybrid(window)
    training_data2.append(features)
    labels2.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1


training_data3, labels3 = [], []
for window in windows3:
    features, best_model = process_window_hybrid(window)
    training_data3.append(features)
    labels3.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data4, labels4 = [], []
for window in windows4:
    features, best_model = process_window_hybrid(window)
    training_data4.append(features)
    labels4.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data5, labels5 = [], []
for window in windows5:
    features, best_model = process_window_hybrid(window)
    training_data5.append(features)
    labels5.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data6, labels6 = [], []
for window in windows6:
    features, best_model = process_window_hybrid(window)
    training_data6.append(features)
    labels6.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1


training_data7, labels7 = [], []
for window in windows7:
    features, best_model = process_window_hybrid(window)
    training_data7.append(features)
    labels7.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data8, labels8 = [], []
for window in windows8:
    features, best_model = process_window_hybrid(window)
    training_data8.append(features)
    labels8.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data9, labels9 = [], []
for window in windows9:
    features, best_model = process_window_hybrid(window)
    training_data9.append(features)
    labels9.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done")
i = i + 1

training_data10, labels10 = [], []
for window in windows10:
    features, best_model = process_window_hybrid(window)
    training_data10.append(features)
    labels10.append(best_model)
print(f"{Fore.YELLOW}{Back.BLACK}Dataset {i} done{Style.RESET_ALL}")
i = i + 1

# windows_list = [
#     windows1, windows2, windows3, windows4, windows5,
#     windows6, windows7, windows8, windows9, windows10
# ]

# # Process all datasets
# training_data_list, labels_list = parallel_process_datasets(windows_list)

# # Unpack results if needed
# (training_data1, training_data2, training_data3, training_data4, training_data5,
#     training_data6, training_data7, training_data8, training_data9, training_data10) = training_data_list

# (labels1, labels2, labels3, labels4, labels5,
#     labels6, labels7, labels8, labels9, labels10) = labels_list

# all_windows = [windows1, windows2, windows3, windows4, windows5,
#                windows6, windows7, windows8, windows9, windows10]

# # Flatten all windows into a single list
# flat_windows = flatten_windows(all_windows)

# # Process all windows in parallel
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     results = pool.map(process_single_window, flat_windows)



feat_extr_end = time.time()
elapsed_time_feat_extr = feat_extr_end - feat_extr_start
formatted_feat_extr_time = str(timedelta(seconds=elapsed_time_feat_extr))

# data_log = np.log(data_series)
# data_log_diff = data_log.diff().dropna()

X = pd.DataFrame(training_data1 + training_data2 + training_data3 + training_data4 + training_data5 + training_data6 + training_data7 + training_data8 + training_data9 + training_data10)
y = labels1 + labels2 + labels3 + labels4 + labels5 + labels6 + labels7 + labels8 + labels9 + labels10

X["labels"] = y
X.to_csv("input.csv")
print(X.head)

with open("nn_input.pkl", "wb") as f:
    pickle.dump(nn_windows, f)

print(nn_windows[0].shape[-1])
print(len(nn_windows[0]))
print(X["labels"].value_counts())
