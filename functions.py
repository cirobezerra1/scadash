import numpy as np
import pandas as pd
import pandas as pd
import streamlit as st


@st.cache_data
def windrose_histogram(wspd, wdir, speed_bins=12, normed=False, norm_axis=None):
    """
    Compute a windrose histogram given wind speed and direction data.

    wspd: array of wind speeds
    wdir: array of wind directions
    speed_bins: Integer or Sequence, datafrdefines the bin edges for the wind speed (default is 12 equally spaced bins)
    normed: Boolean, optional, whether to normalize the histogram values. (default is False)
    norm_axis: Integer, optional, the axis along which the histograms are normalized (default is None)
    """

    # If speed_bins is an integer, we create linearly spaced bins from 0 to max speed
    if isinstance(speed_bins, int):
        speed_bins = np.linspace(0, wspd.max(), speed_bins)

    num_spd = len(speed_bins)
    num_angle = 16

    # Shift wind directions by 11.25 degrees (one sector) to ensure proper alignment
    wdir_shifted = (wdir + 11.25) % 360

    angle_bins = np.linspace(0, 360, num_angle + 1)

    # Generate a 2D histogram using the defined speeds bins and shifted wind directions
    hist, *_ = np.histogram2d(wspd, wdir_shifted, bins=(speed_bins, angle_bins))

    # Normalize if required
    if normed:
        hist /= hist.sum(axis=norm_axis, keepdims=True)
        hist *= 100

    return hist, angle_bins, speed_bins


DIRECTION_NAMES = ("N","NNE","NE","ENE"
                   ,"E","ESE","SE","SSE"
                   ,"S","SSW","SW","WSW"
                   ,"W","WNW","NW","NNW")

DIRECTION_ANGLES = np.arange(0, 2*np.pi, 2*np.pi/16)

# Mapping from direction name to angles in radians
NAME2ANGLE = dict(zip(
    DIRECTION_NAMES,
    DIRECTION_ANGLES
))

@st.cache_data
def make_wind_df(data_df,SPD,DIR,num_partitions = 11, max_speed=None, normed=False, norm_axis=None, month=None):
    """
    This function transforms raw wind speed and direction data into a DataFrame for windrose plotting.

    data_df: Dataframe containing wind data
    num_partitions: Integer, number of partitions to divide the wind speed data
    max_speed: Float, optional, maximum wind speed to be included in the partitions
    normed: Boolean, optional, whether to normalize the frequency values
    norm_axis: Integer, optional, the axis along which the histograms are normalized
    """

    if month is not None:
        data_df = data_df[data_df.MM==month]

    wspd = data_df[SPD].values
    wdir = data_df[DIR].values

    # If max_speed is not specified, we use the maximum value in the 'wspd' data.
    # Otherwise, we include all speeds up to and including max_speed.
    # Additional partitions are created to handle outliers.
    if max_speed is None:
        speed_bins = np.linspace(0, max(wspd), num_partitions + 1)
    else:
        speed_bins = np.append(np.linspace(0, max_speed, num_partitions + 1), np.inf)

    # windrose_histogram function is called to partition data based on the bins created.
    # Additional parameters control how the frequency values are normalised

    h, *_ = windrose_histogram(wspd, wdir, speed_bins, normed=normed, norm_axis=norm_axis)

    # A dataframe is formed containing the histogram data. Column names are for the directions
    wind_df = pd.DataFrame(data=h, columns=DIRECTION_NAMES)

    # speed_bin_names stores speed range strings to describe each interval
    # e.g. when speed_bins is [0, 3, 6], speed_bin_names is ['0-3', '3-6', '>6']
    speed_bin_names = []
    speed_bins_rounded = [round(i, 2) for i in speed_bins]
    for start, end in zip(speed_bins_rounded[:-1], speed_bins_rounded[1:]):
        speed_bin_names.append(f'{start:g}-{end:g}' if end < np.inf else f'>{start:g}')

    wind_df['strength'] = speed_bin_names

    # Reshapes data for plotting. Now, each row represents one sector of the windrose
    wind_df = wind_df.melt(id_vars=['strength'], var_name='direction', value_name='frequency')

    return wind_df

@st.cache_data
def extract_features(dataframe, id_col, val_col):
    """
    Extract selected features from a multivariate time series DataFrame using tsfresh.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame with multivariate time series data.
    id_col (str): The name of the column in the DataFrame that contains the time series ID.
    time_col (str): The name of the column in the DataFrame that contains the time index.

    Returns:
    pd.DataFrame: A DataFrame with extracted features.
    """
    # Ensure the ID and time columns are in the correct format
    dataframe[id_col] = dataframe[id_col].astype(str)

    # List of features to be extracted
    #features_list =  ['absolute_sum_of_changes', 'abs_energy', 'absolute_maximum', 'kurtosis', 'skewness', 'sample_entropy', 'standard_deviation', 'mean_second_derivative_central']
    features_list =  ['absolute_sum_of_changes', 'abs_energy', 'absolute_maximum', 'kurtosis', 'skewness', 'standard_deviation', 'mean_second_derivative_central']
    # Extract features
    extracted_features = pd.DataFrame(index = dataframe[id_col].unique(), columns = features_list)

    for i in extracted_features.index:
        series = dataframe[dataframe[id_col] == i][val_col]
        series = series.dropna().values
        extracted_features.loc[i,'absolute_sum_of_changes'] = absolute_sum_of_changes(series)
        extracted_features.loc[i,'abs_energy'] = abs_energy(series)
        extracted_features.loc[i,'absolute_maximum'] = absolute_maximum(series)
        extracted_features.loc[i,'kurtosis'] = kurtosis(series)
        extracted_features.loc[i,'skewness'] = skewness(series)
        #extracted_features.loc[i,'sample_entropy'] = sample_entropy(series)
        extracted_features.loc[i,'standard_deviation'] = standard_deviation(series)
        extracted_features.loc[i,'mean_second_derivative_central'] = mean_second_derivative_central(series)

    return extracted_features

@st.cache_data
def _into_subchunks(x, subchunk_length, every_n=1):
    """
    Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

    For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

        0  2  4
        1  3  5
        2  4  6

    with the settings subchunk_length = 3 and every_n = 2
    """
    len_x = len(x)

    assert subchunk_length > 1
    assert every_n > 0

    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return np.asarray(x)[indexer]

@st.cache_data
def absolute_sum_of_changes(x):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.sum(np.abs(np.diff(x)))
@st.cache_data
def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)

@st.cache_data
def absolute_maximum(x):
    """
    Calculates the highest absolute value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN
@st.cache_data
def kurtosis(x):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)
@st.cache_data
def skewness(x):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)
@st.cache_data
def sample_entropy(x):
    """
    Calculate and return sample entropy of x.

    .. rubric:: References

    |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
    |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    x = np.array(x)

    # if one of the values is NaN, we can not compute anything meaningful
    if np.isnan(x).any():
        return np.nan

    m = 2  # common value for m, according to wikipedia...
    tolerance = 0.2 * np.std(
        x
    )  # 0.2 is a common value for r, according to wikipedia...

    # Split time series and save all templates of length m
    # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
    xm = _into_subchunks(x, m)

    # Now calculate the maximum distance between each of those pairs
    #   np.abs(xmi - xm).max(axis=1)
    # and check how many are below the tolerance.
    # For speed reasons, we are not doing this in a nested for loop,
    # but with numpy magic.
    # Example:
    # if x = [1, 2, 3]
    # then xm = [[1, 2], [2, 3]]
    # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
    # and from [2, 3] => [[1, 1], [0, 0]]
    # taking the abs and max gives us:
    # [0, 1] and [1, 0]
    # as the diagonal elements are always 0, we substract 1.
    B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

    # Similar for computing A
    xmp1 = _into_subchunks(x, m + 1)

    A = np.sum(
        [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
    )

    # Return SampEn
    return -np.log(A / B)
@st.cache_data
def standard_deviation(x):
    """
    Returns the standard deviation of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.std(x)
@st.cache_data
def mean_second_derivative_central(x):
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN
@st.cache_data
def calculate_wf_metrics(df, IDNode, pwr, nws, TS, wind_farm_capacity):
    """

    """
    CF = (df.groupby(TS)[pwr].sum()/wind_farm_capacity).mean()

    nWTGs = len(df[IDNode].unique())

    return CF,nWTGs
