import logging
import math
import os
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser

from fpdf import FPDF, YPos, XPos
from aniso8601 import parse_datetime, parse_date
from datetime import datetime
from dateutil.relativedelta import relativedelta

import config
from py.config_parser import parse_app_properties
from py.common.minio_handler import ReportMinioStorage
from py.ref_constants import LEFT_MARGIN_COLOR, RIGHT_MARGIN_COLOR, CENTRE_LINE_COLOR, \
    CENTRE_LINE_STYLE, \
    MARGIN_LINE_STYLE, DEFAULT_X_LABEL, PLUS_MINUS, FRCE_LEVEL_1_POWER, FRCE_LEVEL_2_POWER, FRCE_LEVEL_1_PERCENTAGE, \
    FRCE_LEVEL_2_PERCENTAGE, FRCE_LEVEL_1_UNIT, FRCE_LEVEL_2_UNIT, HEADER_FIGURE_LOCATION, HEADER_FIGURE_X, \
    HEADER_FIGURE_Y, HEADER_FIGURE_WIDTH, AT_LEAST_ONE_FULL_YEAR_PERIOD, NOT_EARLIER_THAN_SIX_MONTHS, TIME_FORMAT, \
    START_DATE_KEYWORD, END_DATE_KEYWORD, MEAN_KEYWORD, SUM_KEYWORD, STD_KEYWORD, \
    DETERMINISTIC_FIGURE_NAME, ALL_DATA_KEYWORD, POSITIVE_DATA_KEYWORD, NEGATIVE_DATA_KEYWORD, \
    DET_DESCRIPTION_KEYWORD, MC_DESCRIPTION_KEYWORD, MC_FIGURE_NAME, \
    DATE_FORMAT_FOR_REPORT, TEXT_FONT_SIZE, FONT_FAMILY, HEADING_FONT_SIZE, SUBHEADING_FONT_SIZE, \
    UNDER_SUBHEADING_FONT_SIZE, INPUT_DATA_FIGURE_NAME, INITIAL_ALLOWED, INITIAL_UNCORRECTED, \
    TARGET, LOWER_BOUND, UPPER_BOUND, EXCESS_WHEN_APPLIED, SAMPLING_TIME, REGIONS
from py.common.elk_handler_reserves import ElkHandlerForReserves, FROM_KEYWORD, TO_KEYWORD, PY_ELASTICSEARCH_HOST
from py.common.functions import read_csv, get_aceol_data, NUMBER_OF_SAMPLES, NUMBER_OF_SIMULATIONS, \
    TEMP_ACEOL_DATA_FILE, PERCENTILE_9999, PERCENTILE_9999_VALUE, PERCENTILE_0001, PERCENTILE_0001_VALUE

logger = logging.getLogger(__name__)
parse_app_properties(globals(), config.paths.config.sizing_reserves)


def convert_value_to_int(input_value, backup_value):
    try:
        converted_value = int(input_value)
    except ValueError:
        converted_value = backup_value
    return converted_value


# Constants to be loaded in from config file
PY_OFFSET_IN_MONTHS = convert_value_to_int(OFFSET_IN_MONTHS, NOT_EARLIER_THAN_SIX_MONTHS)
PY_YEARS_FOR_ANALYSIS = convert_value_to_int(YEARS_FOR_ANALYSIS, AT_LEAST_ONE_FULL_YEAR_PERIOD)
PY_ANALYSIS_DATE = ANALYSIS_DATE


class CapacityDataSeriesCenterLine:
    """
    Auxiliary class for storing vertical line information
    """

    def __init__(self, percentile: int = 0, deviation: float = 0, center_line: float | None = None, label: str = None):
        """
        Constructor
        :param center_line: vertical line location in x-axis
        :param percentile: identification of center line
        :param deviation: if center line has a confidence level then this represents the +/- values
        :param label: name of the center line in plots
        """
        self.center_line = center_line
        self.deviation = deviation
        self.margins = {}
        if self.deviation > 0:
            self.margins = {'-': center_line - deviation, '+': center_line + deviation}
        self.percentile = percentile
        self.label = label

    def set_center_line_from_data(self, data: pd.Series):
        """
        If center line is not set and data is given, retrieve from there
        :param data: pandas data series
        :return:
        """
        if self.center_line is None:
            self.center_line = data.quantile(q=self.percentile)


class CapacityDataSeries:
    """
    Auxiliary class for storing single data series information for plotting
    """

    def __init__(self, input_data: pd.DataFrame, column: str, label: str, title: str, center_lines=None):
        """

        :param input_data: dataframe containing the data for plotting
        :param column: name of the column for plotting
        :param label: name of the data series from the input data that will be plotted
        :param title: figure title
        :param center_lines: dictionary of vertical lines for the figure
        """
        if center_lines is None:
            center_lines = {}
        self.data = input_data
        self.label = label
        self.column = column
        self.title = title
        self.center_lines = center_lines

    def get_x_lim_values(self):
        """
        Returns min and max values along the x-axis for the data set that will be plotted
        :return: tuple of min and max value "along the x-axis"
        """
        return min(self.data[self.column]), max(self.data[self.column])


class MinimumCapacityData:
    """
    Auxiliary class for storing data for plotting the results
    """

    def __init__(self, input_data=None, description: str = None):
        """
        Constructor
        :param input_data: dictionary of capacity data series as {series name: capacity series}
        :param description: name of the data (used later as part of the title in report dataframe)
        """
        if input_data is None:
            input_data = []
        self.data = input_data
        self.description = description

    def get_number_of_graphs(self):
        """
        Returns number of data series stored within
        :return: number of data series
        """
        return len(self.data)

    def get_x_limit_values(self):
        """
        Finds overall x limits (all graphs share the same scale)
        :return: tuple of min max values
        """
        x_min = 0
        x_max = 0
        for data_series in self.data:
            new_x_min, new_x_max = data_series.get_x_lim_values()
            x_min = min(x_min, new_x_min)
            x_max = max(x_max, new_x_max)
        return x_min, x_max

    def plot_data(self, file_name: str):
        """
        Plots all the data series in one graph, one subplot per data series
        All data series are plotted as density functions with vertical lines coming from center lines
        :param file_name: name of the file where to store the result image
        :return: None
        """
        margin_colors = [LEFT_MARGIN_COLOR, RIGHT_MARGIN_COLOR]
        x_min, x_max = self.get_x_limit_values()
        fig, axes = plt.subplots(1, self.get_number_of_graphs(), squeeze=False)
        image_counter = 0
        for data_series in self.data:
            data_density = data_series.data[data_series.column].plot.kde(ax=axes[0][image_counter],
                                                                         label=data_series.label)
            for center_line in data_series.center_lines:
                data_density.axvline(x=center_line.center_line,
                                     color=CENTRE_LINE_COLOR,
                                     linestyle=CENTRE_LINE_STYLE,
                                     label=center_line.label)
                margin_counter = 0
                for margin in center_line.margins:
                    margin_color_id = margin_counter % len(margin_colors)
                    data_density.axvline(x=center_line.margins[margin],
                                         color=margin_colors[margin_color_id],
                                         linestyle=MARGIN_LINE_STYLE,
                                         label=margin)
                    margin_counter += 1
            data_density.set_title(data_series.title)
            data_density.set_xlim(x_min, x_max)
            data_density.grid()
            # data_density.legend()
            data_density.set_xlabel(DEFAULT_X_LABEL)
            image_counter += 1
        # plt.show()
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight')

    def get_report(self, index_list: {}):
        """
        Generates a report in a form where description + data series name is column and center line values are
        categorized by the sequence numbers of the data series. In case of absence, value is presented as -
        :param index_list: dictionary of indexes and percentiles to be used
        :return: dataframe with results
        """
        result_columns = {}
        for data_series in self.data:
            data_label = f"{self.description}, {data_series.label}"
            data_values = ['-'] * len(index_list)
            for i, index in enumerate(index_list.keys()):
                for center_line in data_series.center_lines:
                    if center_line.percentile == index_list[index]:
                        if center_line.deviation > 0:
                            data_values[i] = f"{center_line.center_line:.1f}{PLUS_MINUS}{center_line.deviation:.1f}"
                        else:
                            data_values[i] = f"{center_line.center_line:.1f}"
            result_columns[data_label] = data_values
        return pd.DataFrame(result_columns, index=index_list)

    def get_values(self, index_value: int):
        """
        Gets all center line values corresponding to the given index_value as the sequence number
        :param index_value: for the center lines
        :return: list of values
        """
        values = []
        for data_series in self.data:
            for center_line in data_series.center_lines:
                if center_line.percentile == index_value:
                    values.append(center_line.center_line)
        return values

    def get_value(self, index_value, data_name):
        """
        Gets a single center line value
        :param index_value: sequence number of center line
        :param data_name: data series name
        :return: value as float
        """
        for data_series in self.data:
            if data_series.label == data_name:
                for center_line in data_series.center_lines:
                    if center_line.percentile == index_value:
                        return center_line.center_line
        return 0


class FrequencyRestorationControlErrorPercentile:
    """
    Data class to store data about the percentiles
    """

    def __init__(self, min_q: float, max_q: float, span: float):
        """
        Initialization
        :param min_q: lower (negative) value
        :param max_q: upper (positive) value
        :param span: percentage of time between min_q and max_q
        """
        self.lower_value = min_q
        self.upper_value = max_q
        self.time_span = span


class FrequencyRestorationControlErrorLevel:
    """
    Depicts the FRCE limit value consisting of limit value, its unit and the percentage of time
    intervals of the year when residual value (ACE=ACEol - FRR) is allowed to surpass the value
    """

    def __init__(self, level: str, error_value: float, error_percentage: float, error_unit: str = 'MW'):
        """
        Constructor
        :param level: Level name, usually 'Level 1' etc.
        :param error_value: target value
        :param error_percentage: percentage of the time instances in year when ACE can exceed the error value.
        :param error_unit: error value unit (usually MW)
        """
        self.level_name = level
        self.target_value = error_value
        self.percent_of_time_from_year = error_percentage
        self.unit = error_unit
        self.percentage_over_aceol_data = 0
        self.percentile = None

    @property
    def portion_of_percentage(self):
        """
        Return percentage of the allowed time intervals from the year as portion from 0 to 1
        :return: decimal indicating the portion
        """
        return self.percent_of_time_from_year / 100

    def set_percentage_over_aceol_data(self, data: pd.Series):
        """
        Calculate the percentage of the values that are exceeding the target value
        :param data: aceol data
        :return: None
        """
        self.percentage_over_aceol_data = 100 * len(data[abs(data) > self.target_value]) / len(data)

    def set_percentiles_to_aceol_data(self, data: pd.Series):
        """
        Sets percentiles according to the self.portion_of_percentage covering both ends of the normal distribution
        of data
        :param data: input aceol data
        :return: None
        """
        data = data.dropna()
        full_length = len(data)
        negative_position = floor(self.portion_of_percentage / 2, 2)
        positive_position = 1 - negative_position
        negative_quantile = ceil(data.quantile(q=negative_position))
        positive_quantile = floor(data.quantile(q=positive_position))
        percentage_of_time = (100 * len(data[(data >= negative_quantile) & (data <= positive_quantile)]) / full_length)
        self.percentile = FrequencyRestorationControlErrorPercentile(min_q=negative_quantile,
                                                                     max_q=positive_quantile,
                                                                     span=percentage_of_time)

    def calculate_symmetric_percentiles(self, percentile_value, data: pd.Series):
        """
        Finds +, - percentiles based on the input (half from the left end, half from the right end)
        :param percentile_value: input percentile value
        :param data: input data
        :return: dictionary containing percentile, its found values and coverage by target value
        """
        negative_position = percentile_value / 2
        positive_position = 1 - negative_position
        positive_percentile = data.quantile(q=positive_position)
        negative_percentile = data.quantile(q=negative_position)
        off_threshold = 100 * len(data[(data > positive_percentile + self.target_value) |
                                       (data < negative_percentile - self.target_value)]) / len(data)
        in_threshold = 100 * len(data[(data <= positive_percentile + self.target_value) &
                                      (data >= negative_percentile - self.target_value)]) / len(data)
        return {'percent': percentile_value,
                'lower': negative_percentile,
                'upper': positive_percentile,
                'off': off_threshold,
                'in': in_threshold}

    def calculate_all_percentiles(self, data: pd.Series):
        """
        Finds all percentiles in a range
        :param data: input data
        :return: dataframe containing percentile, its found values and coverage by target value
        """
        data = data.dropna()
        max_portion = max(0.49, self.portion_of_percentage)
        percentile_values = np.arange(0, max_portion, 0.001)
        outputs = []
        for percentile_value in percentile_values:
            values = self.calculate_symmetric_percentiles(percentile_value, data)
            outputs.append(values)
        return pd.DataFrame(outputs)

    def set_adjusted_percentiles_to_aceol_data(self, data: pd.Series):
        """
        Finds closest match (exponential function) to cover the cases
        x + self.target_value < self.percent_of_time_from_year
        :param data: input data
        :return: None
        """
        data = data.dropna()
        full_length = len(data)
        brute_force_approach = self.calculate_all_percentiles(data)
        closest_match = brute_force_approach.iloc[(brute_force_approach['off'] - self.percent_of_time_from_year)
                                                  .abs()
                                                  .argsort()[:1]]
        negative_quantile = float(closest_match['lower'].iloc[0])
        positive_quantile = float(closest_match['upper'].iloc[0])
        covered = float(closest_match['in'].iloc[0])
        not_covered = float(closest_match['off'].iloc[0])
        percentage_of_time = (100 * len(data[(data >= negative_quantile) & (data <= positive_quantile)]) / full_length)
        self.percentile = FrequencyRestorationControlErrorPercentile(min_q=negative_quantile,
                                                                     max_q=positive_quantile,
                                                                     span=percentage_of_time)
        logger.info(f"Closest match: +:{positive_quantile:.2f}, -:{negative_quantile:.2f}, "
                    f"Over: {not_covered:.2f}%, within: "
                    f"{covered:.2f}%")

    def get_frr_percent_over_level(self, data: pd.Series, positive_fr: float = 0, negative_fr: float = 0):
        """
        Gets the percentage of the values exceeding the given target level after applying the frequency restoration
        values (positive and negative) to the input (aceol) data. Note that the assumption is that data is centered
        around zero
        :param data: aceol_data as Series
        :param positive_fr: positive frequency restoration
        :param negative_fr: negative frequency restoration
        :return: percentages: in negative direction, in positive direction
        """
        # Assume that the normal distribution is centered around 0
        data = data.dropna()
        data_length = len(data)
        if negative_fr <= 0 <= positive_fr:
            positive_data = data[data > positive_fr] - positive_fr
            negative_data = data[data < negative_fr] - negative_fr
            positive_percentage = 100 * len(positive_data[positive_data > abs(self.target_value)]) / data_length
            negative_percentage = 100 * len(negative_data[negative_data < (-1) * abs(self.target_value)]) / data_length
            return negative_percentage, positive_percentage
        return None, None

    def check_frr_percent_over_level(self, data: pd.Series, positive_fr: float = 0, negative_fr: float = 0):
        """
        Checks if total percentage of exceeding values is smaller than allowed percentage
        (self.percent_of_time_from_year)
        :param data: aceol data series
        :param positive_fr: frequency restoration in positive direction
        :param negative_fr: frequency restoration in negative direction
        :return: True if the summed output was smaller, false otherwise
        """
        negative_percent, positive_percent = self.get_frr_percent_over_level(data,
                                                                             positive_fr=positive_fr,
                                                                             negative_fr=negative_fr)
        if negative_percent is not None and positive_percent is not None:
            return negative_percent + positive_percent <= self.percent_of_time_from_year
        return False


class FrequencyRestorationControlError:
    """
    Dataclass for storing FRCE related data. Currently, contains two fixed levels as level 1 and level 2
    """

    def __init__(self,
                 level_1_value: float = FRCE_LEVEL_1_POWER,
                 level_2_value: float = FRCE_LEVEL_2_POWER,
                 level_1_percentage: float = FRCE_LEVEL_1_PERCENTAGE,
                 level_2_percentage: float = FRCE_LEVEL_2_PERCENTAGE,
                 level_1_unit: str = FRCE_LEVEL_1_UNIT,
                 level_2_unit: str = FRCE_LEVEL_2_UNIT):
        """
        Init method
        :param level_1_value: allowed  ABS(ACE = ACEol - FRR)  (level_1_percentage)% of a year
        :param level_2_value:  allowed  ABS(ACE = ACEol - FRR)  (level_2_percentage)% of a year
        :param level_1_percentage: percentage of time instances of the year that where residual value
                                   (level_1_value: after applying FRR to ACEol) is allowed
        :param level_2_percentage: percentage of time instances of the year that where residual value
                                   (level_2_value: after applying FRR to ACEol) is allowed
        :param level_1_unit: unit for level_1_value
        :param level_2_unit: unit for level_2_value
        """
        self.level_1 = FrequencyRestorationControlErrorLevel(level='Level 1',
                                                             error_value=level_1_value,
                                                             error_percentage=level_1_percentage,
                                                             error_unit=level_1_unit)
        self.level_2 = FrequencyRestorationControlErrorLevel(level='Level 2',
                                                             error_value=level_2_value,
                                                             error_percentage=level_2_percentage,
                                                             error_unit=level_2_unit)
        # Analysis results
        self.aceol_data = None

    def set_aceol_data(self, data: pd.Series):
        """
        Sets aceol data and calculates the initial percentage of the values over level 1 and level 2
        :param data: aceol data (raw form)
        :return: None
        """
        self.aceol_data = data
        self.level_1.set_percentage_over_aceol_data(data)
        self.level_2.set_percentage_over_aceol_data(data)

    def set_percentiles(self):
        """
        Sets the percentiles (level_1_percentile and level_2_percentile) to the aceol_data (self.aceol_data)
        :return: None
        """
        if self.aceol_data is not None:
            self.level_1.set_percentiles_to_aceol_data(self.aceol_data)
            self.level_2.set_percentiles_to_aceol_data(self.aceol_data)

    def set_adjusted_percentiles(self):
        if self.aceol_data is not None:
            self.level_1.set_adjusted_percentiles_to_aceol_data(self.aceol_data)
            self.level_2.set_adjusted_percentiles_to_aceol_data(self.aceol_data)


class PDF(FPDF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def header(self):
        """
        Initializes a custom header, will be used by super when add_page is called
        :return: None
        """
        file_abs_path = os.path.abspath(HEADER_FIGURE_LOCATION)
        if not os.path.isfile(file_abs_path):
            current_directory = os.path.dirname(os.path.realpath(__file__))
            file_abs_path = os.path.join(current_directory, HEADER_FIGURE_LOCATION)
        if os.path.isfile(file_abs_path):
            try:
                self.image(file_abs_path, x=HEADER_FIGURE_X, y=HEADER_FIGURE_Y, w=HEADER_FIGURE_WIDTH)
            except FileNotFoundError:
                logger.error(f"Logo was not found at {file_abs_path}")
        else:
            logger.error(f"Logo was not present at {file_abs_path}")
        self.ln(30)

    def footer(self):
        """
        Initializes a custom footer, will be used by super when add_page is called
        :return:
        """
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "I", 10)
        self.cell(0, 10, f"{self.page_no()}/{{nb}}", align="C")


def get_start_and_end_date(interval_years: int = AT_LEAST_ONE_FULL_YEAR_PERIOD,
                           delay_months: int = NOT_EARLIER_THAN_SIX_MONTHS,
                           current_time_moment: str | datetime = None):
    """
    Generates dates between which the analysis is carried out (in years) ending on determined amount of months
     before current date. Formula is start_date: current date - (delay_months + interval_years)
    end_date: current date - delay months
    :param interval_years: number of years used for data analysis
    :param delay_months: number of months between current date and end date needed for the data analysis
    :param current_time_moment: specify time moment if needed
    :return: dictionary with start and end date
    """
    if isinstance(current_time_moment, str):
        try:
            date_time = parse_datetime(current_time_moment)
        except ValueError:
            date_time = None
        try:
            date_value = parse_date(current_time_moment)
        except ValueError:
            date_value = None
        current_time_moment = date_time or date_value
    if not current_time_moment:
        current_time_moment = datetime.now()
    end_date = current_time_moment - relativedelta(months=delay_months)
    start_date = end_date - relativedelta(years=interval_years)
    start_date = pd.to_datetime(start_date.strftime(TIME_FORMAT))
    end_date = pd.to_datetime(end_date.strftime(TIME_FORMAT))
    return {START_DATE_KEYWORD: start_date, END_DATE_KEYWORD: end_date}, current_time_moment


def resample_by_time(data: pd.DataFrame,
                     index_columns=None,
                     method: str = MEAN_KEYWORD,
                     sampling_time: str = SAMPLING_TIME):
    """
    Resamples the dataframe to the new time interval (for example 15 minutes)
    Question: can ACEol data be regarded as continuous (value in next timestamp is dependent on the value in
    previous timestamp) meaning that when resampling, the data should be meaned. Or data is discrete (meaning that
    in every next timestamp new value is provided and the value from the previous timestamp is handled) meaning that
    when resampling, the data should be summed.
    :param method:
    :param index_columns: columns where timestamps are stored, note that reshape by time can be done using single
    timestamp column, rest must be dropped and recalculated by using the sampling time
    :param data: input dataframe
    :param sampling_time: new sampling rate
    :return: updated dataframe
    """
    post_processing = False
    if index_columns is None:
        index_columns = [FROM_KEYWORD, TO_KEYWORD]
    if len(index_columns) > 2:
        raise ValueError("Too main timestamp columns")
    if len(index_columns) > 0:
        post_processing = True
        data = data.set_index(index_columns[0])
    if len(index_columns) == 2:
        data = data.drop(index_columns[1], axis=1)
    if method == MEAN_KEYWORD:
        resampled_data = data.resample(sampling_time).mean()
    elif method == SUM_KEYWORD:
        resampled_data = data.resample(sampling_time).sum()
    else:
        return data
    if post_processing:
        resampled_data = resampled_data.reset_index(names=index_columns[0])
        if len(index_columns) == 2:
            resampled_data[index_columns[1]] = resampled_data[index_columns[0]] + pd.Timedelta(sampling_time)
    return resampled_data


def slice_data_by_time_range(data: pd.DataFrame,
                             time_ranges: dict | None = None,
                             column_to_slice: str = FROM_KEYWORD):
    """
    Get slice from the data between given dates. Note that slicing
    :param column_to_slice: Name of the column by which the data is sliced (by default is the 'from' column)
    :param data: input dataframe consisting at least columns 'from' and 'to' which can be converted to datetime
    :param time_ranges:
    :return: sliced data
    """
    if time_ranges is None or START_DATE_KEYWORD not in time_ranges or END_DATE_KEYWORD not in time_ranges:
        time_ranges = get_start_and_end_date()
    if FROM_KEYWORD not in data.columns or TO_KEYWORD not in data.columns:
        raise ValueError("Missing 'to' and/or 'from' columns in the input data")
    if column_to_slice != FROM_KEYWORD and column_to_slice != TO_KEYWORD:
        raise ValueError("Unknown column by which to slice")
    start_date = time_ranges[START_DATE_KEYWORD]
    end_date = time_ranges[END_DATE_KEYWORD]
    data[FROM_KEYWORD] = pd.to_datetime(data[FROM_KEYWORD])
    data[TO_KEYWORD] = pd.to_datetime(data[TO_KEYWORD])
    if start_date is not None and end_date is not None:
        data_in_use = data.loc[(data[column_to_slice] >= start_date) & (data[column_to_slice] <= end_date)]
    elif start_date is not None:
        data_in_use = data.loc[(data[column_to_slice] >= start_date)]
    elif end_date is not None:
        data_in_use = data.loc[(data[column_to_slice] <= end_date)]
    else:
        data_in_use = data
    return data_in_use


def str_to_datetime(data: pd.DataFrame, columns: []):
    """
    Converts string to pandas date time
    :param data: input dataframe
    :param columns: list of columns to convert
    :return: updated dataframe
    """
    for column in columns:
        data[column] = pd.to_datetime((data[column]))
    return data


def find_percentiles(data: pd.DataFrame,
                     column_name: str,
                     percentiles: {},
                     percentile_values: {}):
    """

    :param percentile_values:
    :param data:
    :param column_name:
    :param percentiles:
    :return:
    """
    centres = []
    for percentile in percentiles:
        deviation_value = 0
        if percentile in percentile_values and MEAN_KEYWORD in percentile_values[percentile]:
            percentile_value = percentile_values[percentile][MEAN_KEYWORD]
            if STD_KEYWORD in percentile_values[percentile]:
                deviation_value = percentile_values[percentile][STD_KEYWORD]
        else:
            percentile_value = data[column_name].quantile(q=percentiles[percentile])
        center = CapacityDataSeriesCenterLine(center_line=percentile_value,
                                              label=percentile,
                                              percentile=percentiles[percentile],
                                              deviation=deviation_value)
        centres.append(center)
    return centres


def compose_data_series(data: pd.DataFrame,
                        column_name: str,
                        label: str,
                        percentiles: {},
                        percentile_values: {} = None,
                        show_percentiles: bool = True):
    """
    Wraps the data to custom class that draws a figure of data and percentiles with values on it
    :param percentile_values: custom percentile values
    :param data: input data, basis of the graph
    :param column_name: column from which the
    :param label: data[column_name] legend entry
    :param percentiles: percentiles to be added to the figure (note that show_percentiles should be True in this case)
    :param show_percentiles: if percentiles would be shown
    :return: custom class
    """
    if percentile_values is None:
        percentile_values = {}
    center_lines = []
    data_title = ""
    all_lines = len(data)
    if show_percentiles:
        center_lines = find_percentiles(data=data,
                                        column_name=column_name,
                                        percentiles=percentiles,
                                        percentile_values=percentile_values)
        if len(center_lines) == 2:
            center_line_values = [x.center_line for x in center_lines]
            min_center_line = min(center_line_values)
            max_center_line = max(center_line_values)
            if min_center_line < data[column_name].mean() < max_center_line:
                data_title = (
                    f"+: {len(data[(data[column_name] > max_center_line)]) * 100 / all_lines:.3f}% / "
                    f"-: {len(data[(data[column_name] < min_center_line)]) * 100 / all_lines:.3f}%")
    return CapacityDataSeries(input_data=data,
                              label=label,
                              title=data_title,
                              column=column_name,
                              center_lines=center_lines)


def run_deterministic_analysis(data: pd.DataFrame,
                               main_column_name: str,
                               percentiles: {},
                               use_pos_neg_data_separately: bool = False,
                               show_percentiles: bool = True,
                               image_name: str = DETERMINISTIC_FIGURE_NAME):
    """
    Runs the deterministic approach based on assumption that data is normally distributed. Finds center lines
    as fixed percentiles. Handles three cases: all data, extracted positive values and extracted negative values
    :param percentiles:
    :param show_percentiles:
    :param image_name:
    :param use_pos_neg_data_separately:
    :param data: dictionary as {region name: region data as pandas dataframe}
    :param main_column_name: name of the region or the column in which to carry out the analysis
    :return: MinimumCapacityData instance
    """
    ElkHandlerForReserves.sys_print(f"\rAnalysing FRR data in {main_column_name}")
    data_to_image = [compose_data_series(data=data,
                                         column_name=main_column_name,
                                         label=ALL_DATA_KEYWORD,
                                         percentiles=percentiles,
                                         show_percentiles=show_percentiles)]
    # Following part is for illustration purposes only. Do not proceed
    if use_pos_neg_data_separately:
        pos_data = data.loc[data[main_column_name] >= 0]
        neg_data = data.loc[data[main_column_name] <= 0]
        pos_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value >= 0.5}
        neg_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value <= 0.5}
        data_to_image.append(compose_data_series(data=pos_data,
                                                 column_name=main_column_name,
                                                 label=POSITIVE_DATA_KEYWORD,
                                                 percentiles=pos_percentiles,
                                                 show_percentiles=show_percentiles))
        data_to_image.append(compose_data_series(data=neg_data,
                                                 column_name=main_column_name,
                                                 label=NEGATIVE_DATA_KEYWORD,
                                                 percentiles=neg_percentiles,
                                                 show_percentiles=show_percentiles))
    result = MinimumCapacityData(description=DET_DESCRIPTION_KEYWORD,
                                 input_data=data_to_image)
    result.plot_data(image_name)
    ElkHandlerForReserves.sys_print(f"\rAnalysing FRR data in {main_column_name}: Done\n")
    return result


def find_change_speed(data: pd.DataFrame, region: str, time_column: str):
    """
    Finds the speed (delta change) by which input data changes
    :param data: input aceol data
    :param region: region name
    :param time_column:  time column
    :return: dataframe with changes
    """
    start_times = []
    time_differences = []
    change_differences = []
    for i in range(len(data.index) - 1):
        start_times.append(data[time_column].iloc[i])
        time_differences.append(data[time_column].iloc[i + 1] - data[time_column].iloc[i])
        change_differences.append(data[region].iloc[i + 1] + data[region].iloc[i])
    data_dict = {time_column: start_times, 'delta_t': time_differences, region: change_differences}
    changes = pd.DataFrame.from_dict(data_dict)
    return changes


def simulate_mc(data: pd.DataFrame,
                data_column: str,
                percentiles: {},
                number_of_samples: int = NUMBER_OF_SAMPLES,
                number_of_simulations: int = NUMBER_OF_SIMULATIONS):
    """
    Simulates Monte Carlo on data frame (number_of_simulations x number_of_samples). Returns the dataframe consisting
    of percentiles of the requested column retrieved after every simulation
    :param data: input dataframe
    :param data_column: column in which percentiles are found after each simulation
    :param percentiles: dictionary of percentiles {percentile_name: percentile_value}
    :param number_of_samples: number of randomly selected samples per simulation
    :param number_of_simulations: number of simulations
    :return: dataframe consisting of percentile values
    """
    sim_results = []
    for sim in range(number_of_simulations):
        data_mc = data.sample(number_of_samples, replace=True)
        sim_result = {'n': sim}
        for percentile in percentiles:
            sim_result[percentile] = data_mc[data_column].quantile(q=percentiles[percentile])
            sim_results.append(sim_result)
        ElkHandlerForReserves.sys_print(f"\rSimulating FRR in {data_column}: "
                                        f"{100 * sim / number_of_simulations:.2f}% done")
    return pd.DataFrame(sim_results)


def get_mean_of_columns(data: pd.DataFrame, columns: [], find_std: bool = True):
    """
    Calculates mean to columns indicated (and standard deviation if needed)
    :param data: input data frame
    :param columns: list of columns for which mean is needed
    :param find_std: True: calculates standard deviation also
    :return: dictionary {column_name: {'mean': mean of column, Optional('std': standard deviation of column)}}
    """
    results = {}
    for column in columns:
        if column in data.columns:
            results[column] = {MEAN_KEYWORD: data[column].mean()}
            if find_std:
                results[column][STD_KEYWORD] = data[column].std()
    return results


def run_mc_on_data(data: pd.DataFrame,
                   main_column_name: str,
                   percentiles: {},
                   use_pos_neg_data_separately: bool = True,
                   show_percentiles: bool = True,
                   number_of_samples: int = NUMBER_OF_SAMPLES,
                   number_of_simulations: int = NUMBER_OF_SIMULATIONS):
    """
    Runs 'probabilistic' analysis by sampling randomly input data and extracts given fixed percentiles
    as center lined
    :param percentiles:
    :param use_pos_neg_data_separately:
    :param show_percentiles: show percentiles
    :param data: dictionary as {region name: region data as pandas dataframe}
    :param main_column_name: name of the region
    :param number_of_samples: samples per simulation
    :param number_of_simulations: number of simulations
    :return: MinimumCapacityData instance
    """
    all_data_simulated = simulate_mc(data=data,
                                     data_column=main_column_name,
                                     percentiles=percentiles,
                                     number_of_samples=number_of_samples,
                                     number_of_simulations=number_of_simulations)
    all_data_mean = get_mean_of_columns(data=all_data_simulated, columns=percentiles.keys())
    data_series = [compose_data_series(data=data,
                                       column_name=main_column_name,
                                       label=ALL_DATA_KEYWORD,
                                       percentiles=percentiles,
                                       percentile_values=all_data_mean,
                                       show_percentiles=show_percentiles)]
    # Following part is for illustration purposes only. Do not proceed
    if use_pos_neg_data_separately:
        pos_data = data.loc[data[main_column_name] >= 0]
        pos_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value >= 0.5}
        pos_data_simulated = simulate_mc(data=pos_data,
                                         data_column=main_column_name,
                                         percentiles=pos_percentiles,
                                         number_of_samples=number_of_samples,
                                         number_of_simulations=number_of_simulations)
        pos_data_mean = get_mean_of_columns(data=pos_data_simulated, columns=pos_percentiles.keys())
        data_series.append(compose_data_series(data=pos_data,
                                               column_name=main_column_name,
                                               label=POSITIVE_DATA_KEYWORD,
                                               percentiles=pos_percentiles,
                                               percentile_values=pos_data_mean,
                                               show_percentiles=show_percentiles))
        neg_data = data.loc[data[main_column_name] <= 0]
        neg_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value <= 0.5}
        neg_data_simulated = simulate_mc(data=neg_data,
                                         data_column=main_column_name,
                                         percentiles=neg_percentiles,
                                         number_of_samples=number_of_samples,
                                         number_of_simulations=number_of_simulations)
        neg_data_mean = get_mean_of_columns(data=neg_data_simulated, columns=neg_percentiles.keys())
        data_series.append(compose_data_series(data=neg_data,
                                               column_name=main_column_name,
                                               label=NEGATIVE_DATA_KEYWORD,
                                               percentiles=neg_percentiles,
                                               percentile_values=neg_data_mean,
                                               show_percentiles=show_percentiles))
    result = MinimumCapacityData(description=MC_DESCRIPTION_KEYWORD,
                                 input_data=data_series)
    result.plot_data(MC_FIGURE_NAME)
    ElkHandlerForReserves.sys_print(f"\rSimulating FRR in {main_column_name}: Done\n")
    return result


def generate_report(heading_string: str,
                    methodologies: {},
                    summaries: {},
                    region_list: [],
                    tables: {},
                    images: {},
                    references: {} = None,
                    file_name: str = None,
                    time_ranges=None,
                    date_today: str | datetime = None):
    """
    Generates a pdf report from this analysis
    (Taken from https://towardsdtatascience.com/how-to-create-a-pdf-report-fro-your-data-analysis-in-python)
    :param references: list of references
    :param methodologies: Methodology of the analysis process
    :param heading_string: The title of the report
    :param summaries: Main results with comments
    :param region_list: List of regions where analysis was performed
    :param tables: dictionary of tables (table caption: pandas.dataframe as table)
    :param images: dictionary of image addresses (image caption: image location)
    :param time_ranges: start and end date which were used for the analysis
    :param file_name: name of the file where to save the report
    :param date_today:
    """
    if time_ranges is None:
        time_ranges = get_start_and_end_date()
    if not isinstance(date_today, str):
        try:
            date_today = date_today.strftime(DATE_FORMAT_FOR_REPORT)
        except ValueError:
            date_today = pd.Timestamp("today").strftime(DATE_FORMAT_FOR_REPORT)

    cell_height = 8
    line_height = TEXT_FONT_SIZE
    margin_between_paragraphs = 6

    start_date = time_ranges[START_DATE_KEYWORD].strftime(DATE_FORMAT_FOR_REPORT)
    end_date = time_ranges[END_DATE_KEYWORD].strftime(DATE_FORMAT_FOR_REPORT)
    region = ", ".join(str(region_value) for region_value in region_list)

    pdf_file = PDF("P", "mm", "A4")
    pdf_file.set_margins(left=10, top=10)
    pdf_file.set_text_color(r=0, g=0, b=0)
    pdf_file.add_page()
    pdf_file.set_font(FONT_FAMILY, 'B', HEADING_FONT_SIZE)
    pdf_file.cell(w=0, h=3 * cell_height, text=heading_string, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
    pdf_file.cell(w=30, h=cell_height, text='Date:', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf_file.cell(w=30, h=cell_height, text=date_today, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf_file.cell(w=60, h=cell_height, text='Analysis is performed for ', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf_file.cell(w=60, h=cell_height, text=region, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf_file.cell(w=60, h=cell_height, text='From: ', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf_file.cell(w=60, h=cell_height, text=start_date, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf_file.cell(w=60, h=cell_height, text='To:', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf_file.cell(w=60, h=cell_height, text=end_date, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf_file.set_font(FONT_FAMILY, 'B', SUBHEADING_FONT_SIZE)
    pdf_file.cell(w=0, h=2 * cell_height, text="Methodology", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
    for methodology in methodologies:
        pdf_file.set_font(FONT_FAMILY, 'I', UNDER_SUBHEADING_FONT_SIZE)
        pdf_file.cell(w=0, h=2 * cell_height, text=methodology, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
        pdf_file.multi_cell(w=0, h=5, text=methodologies[methodology])
        pdf_file.ln(margin_between_paragraphs)

    pdf_file.set_font(FONT_FAMILY, 'B', SUBHEADING_FONT_SIZE)
    pdf_file.cell(w=0, h=2 * cell_height, text="Results", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
    for summary in summaries:
        pdf_file.multi_cell(w=0, h=5, text=summaries[summary])
        pdf_file.ln(margin_between_paragraphs)

    table_counter = 1
    for table_description in tables:
        pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
        table_heading = f"Table {table_counter}: {table_description}"
        pdf_file.multi_cell(w=0, h=5, text=table_heading)
        pdf_file.ln(margin_between_paragraphs)
        table_counter += 1
        table = tables[table_description]

        cell_width = (210 - 10 - 10) / (len(table.columns) + 1)

        number_of_lines = 1
        for column_name in table.columns:
            new_number_lines = math.ceil(pdf_file.get_string_width(str(column_name)) / cell_width)
            number_of_lines = max(new_number_lines, number_of_lines)
        pdf_file.multi_cell(w=cell_width,
                            h=line_height * number_of_lines * 1,
                            text="",
                            align="C",
                            border="B",
                            new_x="RIGHT",
                            new_y="TOP",
                            max_line_height=line_height)
        for column_name in table.columns:
            pdf_file.multi_cell(w=cell_width,
                                h=line_height * number_of_lines * 1,
                                text=str(column_name),
                                align="C",
                                border="B",
                                new_x="RIGHT",
                                new_y="TOP",
                                max_line_height=line_height)
        pdf_file.ln(line_height * 1 * number_of_lines)

        for index, row in table.iterrows():
            number_of_lines = 1
            new_number_lines = math.ceil(pdf_file.get_string_width(str(index)) / cell_width)
            number_of_lines = max(new_number_lines, number_of_lines)
            for i in range(len(table.columns)):
                new_number_lines = math.ceil(pdf_file.get_string_width(str(row.iloc[i])) / cell_width)
                number_of_lines = max(new_number_lines, number_of_lines)
            pdf_file.multi_cell(w=cell_width,
                                h=line_height * number_of_lines * 1,
                                text=str(index),
                                align="C",
                                border="B",
                                new_x="RIGHT",
                                new_y="TOP",
                                max_line_height=line_height)
            for i in range(len(table.columns)):
                pdf_file.multi_cell(w=cell_width,
                                    h=line_height * number_of_lines * 1,
                                    text=str(row.iloc[i]),
                                    align="C",
                                    border="B",
                                    new_x="RIGHT",
                                    new_y="TOP",
                                    max_line_height=line_height)
            pdf_file.ln(line_height * 1 * number_of_lines)
        pdf_file.ln(cell_height)
    pdf_file.ln(cell_height)

    figure_counter = 1
    for image in images:
        # if figure_counter > 1 & figure_counter % 2 == 1:
        #    pdf_file.add_page()
        pdf_file.image(images[image], w=pdf_file.epw)
        figure_caption = f"Figure {figure_counter}: {image}"
        pdf_file.ln(cell_height)
        pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
        pdf_file.multi_cell(w=0, h=5, text=figure_caption)
        pdf_file.ln(cell_height)
        figure_counter += 1

    if references is not None and len(references) > 0:
        pdf_file.ln(cell_height)
        pdf_file.set_font(FONT_FAMILY, 'B', SUBHEADING_FONT_SIZE)
        pdf_file.cell(w=0, h=2 * cell_height, text="References", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf_file.set_font(FONT_FAMILY, '', TEXT_FONT_SIZE)
        for reference in references:
            pdf_file.cell(w=10, h=5, text=str(reference), new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf_file.multi_cell(w=180, h=5, text=references[reference], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf_file.ln(cell_height)

    if file_name is not None:
        pdf_file.output(file_name)
        webbrowser.open(file_name)
        return None
    else:
        return pdf_file.output()


def ceil(input_value: float, decimals: int = 0):
    """
    Extends built-in ceil to ceil by number of decimals indicated by decimals:
    floor(10.5612, 2) = 10.57
    :param input_value: value to be rounded up
    :param decimals: number of digits after separator
    :return: rounded value
    """
    decimals_value = pow(10, decimals)
    return math.ceil(input_value * decimals_value) / decimals_value


def floor(input_value: float, decimals: int = 0):
    """
    Extends built-in floor to floor by number of decimals indicated by decimals:
    floor(10.5678, 2) = 10.56
    :param input_value: value to be rounded down
    :param decimals: number of digits after separator
    :return: rounded value
    """
    decimals_value = pow(10, decimals)
    return math.floor(input_value * decimals_value) / decimals_value


def draw_input_data(data: pd.DataFrame, regions: [], file_name: str = None):
    """
    Draws a figure depicting the input data
    :param regions: list of regions to show
    :param data: dataframe with ACEol data
    :param file_name: name of location where and if to save
    :return: none
    """
    if FROM_KEYWORD not in data.columns:
        return
    x_min = data[FROM_KEYWORD].min()
    x_max = data[FROM_KEYWORD].max()
    fig_cols = int(min(2, len(regions)))
    fig_rows = int(ceil(len(regions) / fig_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, squeeze=False)
    for i, region in enumerate(regions):
        # y_max = max(abs(data[regions[i]])) * 1.2
        image_col = i % fig_cols
        image_row = i // fig_rows
        time_series = data.plot(x=FROM_KEYWORD, y=region, ax=axes[image_row][image_col], label=region)
        # time_series = data.plot(x=FROM_KEYWORD, y=region)
        time_series.set_title(f"Input ACEol data")
        time_series.set_xlim(x_min, x_max)
        time_series.grid()
        time_series.set_xlabel("Time")
        time_series.set_ylabel("ACEol (MW)")
    if file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight')


def save_report_to_minio(reserves_report):
    minio_instance = ReportMinioStorage()
    link_value = minio_instance.save_file_to_minio_with_link(buffer=reserves_report)
    logger.info(f"Report can be downloaded at {link_value}")
    return link_value


def delete_file_from_local_storage(file_path: str):
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.info(f"Removed {file_path} from local storage")
    else:
        logger.warning(f"Unable to delete {file_path}, file doesn't exist")


if __name__ == '__main__':
    """ RUN THIS """

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    region_to_investigate = REGIONS[0]
    draw_raw_data_image = True  # set it to true if image with raw data adds some value, for example estimate
    # scale of fluctuation on the timescale

    # Local file is elk_query data (covering all the date range) saved from R workspace (to reduce the overhead of the
    # Load in data, use local csv to reduce the overhead when coding and debugging
    """------------------------Load the data-----------------------------------------------------------------------"""
    try:
        aceol_data = read_csv(TEMP_ACEOL_DATA_FILE)
    except FileNotFoundError:
        elk = ElkHandlerForReserves(server=PY_ELASTICSEARCH_HOST)
        reserve_query_dict = {"match_all": {}}
        aceol_data = get_aceol_data(elk_instance=elk, aceol_query=reserve_query_dict)

    """------------------------Preprocessing-----------------------------------------------------------------------"""
    # Preprocessing: resample it to 15 min (by methodology), and slice a period from 1.5 years to 0.5 from current time
    # stamp
    aceol_data = str_to_datetime(aceol_data, [FROM_KEYWORD, TO_KEYWORD])
    aceol_data = resample_by_time(aceol_data)
    time_range, date_of_generation = get_start_and_end_date(interval_years=PY_YEARS_FOR_ANALYSIS,
                                                            delay_months=PY_OFFSET_IN_MONTHS,
                                                            current_time_moment=PY_ANALYSIS_DATE)
    # Leave the left endpoint free: AT LEAST one year of data ending not before than 6 months before the analysis
    # time_range[START_DATE_KEYWORD] = None
    aceol_data_in_time_range = slice_data_by_time_range(data=aceol_data, time_ranges=time_range)
    if not os.path.isfile(INPUT_DATA_FIGURE_NAME):
        if draw_raw_data_image:
            logger.info(f"Saving image with raw and uncompressed data, it may take a while...")
            draw_input_data(aceol_data_in_time_range, [region_to_investigate], INPUT_DATA_FIGURE_NAME)
    # Update the left endpoint from the data
    time_range[START_DATE_KEYWORD] = aceol_data_in_time_range[FROM_KEYWORD].min()
    """------------------------Run 'main analysis'-----------------------------------------------------------------"""
    # Draw percentiles on the raw data
    analysis_percentiles = {PERCENTILE_9999: PERCENTILE_9999_VALUE, PERCENTILE_0001: PERCENTILE_0001_VALUE}

    det_data = run_deterministic_analysis(data=aceol_data_in_time_range,
                                          main_column_name=region_to_investigate,
                                          use_pos_neg_data_separately=False,
                                          percentiles=analysis_percentiles)
    # Just for the fun of it, resample the normal distribution to get the normal distribution, after which draw the
    # percentiles and conclude that results are same than in previous case
    mc_data = run_mc_on_data(data=aceol_data_in_time_range,
                             main_column_name=region_to_investigate,
                             use_pos_neg_data_separately=False,
                             percentiles=analysis_percentiles,
                             number_of_simulations=50,
                             number_of_samples=300000)
    """------------------------Run 'main analysis'-----------------------------------------------------------------"""
    index_values = {'99.99% for +': PERCENTILE_9999_VALUE, '99.99% for -': PERCENTILE_0001_VALUE}
    # Prepare for the report
    det_data_report = det_data.get_report(index_list=index_values)
    mc_data_report = mc_data.get_report(index_list=index_values)
    overall_report = pd.concat([det_data_report, mc_data_report], axis=1)
    # overall_report.to_csv('analysis_results.csv', sep=DELIMITER)

    min_all_value_det = det_data.get_value(index_value=PERCENTILE_0001_VALUE, data_name=ALL_DATA_KEYWORD)
    max_all_value_det = det_data.get_value(index_value=PERCENTILE_9999_VALUE, data_name=ALL_DATA_KEYWORD)
    min_all_value_mc = mc_data.get_value(index_value=PERCENTILE_0001_VALUE, data_name=ALL_DATA_KEYWORD)
    max_all_value_mc = mc_data.get_value(index_value=PERCENTILE_9999_VALUE, data_name=ALL_DATA_KEYWORD)

    # For viewing the speed in which the ACEol changes (over regulated, under regulated etc.)
    # change_speed = find_change_speed(data=aceol_data_in_time_range,
    #                                  time_column=FROM_KEYWORD,
    #                                  region=region_to_investigate)
    # change_analysis = run_deterministic_analysis(data=change_speed,
    #                                              main_column_name=region_to_investigate,
    #                                              use_pos_neg_data_separately=False,
    #                                              show_percentiles=False,
    #                                              percentiles=analysis_percentiles,
    #                                              image_name='Change_speed.png')

    """------------------------Run FRCE analysis------------------------------------------------------------------"""
    frce = FrequencyRestorationControlError()
    frce.set_aceol_data(data=aceol_data_in_time_range[region_to_investigate])
    # frce.set_percentiles()
    frce.set_adjusted_percentiles()
    dt_1_min, dt_1_max = frce.level_1.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_det,
                                                                 negative_fr=min_all_value_det)
    dt_2_min, dt_2_max = frce.level_2.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_det,
                                                                 negative_fr=min_all_value_det)
    mc_1_min, mc_1_max = frce.level_1.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_mc,
                                                                 negative_fr=min_all_value_mc)
    mc_2_min, mc_2_max = frce.level_2.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_mc,
                                                                 negative_fr=min_all_value_mc)
    sf_1_min, sf_1_max = frce.level_1.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=frce.level_1.percentile.upper_value,
                                                                 negative_fr=frce.level_1.percentile.lower_value)
    sf_2_min, sf_2_max = frce.level_2.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=frce.level_2.percentile.upper_value,
                                                                 negative_fr=frce.level_2.percentile.lower_value)
    det_label = f"{det_data.description}, {ALL_DATA_KEYWORD}"
    mc_label = f"{mc_data.description}, {ALL_DATA_KEYWORD}"
    level_1_dict = {INITIAL_ALLOWED: f"{frce.level_1.percent_of_time_from_year}%",
                    INITIAL_UNCORRECTED: f"{frce.level_1.percentage_over_aceol_data:.2f}%",
                    det_label: f"{(dt_1_min + dt_1_max):.4f}%",
                    mc_label: f"{(mc_1_min + mc_1_max):.4f}%"}
    level_2_dict = {INITIAL_ALLOWED: f"{frce.level_2.percent_of_time_from_year}%",
                    INITIAL_UNCORRECTED: f"{frce.level_2.percentage_over_aceol_data:.2f}%",
                    det_label: f"{(dt_2_min + dt_2_max):.4f}%",
                    mc_label: f"{(mc_2_min + mc_2_max):.4f}%"}
    frce_report = pd.DataFrame([level_1_dict, level_2_dict], index=['Level 1', 'Level 2'])

    level_1_self_values = {INITIAL_ALLOWED: f"{frce.level_1.percent_of_time_from_year}%",
                           TARGET: f"{frce.level_1.target_value}MW",
                           LOWER_BOUND: f"{frce.level_1.percentile.lower_value:.2f}MW",
                           UPPER_BOUND: f"{frce.level_1.percentile.upper_value:.2f}MW",
                           EXCESS_WHEN_APPLIED: f"{(sf_1_min + sf_1_max):.4f}%"}
    level_2_self_values = {INITIAL_ALLOWED: f"{frce.level_2.percent_of_time_from_year}%",
                           TARGET: f"{frce.level_2.target_value}MW",
                           LOWER_BOUND: f"{frce.level_2.percentile.lower_value:.2f}MW",
                           UPPER_BOUND: f"{frce.level_2.percentile.upper_value:.2f}MW",
                           EXCESS_WHEN_APPLIED: f"{(sf_2_min + sf_2_max):.4f}%"}
    frce_self_report = pd.DataFrame([level_1_self_values, level_2_self_values], index=['Level 1', 'Level 2'])

    dt_1_passed = frce.level_1.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_det,
                                                            negative_fr=min_all_value_det)
    dt_2_passed = frce.level_2.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_det,
                                                            negative_fr=min_all_value_det)
    mc_1_passed = frce.level_1.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_mc,
                                                            negative_fr=min_all_value_mc)
    mc_2_passed = frce.level_2.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_mc,
                                                            negative_fr=min_all_value_mc)
    det_magic_string = "none of the levels"
    mc_magic_string = det_magic_string
    if dt_1_passed and dt_2_passed:
        det_magic_string = "both of the levels"
    elif dt_1_passed:
        det_magic_string = frce.level_1.level_name
    elif dt_2_passed:
        det_magic_string = frce.level_2.level_name
    if mc_1_passed and mc_2_passed:
        mc_magic_string = "both of the levels"
    elif mc_1_passed:
        mc_magic_string = frce.level_1.level_name
    elif mc_2_passed:
        mc_magic_string = frce.level_2.level_name
    """------------------------Generate the report----------------------------------------------------------------"""
    print("Results: reserve capacities")
    print(overall_report.to_markdown())
    print("Comparison with frce results")
    print(frce_report.to_markdown())
    # overall_report = read_csv('analysis_results.csv')
    # reset the index if contemporary results were loaded from the file
    if 'Unnamed: 0' in list(overall_report.columns):
        overall_report.set_index(['Unnamed: 0'], inplace=True)
    heading = 'Calculation of minimum reserve capacity at SOR level'
    methodology_calc_heading = "Calculation of reserve capacities"
    methodology_calc = (f"Calculation is based on the at least 1 "
                        f"year data ending not later than 6 months "
                        f"before the time when current report was created. The results (positive (+) and "
                        f"negative values (-)) are "
                        f"summed over the SOR (here noted as {region_to_investigate} region)."
                        f"Calculation is performed in two ways:\n"
                        f"a) Deterministic approach (Det.) finds {PERCENTILE_9999_VALUE * 100}% percentiles  from ACEol"
                        f" (Area Control Error Open loop) data directly, assuming that it is normally distributed.\n"
                        f"b) Probabilistic approach (MC) samples the input data {NUMBER_OF_SAMPLES} times per "
                        f"simulation ({NUMBER_OF_SIMULATIONS} in total). From the results mean of  "
                        f"{PERCENTILE_9999_VALUE * 100}% "
                        f"percentiles is taken with 1.96 * standard deviation indicating confidence level of 95% "
                        f"({PLUS_MINUS}) over all the {NUMBER_OF_SIMULATIONS} simulations.")
    methodology_frce_heading = "Comparison of results with FRCE values"
    methodology_frce = (f"Here FRCE (Frequency Restoration Control Error) represents the absolute (in positive and "
                        f"negative direction) limit values for the ACE:\n"
                        f"\n"
                        f"                                                  "
                        f"ACE = ACEol + (mFRR + aFRR)"
                        f"                                                  "
                        f"[1]\n\n"
                        f"Here the following cases are considered:\n"
                        f"a) For {frce.level_1.level_name} the residual value (ACE) cannot exceed "
                        f"{frce.level_1.target_value}{frce.level_1.unit} for more than "
                        f"{frce.level_1.percent_of_time_from_year}% of time intervals of the year\n"
                        f"b) For {frce.level_2.level_name} the residual value (ACE) cannot exceed "
                        f"{frce.level_2.target_value}{frce.level_2.unit} for more than "
                        f"{frce.level_2.percent_of_time_from_year}% of time intervals of the year\n"
                        f"For estimating the amount of exceed cases, the found reserve capacities (varying from 0 to "
                        f"maximum value) were applied to the uncorrected (ACEol) data. From the results extreme "
                        f"cases, that exceed the targets set by levels (a and b) were extracted. Their amount is "
                        f"represented as percentage of the time intervals of the year in corresponding table.\n"
                        f"In order to illustrate the reserve capacities needed to fulfill the FRCE requirements, the "
                        f"reverse process by leveling the data with FRCE target values and determining the needed "
                        f"minimal percentiles numerically were carried out. Corresponding results are depicted in table"
                        f" 3 by the FRCE levels. These values, however, should be regarded informative only as "
                        f"they are based solely on the data and do not represent the real situation.")
    list_of_methodologies = {methodology_calc_heading: methodology_calc, methodology_frce_heading: methodology_frce}
    main_summary_heading = "main_summary"
    main_summary = (f"Based on the instructions, the minimum reserve capacity is found from the summed results"
                    f" (positive and negative values at SOR level (presented on the left side figures below). "
                    f"Analysis solely based on negative and positive values is here only for the reference. "
                    f"(middle and right side figure)\n"
                    f"Method (a) gave for the positive capacity {max_all_value_det:.1f}MW and method "
                    f"(b) {max_all_value_mc:.1f}MW with {max(max_all_value_det, max_all_value_mc):.1f}MW as general "
                    f"recommendation. "
                    f"For the negative values, method (a) produced {min_all_value_det:.1f}MW and method "
                    f"(b) {min_all_value_mc:.1f}MW with {min(min_all_value_det, min_all_value_mc):.1f}MW as "
                    f"general recommendation.\n")
    frce_summary_heading = "frce_summary"
    frce_summary = (f"From the results it is possible to conclude that values proposed by Method (a) satisfy "
                    f"{det_magic_string} and values proposed by Method (b) satisfy {mc_magic_string}")
    list_of_summaries = {main_summary_heading: main_summary, frce_summary_heading: frce_summary}
    deterministic_fig_title = (f'Deterministic solution, blue lines represent values at {PERCENTILE_9999_VALUE * 100}%.'
                               f'Figures from left: a) All data summed, b) extracted positive values, c) extracted '
                               f'negative values.Amount of extreme cases is indicated on the top of the figure')
    mc_fig_title = (f"Monte Carlo results, blue lines represent values at {PERCENTILE_9999_VALUE * 100}%, dashed lines "
                    f"represent {PLUS_MINUS}95%. Figures from left: a) All data summed, b) extracted positive values, "
                    f"c) extracted negative values. Amount of extreme cases is indicated on the top of the figure")
    input_data_fig_title = (f"Input data (ACEol) for {region_to_investigate} "
                            f"from {time_range[START_DATE_KEYWORD].strftime(DATE_FORMAT_FOR_REPORT)} to "
                            f"{time_range[END_DATE_KEYWORD].strftime(DATE_FORMAT_FOR_REPORT)}")
    table_title = (f"Capacities with different methods (MC: Monte Carlo, Det. deterministic) "
                   f"and different datasets (all values in MW)")
    frce_table_title = (f'Comparison of FRCE levels (percentage value in column "{INITIAL_ALLOWED}"  and percentage of '
                        f'time moments when ACEol exceeded the target in column "{INITIAL_UNCORRECTED}". After '
                        f'applying the values found during the calculation of the reserve capacities, the percentage '
                        f'of the time moments when residual value exceeded the target of the level is shown in '
                        f'columns "{det_label}" and "{mc_label}" respectively.')
    frce_self_table_title = (f'Minimum values for reserve capacities considering the requirements of FRCE levels. '
                             f'Column "{INITIAL_ALLOWED}" maximum allowed time moments when ACE can exceed '
                             f'the value in column "{TARGET}". Numerically minimum values are in columns '
                             f'("{UPPER_BOUND}") and ("{LOWER_BOUND}"). '
                             f'Column "{EXCESS_WHEN_APPLIED}" shows percentage of time moments '
                             f'when ACE exceeds the value in "{TARGET}" after applying up and down values.')
    reference_list = {'[1]': "ENTSO-E Proposal for the Regional Coordination Centres' task 'regional sizing of "
                             "reserve capacity' in accordance with Article 37(1)(j) of the regulation (EU) 2019/943 "
                             "of the European Parliament and of the Council of 5 June 2019 on the internal market for "
                             "electricity",
                      '[2]': "Baltic Load-Frequency Control block concept document, 31.12.2020"}
    list_of_figures = {}
    if draw_raw_data_image:
        list_of_figures[input_data_fig_title] = INPUT_DATA_FIGURE_NAME
    list_of_figures[deterministic_fig_title] = DETERMINISTIC_FIGURE_NAME
    list_of_figures[mc_fig_title] = MC_FIGURE_NAME
    list_of_tables = {table_title: overall_report,
                      frce_table_title: frce_report,
                      frce_self_table_title: frce_self_report}
    try:
        report_date = date_of_generation.strftime("%d-%m-%Y")
    except ValueError:
        report_date = pd.Timestamp("today").strftime("%d-%m-%Y")
    report_name = f"report_of_regional_sizing_at_SOR_level_for_Baltics_from_{report_date}.pdf"
    report = generate_report(heading_string=heading,
                             time_ranges=time_range,
                             methodologies=list_of_methodologies,
                             region_list=[region_to_investigate],
                             summaries=list_of_summaries,
                             references=reference_list,
                             images=list_of_figures,
                             tables=list_of_tables,
                             # file_name=report_name
                             date_today=date_of_generation,
                             )
    if report:
        report = BytesIO(report)
        report.name = report_name
        save_report_to_minio(report)
        logger.info(f"Cleaning up")
        delete_file_from_local_storage(INPUT_DATA_FIGURE_NAME)
        delete_file_from_local_storage(DETERMINISTIC_FIGURE_NAME)
        delete_file_from_local_storage(MC_FIGURE_NAME)
    logger.info("Done")
