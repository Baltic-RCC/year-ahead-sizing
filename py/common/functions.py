import pandas as pd
from py.common.elk_handler_reserves import ElkHandlerForReserves, PY_ACEOL_TABLE, VALUE_KEYWORD, NAME_KEYWORD, \
    FROM_KEYWORD, TO_KEYWORD, ACEOL_COLUMNS

DELIMITER = '\t'
NUMBER_OF_SIMULATIONS = 50
NUMBER_OF_SAMPLES = 105000
PERCENTILE_9999 = '99.99%'
PERCENTILE_0001 = '0.01%'
PERCENTILE_9999_VALUE = 0.9999
PERCENTILE_0001_VALUE = 0.0001
TEMP_ACEOL_DATA_FILE = 'elk_query.csv'


def read_csv(file_name: str):
    """
    Temporary workaround, loads data from local files (where it was saved from R workspace, for usage if
    ElasticSearch is down)
    :param file_name: local file name
    :return: data as pandas dataframe
    """
    data_frame = pd.read_csv(file_name, delimiter="\t")
    data_frame.reset_index(drop=True, inplace=True)
    return data_frame


def save_csv(data_frame: pd.DataFrame, file_name: str):
    """
    Saves dataframe for file (to skip data reading-processing steps)
    :param data_frame: data frame to be saved
    :param file_name: file where to be saved
    :return: None
    """
    data_frame.to_csv(file_name, sep=DELIMITER)


def get_raw_aceol_data(elk_instance: ElkHandlerForReserves, aceol_query: dict):
    """
    Queries ACEol data as requested
    :param elk_instance: Instance of the Elk Handler for Reserves (child of Elk api)
    :param aceol_query: dictionary containing query (use dev-tools of ElasticSearch to put it together)
    :return: pandas dataframe with requested data
    """
    return elk_instance.get_data(query=aceol_query,
                                 index=PY_ACEOL_TABLE,
                                 fields=ACEOL_COLUMNS)


def get_aceol_data(elk_instance: ElkHandlerForReserves, aceol_query: dict):
    """
    Requests aceol data and reshapes it to regions (states)
    :param elk_instance: Instance of the Elk Handler for Reserves (child of Elk api)
    :param aceol_query: dictionary containing query (use dev-tools of ElasticSearch to put it together)
    :return: pandas dataframe with columns to-from and states
    """
    aceol_data = get_raw_aceol_data(elk_instance=elk_instance, aceol_query=aceol_query)
    aceol_by_states = pd.pivot_table(aceol_data,
                                     values=VALUE_KEYWORD,
                                     index=[FROM_KEYWORD, TO_KEYWORD],
                                     columns=[NAME_KEYWORD])
    aceol_by_states = aceol_by_states.reset_index(names=[FROM_KEYWORD, TO_KEYWORD])
    return aceol_by_states
