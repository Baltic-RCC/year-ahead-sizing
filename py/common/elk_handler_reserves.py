import sys
import time
import pandas as pd

import config
from py.config_parser import parse_app_properties

try:
    from rcc_common_tools.elk_api import Elastic
except ModuleNotFoundError:
    from elasticsearch import Elasticsearch

    class Elastic:

        def __init__(self, server: str, debug: bool = False):
            self.server = server
            self.debug = debug
            self.client = Elasticsearch(self.server)


from datetime import datetime

INITIAL_SCROLL_TIME = "5m"
CONSECUTIVE_SCROLL_TIME = "2m"
FIELD_NAME = "hits"
DOCUMENT_COUNT = 10000
DEFAULT_COLUMNS = ["value"]
SCROLL_ID_FIELD = '_scroll_id'
RESULT_FIELD = 'hits'
DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
MAGIC_KEYWORD = '_source'
COUNT_DOCUMENTS = 'count'
MAPPINGS_KEYWORD = 'mappings'
PROPERTIES_KEYWORD = 'properties'

parse_app_properties(globals(), config.paths.config.elastic)
PY_ELASTICSEARCH_HOST = ELASTICSEARCH_HOST
PY_ACEOL_TABLE = ACEOL_TABLE
PY_RESERVE_INPUT_TABLE = RESERVE_INPUT_TABLE

VALUE_KEYWORD = 'value'
NAME_KEYWORD = 'name'
FROM_KEYWORD = 'from'
TO_KEYWORD = 'to'
ACEOL_COLUMNS = [FROM_KEYWORD, TO_KEYWORD, NAME_KEYWORD, VALUE_KEYWORD]
RESERVE_COLUMNS = [FROM_KEYWORD, TO_KEYWORD, NAME_KEYWORD, VALUE_KEYWORD]


class ElkHandlerForReserves(Elastic):
    """
    For handling get and post to ELK for calculation of the reserves
    """

    def __init__(self,
                 server,
                 initial_scroll_time: str = INITIAL_SCROLL_TIME,
                 consecutive_scroll_time: str = CONSECUTIVE_SCROLL_TIME,
                 document_count: str = DOCUMENT_COUNT,
                 field_name: str = FIELD_NAME,
                 date_time_format: str = DATE_TIME_FORMAT,
                 debug: bool = False):
        """

        :param server: address of elk server
        :param initial_scroll_time: for scrolling initial value
        :param consecutive_scroll_time: for each consecutive scroll
        :param document_count: number of rows to be extracted
        :param field_name: where results are located
        :param date_time_format: date time format used by elasticsearch
        :param debug: if needed
        """
        self.initial_scroll_time = initial_scroll_time
        self.consecutive_scroll_time = consecutive_scroll_time
        self.document_count = document_count
        self.field_name = field_name
        self.date_time_format = date_time_format
        super().__init__(server=server, debug=debug)

    def get_elk_formatted_date_time(self, date_time_str: str):
        """
        Converts date time string to the format used by elasticsearch
        :param date_time_str: input datetime string
        :return: formatted datetime string
        """
        return datetime.strptime(date_time_str, self.date_time_format).isoformat()

    def get_document_count(self, index: str, query: dict = None):
        """
        Counts number of documents found in the index (by using the query)
        :param index: table where search from
        :param query: optional the query by which to search from
        :return: number of documents
        """
        if query is None:
            results = self.client.count(index=index)
        else:
            results = self.client.count(index=index, query=query)
        return results[COUNT_DOCUMENTS]

    def get_shard_count(self, index: str):
        """
        Returns number of shards
        :param index: table for which shards are needed
        :return: number of shards
        """
        results = self.client.search_shards(index=index)
        return len(results["shards"])

    def get_index_fields(self, index: str):
        """
        Returns list of fields
        :param index: table name
        :return: list of fields(columns) in the index
        """
        response = self.client.indices.get_mapping(index=index)
        first_index = next(iter(response.values()))
        if MAPPINGS_KEYWORD in dict(first_index).keys() and first_index[MAPPINGS_KEYWORD] is not None:
            if (PROPERTIES_KEYWORD in dict(first_index[MAPPINGS_KEYWORD]) and
                    first_index[MAPPINGS_KEYWORD][PROPERTIES_KEYWORD] is not None):
                return list(dict(first_index[MAPPINGS_KEYWORD][PROPERTIES_KEYWORD]).keys())
        return None

    def get_data_by_scrolling(self,
                              query: dict,
                              index: str,
                              fields: [] = None):
        """
        Asks data from elk by scrolling
        Rework this: allocate memory to data structure beforehand and then start writing into it
        :param query: dictionary for asking
        :param index: index (table) from where to ask
        :param fields: fields (columns) to ask from index (table)
        :return:
        """
        if fields is None:
            result = self.client.search(index=index,
                                        query=query,
                                        size=self.document_count,
                                        scroll=self.initial_scroll_time)
        else:
            result = self.client.search(index=index,
                                        query=query,
                                        source=fields,
                                        size=self.document_count,
                                        scroll=self.initial_scroll_time)

        scroll_id = result[SCROLL_ID_FIELD]
        # Extract and return the relevant data from the initial response
        hits = result[RESULT_FIELD][RESULT_FIELD]
        yield hits

        # Continue scrolling through the results until there are no more
        while hits:
            result = self.client.scroll(scroll_id=scroll_id, scroll=self.consecutive_scroll_time)
            hits = result[RESULT_FIELD][RESULT_FIELD]

            yield hits
        # Clear the scroll context after processing all results
        self.client.clear_scroll(scroll_id=scroll_id)

    def get_data(self,
                 query: dict,
                 index: str,
                 fields: [] = None):
        """
        Asks data from elk and stores it to numpy array
        For composing the query use https://test-rcc-logs.elering.sise/app/dev_tools#/console
        :param query: dictionary to query from elk
        :param index: index (table) from where to query
        :param fields: fields (columns) to be extracted, note that these must be strings
        :return: pandas dataframe with fields as columns
        """
        # Get number of documents
        row_count = self.get_document_count(index=index, query=query)
        # Get columns if not specified
        if fields is None:
            fields = self.get_index_fields(index=index)
        self.sys_print(f"Reading {index}:{row_count} documents found")
        counter = 1
        timer_start = time.time()
        # Gather all the results to list (of dictionaries)
        list_of_lines = []
        for hits in self.get_data_by_scrolling(query, index, fields):
            # progress_val = counter * DOCUMENT_COUNT * 100 / row_count
            # if progress_val < 100:
            #     self.sys_print(f"\rReading {index}: {progress_val:.2f}% completed ")
            counter += 1
            for hit in hits:
                list_of_lines.append({field: hit[MAGIC_KEYWORD][field] for field in fields})
        # convert list (of dictionaries) to pandas dataframe
        data_frame = pd.DataFrame(list_of_lines)
        timer_stop = time.time()
        self.sys_print(f"\rReading {index}: done with {(timer_stop - timer_start):.2f}s for request\n")
        return data_frame

    def get_data_from_shard(self, query: dict, index: str, fields: [] = None):
        pass

    def get_data_by_shards(self, query: dict, index: str, fields: [] = None):
        """
        Queries data from the shards available in parallel
        :param query: dictionary to query from elk
        :param index: index (table) from where to query
        :param fields: fields (columns) to be extracted, note that these must be strings
        :return: pandas dataframe with fields as columns
        """
        # shards = self.get_shard_count(index=index)
        pass

    def post_data(self):
        pass

    @staticmethod
    def sys_print(message: str):
        sys.stdout.write(message)
        sys.stdout.flush()


if __name__ == '__main__':
    elk_instance = ElkHandlerForReserves(server=PY_ELASTICSEARCH_HOST)
    start_date_elk = elk_instance.get_elk_formatted_date_time("2023-01-01T00:00:00")
    end_date_elk = elk_instance.get_elk_formatted_date_time("2024-01-01T00:00:00")
    additional_value_name = "Baltics"
    columns = DEFAULT_COLUMNS

    aceol_query = {
        "range": {
            "from": {
                "gte": "2023-01-01T00:00:00",
                "lte": "2024-01-01T00:00:00"
            }
        }
    }
    aceol_values = elk_instance.get_data(query=aceol_query,
                                         index=PY_ACEOL_TABLE,
                                         fields=ACEOL_COLUMNS)
    print(aceol_values)

    reserve_input_query = {
        "range": {
            "from": {
                "gte": "2023-01-01T00:00:00",
                "lte": "2024-01-01T00:00:00"
            }
        }
    }
