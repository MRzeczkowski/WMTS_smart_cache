import requests
import os
import sqlite3
import re
import matplotlib
import io
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd
import pickle
import shutil
from flask import Flask, request, Response, make_response
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from nerual_network import Nerual_Network

matplotlib.use('SVG')

app = Flask(__name__)

# Sliding Window Length
SWL = 1 * 60

SITE_NAME = 'http://mapcache/'
DATA_FOLDER_NAME = './data/'
DB_NAME = DATA_FOLDER_NAME + 'tiles.db'
GET_TILE_STATISTICS_FILE = DATA_FOLDER_NAME + 'getTileStatistics.csv'
NN_FILE_PATH = DATA_FOLDER_NAME + "NEURAL_NETWORK_PARAMS.pkl"
BASELINE_NN_PATH = DATA_FOLDER_NAME + "BASELINE_NEURAL_NETWORK.pkl"
NEURAL_NETWORK = None
CLF = None

# these MIN and MIN values are used for on the fly normalization
# normalized values are passed to the neural network to make predictions
MIN_SIZE = np.finfo(np.float64).min
MAX_SIZE = np.finfo(np.float64).max

MIN_FREQUENCY = np.finfo(np.float64).min
MAX_FREQUENCY = np.finfo(np.float64).max

MIN_RECENCY = np.finfo(np.float64).min
MAX_RECENCY = np.finfo(np.float64).max


def normalize_df(df, columns):
    y_min = -0.9
    y_max = 0.9

    y_const = y_min - y_max

    result = df.copy()
    for column in columns:
        max_value = df[column].max()
        min_value = df[column].min()
        result[column] = y_min + \
            (y_const * ((df[column] - min_value) / (max_value - min_value)))
    return result


def dump_db_to_dataframe():
    con = sqlite3.connect(DB_NAME)

    result = pd.read_sql_query(
        "SELECT matrix, row, column, size, recency, frequency, cacheability FROM tiles", con)

    con.commit()
    return result


def save_neural_networks_to_files():
    with open(NN_FILE_PATH, "wb") as nn_file:
        pickle.dump(NEURAL_NETWORK_PARAMS, nn_file)
        nn_file.close()

    with open(BASELINE_NN_PATH, "wb") as nn_file:
        pickle.dump(CLF, nn_file)
        nn_file.close()


@app.route('/train')
def trigger_training():
    training_set_file_name = DATA_FOLDER_NAME + 'trainigSet.csv'
    cacheability_column_name = 'cacheability'
    network_input = ['size', 'frequency', 'recency']

    if not os.path.isfile(training_set_file_name):
        if not os.path.isfile(GET_TILE_STATISTICS_FILE):
            response = make_response(
                "\'{0}\' does not exist - cannot train neural network without it".format(GET_TILE_STATISTICS_FILE), 200)
            response.mimetype = "text/plain"
            return response

        df = pd.read_csv(GET_TILE_STATISTICS_FILE)
        working_set_dictionary = {}

        for index, row in df.loc[::-1].iterrows():
            tile_position = row['tile_position']
            current_requested_time = row['requested_time']

            if tile_position in working_set_dictionary:
                working_set_entry = working_set_dictionary[tile_position]

                later_requested_time = working_set_entry[0]

                if later_requested_time - current_requested_time < SWL:
                    df.iloc[working_set_entry[1], df.columns.get_loc(
                        cacheability_column_name)] = 1.0

            working_set_dictionary[tile_position] = [
                current_requested_time, index]

        columns_to_work_on = network_input.copy()

        normalized_set = normalize_df(df, columns_to_work_on)

        columns_to_work_on.append(cacheability_column_name)

        training_set = normalized_set[columns_to_work_on]

        training_sets_dir = DATA_FOLDER_NAME + 'trainingSets'

        if not os.path.isdir(training_sets_dir):
            os.makedirs(training_sets_dir)

        training_set.to_csv(
            '{0}/trainigSet_{1}.csv'.format(training_sets_dir, datetime.now()), index=False)

    else:
        training_set = pd.read_csv(training_set_file_name)

    X = training_set[network_input].values
    y = training_set[cacheability_column_name].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    global NEURAL_NETWORK
    NEURAL_NETWORK = Nerual_Network()

    epochs = int(request.args.get('epochs') or 1000)
    learning_rate = float(request.args.get('learning_rate') or 0.001)
    training_method = request.args.get('method') or 'adam'

    trainingRaport = NEURAL_NETWORK.train(
        np.transpose(X_train),
        np.transpose(y_train.reshape((y_train.shape[0], 1))),
        epochs,
        learning_rate,
        training_method)

    Y_test_hat = NEURAL_NETWORK.full_forward_propagation(np.transpose(X_test))

    acc_test = NEURAL_NETWORK.get_accuracy_value(
        Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))

    trainingRaport += os.linesep + \
        "Custom {0} accuracy: {1}".format(training_method, acc_test)

    global CLF
    CLF = MLPClassifier(
        hidden_layer_sizes=(3, 3),
        activation='logistic',
        max_iter=epochs,
        learning_rate_init=learning_rate,
        solver=training_method,
        nesterovs_momentum=False).fit(X_train, y_train)

    trainingRaport += os.linesep + \
        "Sklearn {0} accuracy: {1}".format(
            training_method, CLF.score(X_test, y_test))

    global NEURAL_NETWORK_PARAMS
    NEURAL_NETWORK_PARAMS = NEURAL_NETWORK.params_values

    save_neural_networks_to_files()

    work_data = normalize_df(dump_db_to_dataframe(), network_input)

    input = np.transpose(work_data[network_input])

    targets = NEURAL_NETWORK.convert_prob_into_class(
        NEURAL_NETWORK.full_forward_propagation(input))

    work_data[cacheability_column_name] = np.transpose(
        targets)

    cacheability_baseline_column_name = 'cacheability_baseline'

    targets_baseline = CLF.predict(work_data[network_input])

    work_data[cacheability_baseline_column_name] = np.transpose(
        targets_baseline)

    rows = work_data[[cacheability_column_name, cacheability_baseline_column_name, 'matrix',
                     'row', 'column']].values

    execute_on_database(lambda cur: cur.executemany(
        "UPDATE tiles SET cacheability = ?, cacheability_baseline = ? WHERE matrix = ? AND row = ? AND column = ?", rows))

    # if os.path.isfile(training_set_file_name):
    #    os.remove(training_set_file_name)

    if os.path.isfile(GET_TILE_STATISTICS_FILE):
        statisticsDir = DATA_FOLDER_NAME + 'statistics'
        if not os.path.isdir(statisticsDir):
            os.makedirs(statisticsDir)

        shutil.move(GET_TILE_STATISTICS_FILE,
                    '{0}/getTileStatistics_{1}.csv'.format(statisticsDir, datetime.now()))

    response = make_response(trainingRaport, 200)
    response.mimetype = "text/plain"
    return response


def extract_tile_info(statitic_type, matrix_id):
    def select(cur): return cur.execute(
        "SELECT row, column, {0} FROM tiles WHERE matrix = ?".format(statitic_type), (matrix_id,)).fetchall()

    return execute_on_database(select)


@app.route('/matrixStatistics/<int:matrix_id>')
def get_matrix_statistics(matrix_id):

    available_statistics_types = [
        'size', 'frequency', 'recency', 'cacheability', 'cacheability_baseline']

    rows = 2
    columns = 3

    fig = plt.figure(figsize=(15, 10))

    image_number = 1

    for statitic_type in available_statistics_types:

        if statitic_type not in available_statistics_types:
            return 'Incorrect statistic type. Chose one of the following: ' + ', '.join(available_statistics_types)

        tile_info = extract_tile_info(statitic_type, matrix_id)

        width = max(map(lambda row: row[0], tile_info)) + 1
        length = max(map(lambda row: row[1], tile_info)) + 1

        data = np.zeros(shape=(width, length))

        for row in tile_info:
            statistic_value = row[2]

            data[row[0], row[1]] = statistic_value

        fig.add_subplot(rows, columns, image_number)
        image_number += 1
        plt.title(statitic_type)

        if length == 1 and width == 1:
            plt.axis('off')

        else:
            max_bins = 12

            max_x_bins = max_bins
            max_y_bins = max_bins

            if width < max_x_bins:
                max_x_bins = width

            if length < max_y_bins:
                max_y_bins = length

            plt.locator_params(axis='x', nbins=max_x_bins)
            plt.locator_params(axis='y', nbins=max_y_bins)

        plt.imshow(data, aspect='equal')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def updateGlobalsStats(size, frequency, recency):
    global MIN_SIZE
    global MAX_SIZE

    global MIN_FREQUENCY
    global MAX_FREQUENCY

    global MIN_RECENCY
    global MAX_RECENCY

    if size < MIN_SIZE:
        MIN_SIZE = size

    if size > MAX_SIZE:
        MAX_SIZE = size

    if frequency < MIN_FREQUENCY:
        MIN_FREQUENCY = frequency

    if frequency > MAX_FREQUENCY:
        MAX_FREQUENCY = frequency

    if recency < MIN_RECENCY:
        MIN_RECENCY = recency

    if recency > MAX_RECENCY:
        MAX_RECENCY = recency


def normalize_value(value, min, max):
    y_min = -0.9
    y_max = 0.9

    y_const = y_max - y_min

    return y_min + (y_const * ((value - min) / (max - min)))


def update_statistics_for_tile(tile_position, tile_size):

    request_unix_timestamp = int(time.time())

    def select(cur): return cur.execute("""SELECT frequency, last_requested_unix_timestamp, cacheability, cacheability_baseline
        FROM tiles 
        WHERE matrix = ? AND row = ? AND column= ?""", tile_position).fetchone()

    last_request_statistics = execute_on_database(select)

    frequency = last_request_statistics[0]
    last_requested_unix_timestamp = last_request_statistics[1]

    time_since_last_request = request_unix_timestamp - last_requested_unix_timestamp

    if time_since_last_request <= SWL:
        frequency += 1
    else:
        frequency = round(
            max(frequency / (time_since_last_request/SWL), 1))

    recency = 0.0

    if last_requested_unix_timestamp == 0:  # first request
        recency = SWL
    else:
        recency = max(SWL, time_since_last_request)

    updateGlobalsStats(tile_size, frequency, recency)

    cacheability = last_request_statistics[2]
    cacheability_baseline = last_request_statistics[3]

    if NEURAL_NETWORK and CLF:

        normalized_size = normalize_value(tile_size, MIN_SIZE, MAX_SIZE)

        normalized_frequency = normalize_value(
            frequency, MIN_FREQUENCY, MAX_FREQUENCY)

        normalized_recency = normalize_value(recency, MIN_RECENCY, MAX_RECENCY)

        input = np.array([normalized_size, normalized_frequency,
                         normalized_recency])
        output = NEURAL_NETWORK.full_forward_propagation(input.reshape(3, 1))
        cacheability = NEURAL_NETWORK.convert_prob_into_class(output).item(0)

        cacheability_baseline = CLF.predict(input.reshape(1, 3)).item(0)

    most_recent_statistics = (tile_size, frequency,
                              recency, cacheability, cacheability_baseline, request_unix_timestamp)

    def update(cur): return cur.execute(
        """UPDATE tiles 
            SET
                size = ?,
                frequency = ?,
                recency = ?,
                cacheability = ?,
                cacheability_baseline = ?,
                last_requested_unix_timestamp = ?
            WHERE matrix = ? AND row = ? AND column= ?""",
        (most_recent_statistics + tile_position))

    execute_on_database(update)

    fileExsists = os.path.isfile(GET_TILE_STATISTICS_FILE)

    with open(GET_TILE_STATISTICS_FILE, 'a', newline=os.linesep) as csvfile:
        fieldnames = ['tile_position', 'requested_time',
                      'size', 'frequency', 'recency', 'cacheability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not fileExsists:
            writer.writeheader()

        writer.writerow(
            {'tile_position': tile_position,
             'requested_time': request_unix_timestamp,
             'size': tile_size,
             'frequency': frequency,
             'recency': recency,
             'cacheability': 0.0})


def run_with_retry(func, retry_count=3, wait_time_seconds=0.2):
    for attempy in range(retry_count):
        try:
            return func()
        except Exception as ex:
            print(ex)
            time.sleep(wait_time_seconds)


@app.route('/<path:path>')
def proxy(path):
    global SITE_NAME

    method = request.method
    new_url = request.url.replace(
        '%2F', '/').replace(request.host_url, SITE_NAME)  # workaround for slash encoding
    headers = {key: value for (key, value) in request.headers if key != 'Host'}
    data = request.get_data()
    cookies = request.cookies

    resp = run_with_retry(
        lambda: requests.request(
            method=method,
            url=new_url,
            headers=headers,
            data=data,
            cookies=cookies,
            allow_redirects=False))

    backend_response_content = resp.content

    new_url_lower = new_url.lower()

    if 'gettile' in new_url_lower:

        matrix = re.search(r'tilematrix=(\d*)', new_url_lower).group(1)
        row = re.search(r'tilerow=(\d*)', new_url_lower).group(1)
        column = re.search(r'tilecol=(\d*)', new_url_lower).group(1)

        update_statistics_for_tile(
            (matrix, row, column), len(backend_response_content))

    if 'getcapabilities' in new_url_lower:
        backend_response_content = backend_response_content.decode().replace(SITE_NAME,
                                                                             request.host_url)

    excluded_headers = ['content-encoding',
                        'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    response = Response(backend_response_content, resp.status_code, headers)
    return response


def execute_on_database(func):
    con = sqlite3.connect(DB_NAME, 10)  # 10s timeout
    cur = con.cursor()

    result = func(cur)

    con.commit()
    cur.close()

    return result


def setup_tiles_database():

    def setup_and_seed_tiles_table(cur):
        cur.execute("""CREATE TABLE tiles(
            matrix INTEGER, 
            row INTEGER, 
            column INTEGER,
            size INTEGER DEFAULT 0,
            recency INTEGER DEFAULT 0,
            frequency REAL DEFAULT 0,
            cacheability REAL DEFAULT 0,
            cacheability_baseline REAL DEFAULT 0,
            last_requested_unix_timestamp INTEGER DEFAULT 0)""")
        cur.execute(
            "CREATE UNIQUE INDEX tile_position ON tiles(matrix, row, column)")

        for matrix in range(11):
            matrix_size = pow(2, matrix)
            for row in range(matrix_size):
                for column in range(matrix_size):
                    tile_position = (matrix, row, column)
                    cur.execute(
                        "INSERT INTO tiles VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0)", tile_position)

    execute_on_database(setup_and_seed_tiles_table)


def get_min_from_db(column_name):
    return execute_on_database(lambda cur: cur.execute("SELECT min({0}) FROM tiles".format(column_name)).fetchone()[0])


def get_max_from_db(column_name):
    return execute_on_database(lambda cur: cur.execute("SELECT max({0}) FROM tiles".format(column_name)).fetchone()[0])


if __name__ == '__main__':
    if not os.path.isfile(DB_NAME):
        setup_tiles_database()
    else:
        MIN_SIZE = get_min_from_db("size")
        MAX_SIZE = get_max_from_db("size")

        MIN_FREQUENCY = get_min_from_db("frequency")
        MAX_FREQUENCY = get_max_from_db("frequency")

        MIN_RECENCY = get_min_from_db("recency")
        MAX_RECENCY = get_max_from_db("recency")

    if os.path.isfile(NN_FILE_PATH):
        with open(NN_FILE_PATH, "rb") as nn_file:
            NEURAL_NETWORK_PARAMS = pickle.load(nn_file)
            nn_file.close()

        NEURAL_NETWORK = Nerual_Network(params_values=NEURAL_NETWORK_PARAMS)

    if os.path.isfile(BASELINE_NN_PATH):
        with open(BASELINE_NN_PATH, "rb") as nn_file:
            CLF = pickle.load(nn_file)
            nn_file.close()

    app.run(debug=False, host="0.0.0.0", port=80)
