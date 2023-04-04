# Databricks notebook source
# MAGIC %md
# MAGIC Install the <a href="https://github.com/dparrini/python-comtrade">`python-comtrade`</a>, <a href="https://github.com/ijl/orjson">`orjson`</a>, <a href="https://github.com/relihanl/comtradehandlers">`ComtradeHandlers`</a> libraries via pip.

# COMMAND ----------

# MAGIC %pip install comtrade orjson fsspec s3fs git+https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

from comtradehandlers import writer
import comtrade
from comtrade import Comtrade

from pathlib import Path
from random import randint
import uuid
import pandas as pd
import io
import pickle
import numpy as np
import datetime
from typing import Iterator, List
import orjson
import matplotlib.pyplot as plt
import os

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, StringType, StructType, StructField, ArrayType, DoubleType, MapType, TimestampType, IntegerType, LongType
from joblib import Parallel, delayed

import tensorflow as tf
import mlflow
from mlflow.keras import log_model

# COMMAND ----------

# MAGIC %md
# MAGIC Turn off mlflow logging so that we can manually log and transition our ML model.

# COMMAND ----------

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Constants

# COMMAND ----------

# MAGIC %md
# MAGIC #### Paths

# COMMAND ----------

# Filestore, def. want to change this
# FILESTORE_PATH = "colton"
# TABLE_NAME = "edg_ieee_transients"

# COMTRADE_FILES_PATH = f"/FileStore/tables/{FILESTORE_PATH}/{TABLE_NAME}"
S3_COMTRADE_DATA_PATH = "s3://db-gtm-industry-solutions/data/rcg/comtrade/transient disturbances/transient disturbances/"
SAVE_PATH_PREFIX = "/dbfs/FileStore/tables/colton/edg_s3_comtrade_files"

# Database and table names
SCHEMA_NAME = "comtrade_db"
FINAL_OUTPUT_TABLE = f"{SCHEMA_NAME}.pivoted_current"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Config and datafile attributes

# COMMAND ----------

FILENAME = "filename"
DIRECTORY = "directory"
CONTENT = "content"
CONTENTS_CFG = "config_content"
CONTENTS_DAT = "dat_content"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output columns from UDF

# COMMAND ----------

TIME_MILLIS = "time_millis"
TIME_MICRO = "microseconds"
VALUE = "value"
CHANNEL_ID = "channel_id"
CHANNEL_TYPE = "channel_type"
ANALOG = "analog"
ANALOG_UNITS = "analog_units"
STATUS = "status"
STATION_NAME = "station_name"
REC_DEV_ID = "rec_dev_id"
REV_YEAR = "rev_year"
FREQ = "frequency"
PHASE = "phase"
ANALOG_CHANNEL_NAMES = "analog_channel_names"
STATUS_CHANNEL_NAMES = "status_channel_names"

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Phase attributes

# COMMAND ----------

CURRENT = "I"
AMPS = "A"
A_PHS = "A"
B_PHS = "B"
C_PHS = "C"
IA = CURRENT + A_PHS
IB = CURRENT + B_PHS
IC = CURRENT + C_PHS
TIME = "time"

# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Data
# MAGIC The dataset in the s3 path is from <a href="https://ieee-dataport.org/open-access/transients-and-faults-power-transformers-and-phase-angle-regulators-dataset">*"Transients and Faults in Power Transformers and Phase Angle Regulators”* dataset from the IEEE Dataport</a>. You can download it directly from the link by creating a free IEEE account first. Otherwise, you can use the files in our s3 bucket.
# MAGIC 
# MAGIC The dataset consists of a few thousand simulated examples of various power quality transient events. They are formatted as flat text files. In order to showcase the solution acclerator, we'll need to convert them to COMTRADE files using the installed `ComtradeHandlers` library.

# COMMAND ----------

# getting the files located in s3
def deep_ls(path: str):
    # https://stackoverflow.com/questions/67601129/databricks-dbutils-get-filecount-and-filesize-of-all-subfolders-in-azure-data
    """List all files in base path recursively."""
    for x in dbutils.fs.ls(path):
        if x.path[-1] != "/":
            yield x
        else:
            for y in deep_ls(x.path):
                yield y

# going to hand wave this for now                
all_txt_files = [p.path for p in list(deep_ls(S3_COMTRADE_DATA_PATH))]

# COMMAND ----------

# MAGIC %md
# MAGIC If necessary, create a new directory to store the newly created COMTRADE files.

# COMMAND ----------

def file_exists(dir):
  try:
    dbutils.fs.ls(dir)
  except:
    return False  
  return True

# COMMAND ----------

if file_exists(SAVE_PATH_PREFIX.replace("/dbfs",  "dbfs:")):
  print("This directory exists already.")
else:
  dbutils.fs.mkdirs(SAVE_PATH_PREFIX.replace("/dbfs",  "dbfs:"))
  print("Directory created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC Read the data from s3 and use the `ComtradeHandlers` library to write out the data as comtrade files in local dbfs.

# COMMAND ----------

def write_comtrade_file(output_file_prefix : str, station_name : str, rec_dev_id : int, input_data : pd.DataFrame) -> None:
    start_time = datetime.datetime.now()
    trigger_time = start_time + datetime.timedelta(milliseconds=20)
    
    # Create comtrade writer object
    comtradeWriter = writer.ComtradeWriter(
      f"{SAVE_PATH_PREFIX}/{output_file_prefix}.cfg",
      start_time, 
      trigger_time, 
      station_name = station_name, 
      rec_dev_id=rec_dev_id
    )
    # A Current
    comtradeWriter.add_analog_channel(IA, A_PHS, CURRENT, uu=AMPS, skew=0, min=-500, max=500, primary=1, secondary=1)
    
    # B Current
    comtradeWriter.add_analog_channel(IB, B_PHS, CURRENT, uu=AMPS, skew=0, min=-500, max=500, primary=1, secondary=1)
    
    # C Current
    comtradeWriter.add_analog_channel(IC, C_PHS, CURRENT, uu=AMPS, skew=0, min=-500, max=500, primary=1, secondary=1)
    
    # Write the pandas dataframe out into the comtrade file
    for i in range(input_data.shape[0]):
        row = input_data.iloc[i]
        comtradeWriter.add_sample_record(int(row[TIME]),[row[IA], row[IB], row[IC]],[])
    comtradeWriter.finalize() # This writes the final file out

def read_txt_file_to_pd(file_path : str) -> pd.DataFrame:
    txt_pd = pd.read_csv(file_path, header=None, names=[TIME, IA, IB, IC])
    txt_pd[TIME] = (txt_pd[TIME] * 1e6).astype("int")
    return txt_pd.astype("str")

# COMMAND ----------

# MAGIC %md
# MAGIC Process the txt files into COMTRADE files. This is using parallelism (which you can configure in the `n_jobs` argument below) to speed up the process, but this can still take quite some time!

# COMMAND ----------

def txt_file_generator(gen_txt_files):
    _all_txt = gen_txt_files
    _n = 0
    while _n < len(_all_txt):
        yield _all_txt[_n]
        _n += 1

def process_file(txt_file_path):
    _path_prefix = all_txt_files[0].split("/")[-1].split(".")[0] # get the filename with no extension or path.
    uuid_suffix = str(uuid.uuid4()).split("-")[-1]
    _tmp = read_txt_file_to_pd(txt_file_path)
    write_comtrade_file(_path_prefix + uuid_suffix, _path_prefix, randint(0,9999), _tmp)
    print(f"Processing: {txt_file_path}")

Parallel(n_jobs=16, prefer="threads")(delayed(process_file)(txt_file) for txt_file in txt_file_generator(all_txt_files))

# COMMAND ----------

# MAGIC %md
# MAGIC Load and plot an example waveform

# COMMAND ----------

example_path = dbutils.fs.ls(SAVE_PATH_PREFIX.replace("/dbfs",  "dbfs:"))[0].path.split(".")[0].replace("dbfs:","/dbfs")

rec = Comtrade()
rec.load(f"{example_path}.cfg", f"{example_path}.dat")

plt.plot(rec.analog[0])
plt.plot(rec.analog[1])
plt.plot(rec.analog[2])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read COMTRADE files into Spark
# MAGIC 
# MAGIC * Read COMTRADE `.cfg` and `.dat` files as binary 
# MAGIC * Convert the `.cfg` file contents to string. The `.dat` file contents are left as binary because some `.dat` files will be binary type while others will be ASCII strings. We don't know for each file yet which type it is until we examine the associated `.cfg` file.
# MAGIC * Join together on the filename.

# COMMAND ----------

configs = (
    spark
    .read
    .format("binaryFile")
    .option("pathGlobFilter", "*.cfg")
    .load(SAVE_PATH_PREFIX.replace("/dbfs", "dbfs:"))
    .withColumn(FILENAME, F.element_at(F.split(F.input_file_name(),"\."),1))
    .withColumn(CONTENTS_CFG, F.col(CONTENT).cast("string"))
    .drop("path",CONTENT,"length")
)
display(configs)

# COMMAND ----------

data_files = (
    spark
    .read
    .format("binaryFile") # Load data as binary
    .option("pathGlobFilter", "*.dat") # look for the .dat file extension
    .load(SAVE_PATH_PREFIX.replace("/dbfs", "dbfs:"))
    .withColumn(FILENAME, F.element_at(F.split(F.input_file_name(),"\."),1)) # split the filepath on the "." and only keep the first second (e.g. "/path/meter1.dat" ==> "/path/meter1")
    .withColumnRenamed(CONTENT,CONTENTS_DAT) # Rename the content column
    .drop("path","length") # Drop unneeded columns
)
display(data_files)

# COMMAND ----------

# MAGIC %md
# MAGIC Join the `configs` and `data_files` DataFrames on the parsed `FILENAME` column.

# COMMAND ----------

joined_comtrade = (
    configs
    .join(data_files.select(FILENAME, CONTENTS_DAT), on=FILENAME, how="inner")
)
display(joined_comtrade)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the `comtrade` library to parse file pairs
# MAGIC 
# MAGIC The `comtrade` library is typically used by creating a new `comtrade.Comtrade` object and calling `.load()` and specifying the path to the `.cfg` and `.dat` files:
# MAGIC 
# MAGIC ```
# MAGIC rec = Comtrade()
# MAGIC rec.load("sample_files/sample_ascii.cfg", "sample_files/sample_ascii.dat")
# MAGIC ```
# MAGIC 
# MAGIC However, in our case, we want to use the power of distributed computing with Spark to read all of the files and process chunks of Comtrade pairs in parallel.
# MAGIC 
# MAGIC The `comtrade.Comtrade` object also exposes a `.read()` function which accepts StringIO or ByteIO objects directly, but it assumes that the `.dat` file format has been read the right type (string or bytes object). We don't know yet for each file if it should be read as a String or Binary object, so we have read them all as Binary objects. In this case, we need to write a new `.read_comtrade_dynamic` function which accepts a bytes object and uses the `.cfg` contents to determine the proper formatting of the `.dat` contents.

# COMMAND ----------

def read_comtrade_dynamic(cfg : str, dat : bytes) -> comtrade.Comtrade:
    # NEW : Create new instance of comtrade.Comtrade
    ct = comtrade.Comtrade()

    # These lines are the same as Comtrade.read():
    ct._cfg.read(cfg)
    ct._cfg_extract_channels_ids(ct._cfg)
    ct._cfg_extract_phases(ct._cfg)
    dat_proc = ct._get_dat_reader()
    
    # NEW : Add the following line to dynamically determine if the bytes object should be converted to a string with .decode()
    dat = dat.decode() if (ct.ft == "ASCII") else dat
    
    # Below lines the same as Comtrade.read()
    dat_proc.read(dat,ct._cfg)
    ct._dat_extract_data(dat_proc)
    
    # NEW : return the comtrade.Comtrade object.
    return ct

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have created the `Comtrade` object we need to retrieve from it the information desired. Below are the information desired and how they are represented in the `Comtrade` object:
# MAGIC 
# MAGIC * Analog Channels - available in `Comtrade.analog` as a `List[List[float]]` with one list for each Analog channel
# MAGIC * Analog Units - not exposed directly, must read from internal `comtrade.Channel.uu` objects: `comtrade.Cfg.analog_channels.uu`
# MAGIC * Status Channels - available in `Comtrade.status` as a `List[List[float]]` with one list for each Status channel
# MAGIC * Analog Channel Names - available in `Comtrade.analog_channel_ids` which is a `List[str]`
# MAGIC * Status Channel Names - available in `Comtrade.status_channel_ids` which is a `List[str]`
# MAGIC * Frequency - available in `Comtrade.frequency` as a `float`
# MAGIC * Recording Device ID - available as a `str` in `Comtrade.rec_dev_id`
# MAGIC * Station Name - available as a `str` in `Comtrade.station_name`
# MAGIC * Sample Timestamps - available by adding the `datetime.datetime` `Comtrade.start_timestamp` object to the fractional seconds in `Comtrade.time` which is of type `List[float]`
# MAGIC 
# MAGIC Since this will all be evaluated within a `pandas_udf` from Spark it will be most efficient to return this as a `BinaryType` JSON object using `orjson.dumps`

# COMMAND ----------

def retrieve_dict(cfg : str, dat : bytes) -> str:
    # Pass in the config file string and data file binary contents to create a comtrade.Comtrade object
    _comtrade = read_comtrade_dynamic(cfg, dat)

    # How many analog and status channels exist?
    analog_count = _comtrade.analog_count
    status_count = _comtrade.status_count
    
    # get the starting time in milliseconds
    start_millis = int(_comtrade.start_timestamp.timestamp())
    
    # get the starting time microseconds
    start_micros = _comtrade.start_timestamp.microsecond
    
    # initialize an empty dicitonary
    ret = {}
    
    # Initialize empty analog return values
    _analog_list = []
    _analog_units = []
    _analog_channel_names = []
    
    if (analog_count > 0):
        # Use numpy.vstack to stack the analog channels into an np.array, transpose it, then convert it to a List[List[float]] object. 
        # where each list corresponds to the value of all analog channels at a specify timestamp.
        _analog_list = np.vstack(_comtrade.analog).transpose().tolist()
        _analog_channels = [channel.uu for channel in _comtrade._cfg.analog_channels]
        _analog_channel_names = _comtrade.analog_channel_ids
    ret[ANALOG] = _analog_list
    ret[ANALOG_UNITS] = _analog_units
    ret[ANALOG_CHANNEL_NAMES] = _analog_channel_names
    
    # Initialize empty status return values
    _status_list = []
    _status_channel_names = []
    
    if (status_count > 0):
        # get the value of each status channel at each timestamp
        _status_list = np.vstack(_comtrade.status).transpose().tolist()
        _status_channel_names = _comtrade.status_channel_ids
    ret[STATUS] = _status_list
    
    # get the frequency, rec_dev_id, and station name
    ret[FREQ] = _comtrade.frequency
    ret[REC_DEV_ID] = _comtrade.rec_dev_id
    ret[STATION_NAME] = _comtrade.station_name
    
    # Get microseconds since epoch
    ret[TIME_MICRO] = [int(_comtrade.start_timestamp.timestamp()) + int(second * 1e6) for second in _comtrade.time]
    
    # Dump the dictionary to a binary string.
    return orjson.dumps(ret)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we'll create a `pandas_udf` function which can be called from Spark on chunks of rows in the `joined_comtrade` DataFrame. Remember, each row corresponds to one `.cfg` and `.dat` pair. This `pandas_udf` will need to take the `config_content` and `dat_content` columns as input and output a `BinaryType` JSON string which will come from the `retrieve_dict` function we have created above.

# COMMAND ----------

@pandas_udf(BinaryType())
def get_comtrade_json(cfg_series : pd.Series, dat_series : pd.Series) -> pd.Series:
    # Put the two pandas series into a pandas DataFrame
    _df = pd.DataFrame({"cfg" : cfg_series, "dat" : dat_series})
    # For every row, apply the retrieve_dict function to get the binary json string
    return _df.apply(lambda row : retrieve_dict(row["cfg"], row["dat"]),1)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the results. Depending on the size of your dataset and size of your Spark cluster the amount of time for this will vary.

# COMMAND ----------

json_retrieved = (
    joined_comtrade
    .withColumn("binary_json", get_comtrade_json(F.col("config_content"), F.col("dat_content")))
)
#json_retrieved.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parsing the JSON
# MAGIC Great! Now we have some binary JSON, but how can we use it? First we'll define a Spark schema, then we'll use Spark's `from_json` function to parse the JSON.

# COMMAND ----------

json_schema = StructType([
    StructField(ANALOG, ArrayType(ArrayType(DoubleType()))),
    StructField(ANALOG_UNITS, ArrayType(StringType())),
    StructField(STATUS, ArrayType(ArrayType(DoubleType()))),
    StructField(ANALOG_CHANNEL_NAMES, ArrayType(StringType())),
    StructField(STATUS_CHANNEL_NAMES, ArrayType(StringType())),
    StructField(FREQ, DoubleType()),
    StructField(REC_DEV_ID, StringType()),
    StructField(STATION_NAME, StringType()),
    StructField(TIME_MICRO, ArrayType(LongType()))
])

# COMMAND ----------

proc = (
    json_retrieved
    .withColumn("parsed", F.from_json(F.col("binary_json").cast("string"), json_schema))
    .select(FILENAME, "parsed.*")
)
#proc.show()

# COMMAND ----------

proc.show()

# COMMAND ----------

# MAGIC %md
# MAGIC In this case all of our COMTRADE files define captures with only 3 analog channels: IA, IB, and IC. In many cases different COMTRADE files may have many very different channel names and some data engineering efforts will be required to standardize those channels before pivoting the data in the following steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pivoting the Data
# MAGIC We'd like to get columns for IA, IB, IC, the timestamp columns, and keep the metadata columns.
# MAGIC First, let's create a dataframe of just the metadata, which we'll rejoin to the larger dataset at the end using `FILENAME` as a key.

# COMMAND ----------

metadata_cols = [FREQ, STATION_NAME, REC_DEV_ID]
metadata_df = (
    proc
    .select(FILENAME, *metadata_cols)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we use `arrays_zip` and `explode` to get the dataframe flattened.

# COMMAND ----------

# Name some temporary columns for intermediary transformations
_ARRAY_COLS = "_array_cols"
_STRUCT_COL = "_struct_col"
_ANALOG_CHANNELS_PER_TS = "_analog_channels_per_timestamp"
_ANALOG_CHANNEL = "_analog_channel"

WILDCARD = "*"


array_columns_to_bundle = [TIME_MICRO, ANALOG] # Include status here too, if you need status channels
unneeded_columns = [ANALOG_UNITS, STATUS, STATUS_CHANNEL_NAMES]
pivoted_current = (
    proc
    .drop(*metadata_cols, *unneeded_columns)
    .select(
        WILDCARD,
        F.arrays_zip(*array_columns_to_bundle).alias(_ARRAY_COLS)
    )
    .drop(*array_columns_to_bundle)
    .select(WILDCARD,F.explode(_ARRAY_COLS).alias(_STRUCT_COL))
    .drop(_ARRAY_COLS)
    .select(WILDCARD,f"{_STRUCT_COL}.{WILDCARD}")
    .drop(_STRUCT_COL)
    .withColumn(_ANALOG_CHANNELS_PER_TS, F.arrays_zip(ANALOG_CHANNEL_NAMES,ANALOG))
    .drop(ANALOG_CHANNEL_NAMES,ANALOG)
    .select(WILDCARD, F.explode(_ANALOG_CHANNELS_PER_TS).alias(_ANALOG_CHANNEL))
    .drop(_ANALOG_CHANNELS_PER_TS)
    .select(WILDCARD, f"{_ANALOG_CHANNEL}.{WILDCARD}")
    .drop(_ANALOG_CHANNEL)
    .filter(F.col(ANALOG_CHANNEL_NAMES).isin([IA, IB, IC])) # This will ensure we only keep these three variables
    .groupby(FILENAME, TIME_MICRO)
    .pivot(ANALOG_CHANNEL_NAMES,[IA, IB, IC]) # If you don't know all the possible column names this second argument can be left blank but the computation time will take longer.
    .agg(F.first(ANALOG))
)

# COMMAND ----------

display(pivoted_current)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's plot an example waveform

# COMMAND ----------

fn_to_examine = pivoted_current.select(FILENAME).distinct().limit(1).toPandas()[FILENAME].iloc[0]
examine = pivoted_current.filter(F.col(FILENAME) == fn_to_examine).toPandas()
sorted_examine = examine.sort_values([TIME_MICRO]).reset_index(drop=True)

# COMMAND ----------

# Create a plot of the waveform
plt.plot(sorted_examine[IA])
plt.plot(sorted_examine[IB])
plt.plot(sorted_examine[IC])
plt.suptitle(sorted_examine[FILENAME].iloc[0])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model training

# COMMAND ----------

PHASE_A = "IA"
PHASE_B = "IB"
PHASE_C = "IC"
IS_FAULT = "is_fault"

# COMMAND ----------

raw_schema = StructType([
    StructField(TIME, DoubleType()),
    StructField(PHASE_A, DoubleType()),
    StructField(PHASE_B, DoubleType()),
    StructField(PHASE_C, DoubleType())
])

df = (
    spark
    .read
    .schema(raw_schema)
    .option("recursiveFileLookup",True)
    .csv(S3_COMTRADE_DATA_PATH)
    .withColumn("filename", F.input_file_name())
    .withColumn(IS_FAULT, F.col("filename").rlike("external").cast("int"))
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Since it isn't too large, pull the dataset into memory as a Pandas DataFrame.

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the Pandas DataFrame into a tensor of shape:
# MAGIC 
# MAGIC Data:
# MAGIC (Number of waveforms, length of waveforms, number of channels)
# MAGIC 
# MAGIC Labels:
# MAGIC (Number of waveforms, 1 if fault 0 otherwise)

# COMMAND ----------

WAVEFORM_LENGTH = 726

# Select all rows for the 3 phase current
current_signals = df_pd.loc[:, [PHASE_A, PHASE_B, PHASE_C]]
# Use signal.frame to convert to (N, 726, 3)
current_signals_tensor = tf.signal.frame(current_signals.to_numpy(), WAVEFORM_LENGTH, WAVEFORM_LENGTH, axis=0)
# Use signal.frame to create label tensor of shape (N, 1)
labels_tensor = tf.signal.frame(df_pd[IS_FAULT].iloc[::WAVEFORM_LENGTH].to_numpy(), 1, 1)
# Print the shapes
print(current_signals_tensor.shape, labels_tensor.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Before splitting into training and test sets we need to shuffle the order of the waveforms (don't shuffle the time series of the waveforms themselves!)

# COMMAND ----------

tf.random.set_seed(42)

# We're interested in shuffling on the 0th axis (N waveforms), get the range of indices
indices = tf.range(start=0, limit=tf.shape(current_signals_tensor)[0], dtype=tf.int32)
# Shuffle the indices
shuffled_indices = tf.random.shuffle(indices)

# Use gather to shuffle the data tensor and the labels tensor in the same order (so that the Nth waveform still corresponds to the Nth label)
shuffled_signals = tf.gather(current_signals_tensor, shuffled_indices)
shuffled_labels = tf.gather(labels_tensor, shuffled_indices)

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate how many waveforms should be in the training, validation, and test sets based upon a 70/15/15 split.

# COMMAND ----------

total_waveforms = shuffled_signals.shape[0]
train_waveforms = int(total_waveforms * 0.7)
val_waveforms = int((total_waveforms - train_waveforms) * 0.5)
test_waveforms = total_waveforms - val_waveforms - train_waveforms

print(train_waveforms, val_waveforms, test_waveforms)

# COMMAND ----------

# MAGIC %md
# MAGIC Use indexing to split the waveforms into the training/validation/test sets.

# COMMAND ----------

train_signals = shuffled_signals[:train_waveforms]
train_labels = shuffled_labels[:train_waveforms]

val_signals = shuffled_signals[train_waveforms:train_waveforms + val_waveforms]
val_labels = shuffled_labels[train_waveforms:train_waveforms + val_waveforms]

test_signals = shuffled_signals[-test_waveforms:]
test_labels = shuffled_labels[-test_waveforms:]

print(train_signals.shape[0], val_signals.shape[0], test_signals.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC Create a function which generates a new convolutional neural network for binary classification.

# COMMAND ----------

def create_convolutional_classification_model() -> tf.keras.Model:
    tf.random.set_seed(13)
    inp = tf.keras.Input(shape=[WAVEFORM_LENGTH,3])
    pipe = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same") (inp)
    pipe = tf.keras.layers.MaxPooling1D(pool_size=4) (pipe)
    pipe = tf.keras.layers.Conv1D(32,3, activation="relu", padding="same") (pipe)
    pipe = tf.keras.layers.MaxPooling1D(pool_size=4) (pipe)
    pipe = tf.keras.layers.Flatten() (pipe)
    pipe = tf.keras.layers.Dropout(0.5) (pipe)
    pipe = tf.keras.layers.Dense(1, activation="sigmoid") (pipe)

    mod = tf.keras.Model(inp,pipe)
    mod.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["Precision","Recall"])
    return mod

# COMMAND ----------

# MAGIC %md
# MAGIC Fit the classification model to the training set. We should attempt to overfit to find the right number of epochs to use for a final fit.

# COMMAND ----------

model1 = create_convolutional_classification_model()

model1_history = model1.fit(
  x=train_signals, 
  y=train_labels, 
  batch_size=16, 
  epochs=64, 
  validation_data=(val_signals,val_labels)
)

plt.plot(model1_history.history["loss"], label="loss")
plt.plot(model1_history.history["val_loss"], label="val_loss")
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like we overfit to the data very quickly after around epoch 10. We could add more regularization to slow the overfit, but in our case this model is a good enough start so we'll concatenate the training and validation sets and retrain to 10 epochs.

# COMMAND ----------

train_val_signals = tf.concat([train_signals, val_signals],axis=0)
train_val_labels = tf.concat([train_labels, val_labels],axis=0)

model2 = create_convolutional_classification_model()
model2_history = model2.fit(
  x=train_val_signals, 
  y=train_val_labels, 
  batch_size=16, 
  epochs=10
)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate our final model against the test data set.

# COMMAND ----------

test_metrics = model2.evaluate(x=test_signals,y=test_labels)
print(test_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC Log the model to mlflow and transition it into Production so we can easily load it in our pipeline.

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.log_metric("test_loss", test_metrics[0])
    mlflow.log_metric("test_precision", test_metrics[1])
    mlflow.log_metric("test_recall", test_metrics[2])
    log_model(model2, artifact_path="model")

register_info = mlflow.register_model(f"runs:/{run.info.run_id}/model", "fault_detection")

client = mlflow.MlflowClient()
client.transition_model_version_stage(name = "fault_detection", version=int(register_info.version), stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC # What would a data pipeline look like?
# MAGIC Saving the data should be done in the <a href="https://www.databricks.com/glossary/medallion-architecture">medallion architecture</a>. As such the flow would look like:
# MAGIC * Saving the raw binary reads of the `.cfg` and `.dat` files into bronze tables.
# MAGIC * Joining the previous two tables on their filenames and saving as a bronze table.
# MAGIC * Use the defined `pandas_udf` to extract binary JSON, parse the JSON, and store it in a silver table.
# MAGIC * Create silver tables for the flattened data and metadata from the parsed JSON.
# MAGIC * Perform filtering for the desired columsn and pivot the data into a silver table.
# MAGIC * Use a trained tensorflow model to detect faults in the waveforms and store the results in a gold table.
