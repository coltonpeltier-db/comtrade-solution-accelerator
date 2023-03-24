# Databricks notebook source
# MAGIC %md
# MAGIC Install the <a href="https://github.com/dparrini/python-comtrade">`python-comtrade`</a>, <a href="https://github.com/ijl/orjson">`orjson`</a>, <a href="https://github.com/relihanl/comtradehandlers">`ComtradeHandlers`</a> libraries via pip.

# COMMAND ----------

# MAGIC %pip install comtrade orjson git+https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

from comtradehandlers import writer as ComtradeWriter
import pyspark.sql.functions as F
import comtrade
from comtrade import Comtrade
from pathlib import Path
from random import randint
import uuid
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, StringType, StructType, StructField, ArrayType, DoubleType, MapType, TimestampType, IntegerType, LongType
import pandas as pd
import io
import pickle
import numpy as np
import datetime
from typing import Iterator, List
import orjson
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC # Constants

# COMMAND ----------

FILENAME = "filename"
DIRECTORY = "directory"
CONTENT = "content"
CONTENTS_CFG = "config_content"
CONTENTS_DAT = "dat_content"

# Output Columns from UDF
TIME = "timestamp"
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

COMTRADE_FILES_PATH = "/FileStore/tables/colton/ieee_transients_comtrade_v2"

# Database and table names
SCHEMA_NAME = "comtrade_db"
FINAL_OUTPUT_TABLE = f"{SCHEMA_NAME}.pivoted_current"

DBFS_COLON = "dbfs:"
DBFS_SLASH = "/dbfs"

# COMMAND ----------

# MAGIC %md
# MAGIC # Create COMTRADE Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Data
# MAGIC Download the <a href="https://ieee-dataport.org/open-access/transients-and-faults-power-transformers-and-phase-angle-regulators-dataset">*"Transients and Faults in Power Transformers and Phase Angle Regulators”* dataset from the IEEE Dataport</a>. You'll need to create a free IEEE account to do so.
# MAGIC 
# MAGIC The dataset consists of a few thousand simulated examples of various power quality transient events. They are formatted as flat text files. In order to showcase the solution acclerator, we'll need to convert them to COMTRADE files using the installed `ComtradeHandlers` library.
# MAGIC 
# MAGIC After downloading the .zip file, unzip it locally on your computer and upload the underlying files and folders to your databricks workspace using the **Data Explorer** or <a href="https://docs.databricks.com/dev-tools/cli/index.html">CLI</a> and specify the location of the files in the below `folder_location_ieee_zip_contents` variable.
# MAGIC 
# MAGIC As an example here are the CLI commands I executed from my computer to create the desired directory and upload the files and folders:
# MAGIC 
# MAGIC ```
# MAGIC $ databricks fs mkdirs dbfs:/FileStore/power_quality_transients --profile mlpractice
# MAGIC $ databricks fs cp -r "Dataset for Transformer & PAR transients" dbfs:/FileStore/power_quality_transients --profile mlpractice
# MAGIC ```

# COMMAND ----------

folder_location_ieee_zip_file = f"FileStore/power_quality_transients" # TODO : fill out with your folder path

# COMMAND ----------

# MAGIC %md
# MAGIC Create a subfolder to output the files to.

# COMMAND ----------

# Create a directory to unzip to
dir_created = dbutils.fs.mkdirs(f"{DBFS_COLON}/{folder_location_ieee_zip_file}/{output_sub_folder}")
if (dir_created == False):
    print("Error creating the directory!")
else:
    print("Destination directory created successfully or it already existed.")

# COMMAND ----------

# MAGIC %md
# MAGIC Now the uploaded zip file can be extracted to the desired path. This is quite slow, and will likely take about a few minutes as it is a large zip file.
# MAGIC 
# MAGIC TODO : Update the below command with the path to your zip file and output directory!

# COMMAND ----------

# MAGIC %sh unzip "/dbfs/FileStore/ieee/Dataset for Transformer & PAR transients.zip" -d "/dbfs/FileStore/ieee/unzipped_ieee_data_final"

# COMMAND ----------

# MAGIC %md
# MAGIC Visually confirm the `.txt` files have been extracted properly.

# COMMAND ----------

dbutils.fs.ls(f"{DBFS_COLON}/{folder_location_ieee_zip_file}/{output_sub_folder}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Read COMTRADE files into Spark
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
    .load(COMTRADE_FILES_PATH)
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
    .load(COMTRADE_FILES_PATH) # specify the path to look in
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
# MAGIC # Using the `comtrade` library to parse file pairs
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
    # Because spark timestamps only go down to the millisecond level, we'll need to track microseconds separately.
    ret[TIME_MILLIS] = [int(_comtrade.start_timestamp.timestamp()) + int((t * 1e6) // 1e3) for t in _comtrade.time]
    ret[TIME_MICRO] = [int(second * 1e6 % 1e3) for second in _comtrade.time] # Get the modulus of microseconds to milliseconds
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
json_retrieved.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parsing the JSON
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
    StructField(TIME_MILLIS, ArrayType(TimestampType())),
    StructField(TIME_MICRO, ArrayType(IntegerType()))
])

# COMMAND ----------

proc = (
    json_retrieved
    .withColumn("parsed", F.from_json(F.col("binary_json").cast("string"), json_schema))
    .select(FILENAME, "parsed.*")
)
proc.show()

# COMMAND ----------

# MAGIC %md
# MAGIC In this case all of our COMTRADE files define captures with only 3 analog channels: IA, IB, and IC. In many cases different COMTRADE files may have many very different channel names and some data engineering efforts will be required to standardize those channels before pivoting the data in the following steps.

# COMMAND ----------

# MAGIC %md
# MAGIC # Pivoting the Data
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

array_columns_to_bundle = ["time_millis", "microseconds", "analog"] # Include status here too, if you need status channels
unneeded_columns = ["analog_units","status", "status_channel_names"]
pivoted_current = (
    proc
    .drop(*metadata_cols, *unneeded_columns)
    .select(
        "*",
        F.arrays_zip(*array_columns_to_bundle).alias("array_cols")
    )
    .drop(*array_columns_to_bundle)
    .select("*",F.explode("array_cols").alias("struct_col"))
    .drop("array_cols")
    .select("*","struct_col.*")
    .drop("struct_col")
    .withColumn("analog_channels_per_timestamp", F.arrays_zip("analog_channel_names","analog"))
    .drop("analog_channel_names","analog")
    .select("*", F.explode("analog_channels_per_timestamp").alias("analog_channel"))
    .drop("analog_channels_per_timestamp")
    .select("*", "analog_channel.*")
    .drop("analog_channel")
    .filter(F.col("analog_channel_names").isin(["IA","IB","IC"])) # This will ensure we only keep these three variables
    .groupby(FILENAME, TIME_MILLIS, TIME_MICRO)
    .pivot("analog_channel_names",["IA","IB","IC"]) # If you don't know all the possible column names this second argument can be left blank but the computation time will take longer.
    .agg(F.first("analog"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's plot an example waveform

# COMMAND ----------

fn_to_examine = pivoted_current.select(FILENAME).distinct().limit(1).toPandas()[FILENAME].iloc[0]
examine = pivoted_current.filter(F.col(FILENAME) == fn_to_examine).toPandas()
sorted_examine = examine.sort_values([TIME_MILLIS,TIME_MICRO]).reset_index(drop=True)

# COMMAND ----------

# Create a plot of the waveform
plt.plot(sorted_examine["IA"])
plt.plot(sorted_examine["IB"])
plt.plot(sorted_examine["IC"])
plt.suptitle(sorted_examine[FILENAME].iloc[0])
plt.show()

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC # What would a data pipeline look like?
# MAGIC Saving the data should be done in the <a href="https://www.databricks.com/glossary/medallion-architecture">medallion architecture</a>. As such the flow would look like:
# MAGIC * Saving the raw binary reads of the `.cfg` and `.dat` files into bronze tables.
# MAGIC * Joining the previous two tables on their filenames and saving as a bronze table.
# MAGIC * Use the defined `pandas_udf` to extract binary JSON, parse the JSON, and store it in a silver table.
# MAGIC * Create silver tables for the flattened data and metadata from the parsed JSON.
# MAGIC * Perform filtering for the desired columsn and pivot the data into a silver table.
# MAGIC * Use a trained tensorflow model to detect faults in the waveforms and store the results in a gold table.

# COMMAND ----------

df = spark.table("fault_detection.pivoted_current_silver")
df.display()

# COMMAND ----------

df.groupby("filename").agg(F.count("*")).display()

# COMMAND ----------

spark.table("fault_detection.comtrade_json_silver").select(F.size("analog")).distinct().show()

# COMMAND ----------

spark.table("fault_detection.comtrade_json_silver").select("filename",F.explode("timestamp").alias("timestamp")).groupby("filename").agg(F.countDistinct("timestamp")).show()

# COMMAND ----------

x = (
    df
    .select(F.struct(F.col("timestamp"),F.col("IA"), F.col("IB"), F.col("IC")).alias("timestep"),"filename")
    .groupby("filename")
    .agg(
        F.collect_list(F.col("timestep")).alias("timestep")
    )
    .withColumn("timestep", F.array_sort("timestep"))
    .withColumn("timestep_array", F.transform("timestep", lambda x: F.array(x["IA"],x["IB"],x["IC"])))
)

# COMMAND ----------

y = x.drop("timestep").limit(10).toPandas()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
result = np.vstack(y["timestep_array"].iloc[0])
result.shape

# COMMAND ----------

def fault_identification(waveform_pd):
    # Load model
    fault_model = mlflow.keras.load_model("models:/fault_detection/Production")
    wfs = np.vstack(waveform_pd.map(lambda x: np.expand_dims(np.vstack(x),0)).tolist())
    scores = np.squeeze(fault_model.predict(wfs))
    return pd.Series(scores)

# COMMAND ----------

fault_identification(y["timestep_array"].iloc[:1]).iloc[0]

# COMMAND ----------



# COMMAND ----------

wfs = np.vstack(y["timestep_array"].map(lambda x: np.expand_dims(np.vstack(x),0)).tolist())
wfs.shape

# COMMAND ----------

import mlflow
import tensorflow as tf
fault_model = mlflow.keras.load_model("models:/fault_detection/Production")

# COMMAND ----------

pd.Series(tf.squeeze(fault_model.predict(wfs))).iloc[0]

# COMMAND ----------

plt.plot(result[:,0])
plt.plot(result[:,1])
plt.plot(result[:,2])

# COMMAND ----------

import comtrade
import datetime
import numpy as np

# COMMAND ----------

_com = comtrade.Comtrade()
_com.load("/dbfs/FileStore/tables/colton/ieee_transients_comtrade_v3/cap1f_0100047ec92b85.cfg","/dbfs/FileStore/tables/colton/ieee_transients_comtrade_v3/cap1f_0100047ec92b85.dat")

# COMMAND ----------

0.0501 * 1e6

# COMMAND ----------

np.subtract(_com.time[1:], _com.time[:-1])

# COMMAND ----------

dts = [_com.start_timestamp + datetime.timedelta(seconds = fractional_second) for fractional_second in _com.time]
dts

# COMMAND ----------

[_dt.microsecond for _dt in dts]

# COMMAND ----------

st_60_2048 = (1 / 60 / 2048)
st_60_2048

# COMMAND ----------

rn = datetime.datetime.now()
rn + datetime.timedelta(seconds = st_60_2048), rn + 2*datetime.timedelta(seconds = st_60_2048)

# COMMAND ----------


