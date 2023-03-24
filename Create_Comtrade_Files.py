# Databricks notebook source
# MAGIC %pip install comtrade

# COMMAND ----------

# MAGIC %pip install git+https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers

# COMMAND ----------

from comtradehandlers import writer as ComtradeWriter

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports and Constants

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import datetime
import pandas as pd
from comtrade import Comtrade
import matplotlib.pyplot as plt
from pathlib import Path
from random import randint
import uuid

# COMMAND ----------

# MAGIC %md
# MAGIC ## Constants

# COMMAND ----------

all_txt_files = [str(path) for path in list(Path("/dbfs/FileStore/tables/colton/ieee_transients/transient disturbances").rglob("*.txt"))]
save_path_prefix = "/dbfs/FileStore/tables/colton/ieee_transients_comtrade_v3"
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

def write_comtrade_file(output_file_prefix : str, station_name : str, rec_dev_id : int, input_data : pd.DataFrame) -> None:
    start_time = datetime.datetime.now()
    trigger_time = start_time + datetime.timedelta(milliseconds=20)
    # Create comtrade writer object
    comtradeWriter = ComtradeWriter(f"{save_path_prefix}/{output_file_prefix}.cfg", start_time, trigger_time, station_name = station_name, rec_dev_id=rec_dev_id)
    # A Current
    comtradeWriter.add_analog_channel(IA, A_PHS, CURRENT, uu=AMPS, skew=0, min=-500, max=500, primary=1,secondary=1)
    # B Current
    comtradeWriter.add_analog_channel(IB, B_PHS, CURRENT, uu=AMPS, skew=0, min=-500, max=500, primary=1,secondary=1)
    # C Current
    comtradeWriter.add_analog_channel(IC, C_PHS, CURRENT, uu=AMPS, skew=0, min=-500, max=500, primary=1,secondary=1)
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

for txt_file_path in all_txt_files:
    fn = all_txt_files[0].split("/")[-1].split(".")[0] # get the filename with no extension or path.
    uuid_suffix = str(uuid.uuid4()).split("-")[-1]
    _tmp = read_txt_file_to_pd(txt_file_path)
    write_comtrade_file(fn + uuid_suffix, fn, randint(0,9999), _tmp)

# COMMAND ----------

rec = Comtrade()
rec.load(f"{save_path_prefix}/{fn + uuid_suffix}.cfg", f"{save_path_prefix}/{fn + uuid_suffix}.dat")

# COMMAND ----------

plt.plot(rec.analog[0])
plt.plot(rec.analog[1])
plt.plot(rec.analog[2])

# COMMAND ----------


