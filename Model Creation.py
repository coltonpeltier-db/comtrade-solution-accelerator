# Databricks notebook source
# MAGIC %md
# MAGIC # Dataset Declaration
# MAGIC 
# MAGIC Pallav K Bera, Can Isik, Vajendra Kumar, February 13, 2020, "Transients and Faults in Power Transformers and Phase Angle Regulators (DATASET)", IEEE Dataport, doi: https://dx.doi.org/10.21227/1d1w-q940.

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports
# MAGIC Import any necessary libraries here

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
import tensorflow as tf
import numpy as np
import mlflow
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC Disable MLFlow autolog so we can manually log and register the final model.

# COMMAND ----------

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Constants
# MAGIC Define any constants here.

# COMMAND ----------

dir_path = "dbfs:/FileStore/tables/colton/ieee_transients/transient disturbances/"
TIME = "time"
PHASE_A = "IA"
PHASE_B = "IB"
PHASE_C = "IC"
IS_FAULT = "is_fault"

# COMMAND ----------

# MAGIC %md
# MAGIC # File Read
# MAGIC Read all the text files from the data sets into spark and add a variable for whether the waveform is a fault or not.

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
    .csv(dir_path)
    .withColumn("filename", F.input_file_name())
    .withColumn(IS_FAULT, F.col("filename").rlike("external").cast("int"))
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC There are only 600 waveforms of length 726 so we can read this directly into Pandas for in-memory processing.

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the pandas dataframe into two tensors: one for the signals themselves and one for the labels (whether the signals contain a fault or not).

# COMMAND ----------

current_signals = df_pd.loc[:, [PHASE_A, PHASE_B, PHASE_C]]
current_signals_tensor = tf.signal.frame(current_signals.to_numpy(),726,726,axis=0)
labels_tensor = tf.signal.frame(df_pd[IS_FAULT].iloc[::726].to_numpy(),1,1)
print(current_signals_tensor.shape, labels_tensor.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Shuffle the data and labels, then split the waveforms into three sets: train, validation, and test.

# COMMAND ----------

indices = tf.range(start=0, limit=tf.shape(current_signals_tensor)[0], dtype=tf.int32)
tf.random.set_seed(42)
shuffled_indices = tf.random.shuffle(indices)

shuffled_signals = tf.gather(current_signals_tensor, shuffled_indices)
shuffled_labels = tf.gather(labels_tensor, shuffled_indices)

# COMMAND ----------

train_signals = shuffled_signals[:480]
train_labels = shuffled_labels[:480]

val_signals = shuffled_signals[480:540]
val_labels = shuffled_labels[480:540]

test_signals = shuffled_signals[540:]
test_labels = shuffled_labels[540:]

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC Create a function which will create a new compiled convolutional binary classification model.

# COMMAND ----------

def create_convolutional_classification_model() -> tf.keras.Model:
    tf.random.set_seed(13)
    inp = tf.keras.Input(shape=[726,3])
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
# MAGIC ## Model Training
# MAGIC Get an instance of the classification model and make sure the model can be overtrained. Then choose the right stopping epoch and train a final model.

# COMMAND ----------

model1 = create_convolutional_classification_model()

# COMMAND ----------

history = model1.fit(x=train_signals, y=train_labels, batch_size=16, epochs=64, validation_data=(val_signals,val_labels))

# COMMAND ----------

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC The validation loss and training loss seem to drop together very well until around epoch 26 where we start to see diminishing returns and some bumpiness in the decrease of the loss (both validation and training). Because we have such little data, let's train to the 26th epoch and save our model.

# COMMAND ----------

train_val_signals = tf.concat([train_signals, val_signals],axis=0)
train_val_labels = tf.concat([train_labels, val_labels],axis=0)

model2 = create_convolutional_classification_model()
history = model2.fit(x=train_val_signals, y=train_val_labels, batch_size=16, epochs=26)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now evaluate the model on the test data.

# COMMAND ----------

test_metrics = model2.evaluate(x=test_signals,y=test_labels)
print(test_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register Model
# MAGIC The training and evaluation metrics for the model look quite good, so now we'll log and register the model in mlflow.

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.log_metric("test_loss", test_metrics[0])
    mlflow.log_metric("test_precision", test_metrics[1])
    mlflow.log_metric("test_recall", test_metrics[2])
    mlflow.keras.log_model(keras_model=model2, artifact_path="model")

# COMMAND ----------

register_info = mlflow.register_model(f"runs:/{run.info.run_id}/model", "fault_detection")

# COMMAND ----------

client = mlflow.MlflowClient()
client.transition_model_version_stage(name = "fault_detection", version=int(register_info.version), stage="Production")

# COMMAND ----------


