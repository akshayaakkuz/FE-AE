"""Paths and experiment settings. Edit DATA_DIR / OUT_DIR and the file names to match your copies of the datasets."""
import os

DATA_DIR = os.environ.get("FEAE_DATA", "data")       # folder that contains the raw datasets
OUT_DIR = os.environ.get("FEAE_OUT", "results")      # prepared streams (pickles) and result tables

SEEDS = (11, 21, 42, 52, 84)        # stationary streams, synthetic streams
SEEDS3 = (11, 42, 84)               # emerging-attack and robustness scenarios
SAMPLE = 60000                      # flows sampled per dataset
TOL = 1536                          # a change counts as detected if an alarm follows within TOL samples (3 windows)

# raw files, relative to DATA_DIR
FILES = dict(
    NSL=("KDDTrain+.txt", "KDDTest+.txt"),                                   # NSL-KDD public files (with attack names)
    UNSW=("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"),        # UNSW-NB15 partitioned sets
    ToN="train_test_network.csv",                                            # ToN-IoT network flows
    DDoS="cicddos2019/*.parquet",                                            # CIC-DDoS2019 (parquet export)
    IoT="IoT Network Intrusion Dataset.csv",                                 # IoT Network Intrusion Dataset (Kang et al.)
)

# identifier, address, timestamp and label-derived columns removed before training (leakage control)
EXCLUDE = dict(
    NSL=["attack", "difficulty_level"],
    UNSW=["id", "attack_cat"],
    ToN=["src_ip", "dst_ip", "type"],
    DDoS=["Flow ID", "Source IP", "Destination IP", "Timestamp", "Unnamed: 0", "SimillarHTTP", "Source Port",
          "Destination Port", "atk"],
    IoT=["Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Sub_Cat", "Cat"],
)
