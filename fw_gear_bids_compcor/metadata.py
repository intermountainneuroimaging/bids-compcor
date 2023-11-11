import os, sys
import pandas as pd
import logging

log = logging.getLogger(__name__)


def ingest_labels(filename):
    df = pd.read_csv(filename, sep=',', skiprows=1, header=None, skipinitialspace=True, skipfooter=1, engine='python')
    df.columns = ['component', 'type', 'bool', 'weights']

    return df


def ingest_icstats(filename):
    df = pd.read_csv(filename, delim_whitespace=True, header=None)
    df.columns = ['prc_explained_variance', 'prc_total_variance', 'ignore_1', 'ignore_2']

    return df


def report_metrics(df):

    metrics = df.groupby(['type'])[['prc_explained_variance', 'prc_total_variance', 'weights']].sum().reset_index()
    metrics['count'] = metrics['type'].map(df['type'].value_counts())

    return metrics


def find_matching_acq(bids_name, context):
    """
    Args:
        bids_name (str): partial filename used in HCPPipeline matching BIDS filename in BIDS.info
        context (obj): gear context
    Returns:
        acquisition and file objects matching the original image file on which the
        metrics were completed.
    """
    fw = context.client
    dest_id = context.destination["id"]
    destination = fw.get(dest_id)
    session = fw.get_session(destination.parents["session"])

    # assumes reproin naming scheme for acquisitions!
    for acq in session.acquisitions.iter_find():
        full_acq = fw.get_acquisition(acq.id)
        if ("func-bold" in acq.label) and (bids_name in acq.label) and ("sbref" not in acq.label.lower()) and ("ignore-BIDS" not in acq.label):
            for f in full_acq.files:
                if bids_name in f.info.get("BIDS").get("Filename") and "nii" in f.name:
                    return full_acq, f

