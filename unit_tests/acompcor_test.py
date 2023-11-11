import logging
import os
from pathlib import Path
import subprocess as sp
import numpy as np
import pandas as pd
import re
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile
import errorhandler
from typing import List, Tuple, Union
import nibabel as nib
from flywheel_gear_toolkit import GearToolkitContext

from utils.command_line import exec_command
from nipype.algorithms.confounds import ACompCor
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import ErodeImage

from fw_gear_bids_compcor.main import identify_paths, acompcor, motion_confounds

log = logging.getLogger(__name__)

app_options = dict()
gear_options = dict()


gear_options["work-dir"] = "/flywheel/v0/work/64fba1577a7796b91f5f3ec7"
gear_options["dry-run"] = False
app_options["work-dir"] = "/flywheel/v0/work/64fba1577a7796b91f5f3ec7"
app_options["sid"] = "085"
app_options["sesid"] = "RRAYS2"
app_options["acqid"] = "task-rest_run-01"

identify_paths(gear_options, app_options)

#skip registration this time
app_options["highres_csf_mask_file"] = app_options["highres_file"].replace(".nii.gz", "_resamp_csf_mask_eroded.nii.gz")
app_options["highres_wm_mask_file"] = app_options["highres_file"].replace(".nii.gz", "_resamp_wm_mask_eroded.nii.gz")
app_options["highres_mask_file"] = app_options["highres_file"].replace(".nii.gz", "_resamp_mask.nii.gz")

acompcor(gear_options, app_options)

motion_confounds(gear_options, app_options)


