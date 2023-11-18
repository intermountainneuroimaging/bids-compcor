"""Main module."""

import logging
import os, sys
import os.path as op
from pathlib import Path
import subprocess as sp
import numpy as np
import pandas as pd
import re
import shutil
import tempfile
import math
from zipfile import ZIP_DEFLATED, ZipFile
import errorhandler
from typing import List, Tuple, Union
import nibabel as nib
from flywheel_gear_toolkit import GearToolkitContext

from utils.command_line import exec_command
from nipype.algorithms.confounds import ACompCor, FramewiseDisplacement, ComputeDVARS
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import ErodeImage

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()


def prepare(
        gear_options: dict,
        app_options: dict,
) -> Tuple[List[str], List[str]]:
    """Prepare everything for the algorithm run.

    It should:
     - Install FreeSurfer license (if needed)

    Same for FW and RL instances.
    Potentially, this could be BIDS-App independent?

    Args:
        gear_options (Dict): gear options
        app_options (Dict): options for the app

    Returns:
        errors (list[str]): list of generated errors
        warnings (list[str]): list of generated warnings
    """
    # pylint: disable=unused-argument
    # for now, no errors or warnings, but leave this in place to allow future methods
    # to return an error
    errors: List[str] = []
    warnings: List[str] = []

    return errors, warnings
    # pylint: enable=unused-argument


def run(gear_options: dict, app_options: dict, gear_context: GearToolkitContext) -> int:
    """Run FSL-FEAT using generic bids-derivative inputs.

    Arguments:
        gear_options: dict with gear-specific options
        app_options: dict with options for the BIDS-App

    Returns:
        run_error: any error encountered running the app. (0: no error)
    """

    log.info("This is the beginning of the run file")

    # psudocode
    # 1. identify anatomical scan (white and grey matter masks)
    # 2. identify func files (if suffix provide otherwise use "desc-preproc_bold"
    # 3. identify confounds file (import 6 DOF motion)
    # 4. check if confounds already exist - if not compute them
    # 5.    ACompCor
    # 6.    motion deriv
    # 7.    framewise displacement
    # 8.    DVARS
    # 9.    outliers (MotionOutliers -- fsl_motion_outliers, metric: ('refrms' or 'dvars' or 'refmse' or 'fd' or 'fdrms') -- threshold)
    #           defaults are FD > 0.5 mm or DVARS > 1.5
    #           save unique set of outliers (grab non-steady state from mriqc also)

    outfiles = []; run_error = 0
    for acq in app_options["runs"]:

        app_options["acqid"] = acq

        log.info("Generating regressors: %s", acq)
        # identify filepaths for analysis
        identify_paths(gear_options, app_options)

        # if relavent filepaths exist, generate new denoise regressors - else skip
        if os.path.exists(app_options["highres_file"]) and os.path.exists(app_options["func_file"]):
            acompcor(gear_options, app_options)

            motion_confounds(gear_options, app_options)

            outliers(gear_options, app_options)

            outfiles.append(app_options["confounds_file"])

        else:
            log.warning("Files needed for ACompCor and motion regression not found...skipping: %s", acq)
            continue

        if error_handler.fired:
            log.critical('Failure: exiting with code 1 due to logged errors')
            run_error = 1
            return run_error

    # move output files to path with destination-id
    for f in outfiles:
        newpath = f.replace(str(gear_options["work-dir"]), op.join(gear_options["work-dir"],gear_options["destination-id"]))
        os.makedirs(op.dirname(newpath), exist_ok=True)
        shutil.copy(f, newpath, follow_symlinks=True)

    # zip results
    cmd = "zip -q -r " + os.path.join(gear_options["output-dir"],
                                      "compcor_" + str(gear_options["destination-id"])) + ".zip " + gear_options[
              "destination-id"]
    execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["work-dir"])

    return run_error


def identify_paths(gear_options: dict, app_options: dict):

    # apply filemapper to each file pattern and store
    if os.path.isdir(os.path.join(gear_options["work-dir"], "fmriprep")):
        pipeline = "fmriprep"
    elif os.path.isdir(os.path.join(gear_options["work-dir"], "bids-hcp")):
        pipeline = "bids-hcp"
    elif len(os.walk(gear_options["work-dir"]).next()[1]) == 1:
        pipeline = os.walk(gear_options["work-dir"]).next()[1]
    else:
        log.error("Unable to interpret pipeline for analysis. Contact gear maintainer for more details.")
    app_options["pipeline"] = pipeline

    lookup_table = {"WORKDIR": str(gear_options["work-dir"]), "PIPELINE": pipeline, "SUBJECT": app_options["sid"],
                    "SESSION": app_options["sesid"], "ACQ": app_options["acqid"]}

    # select high_resolution file - white and grey matter masks if exist
    if pipeline == "bids-hcp":
        app_options["highres_file"] = apply_lookup("{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/sub-{SUBJECT}_ses-{SESSION}_space-MNI152NLin6Asym_desc-brain_T1w.nii.gz", lookup_table)
        space = [s for s in app_options["highres_file"].split("_") if "space" in s]
        app_options["output_space"] = space

        # confounds file
        "_desc-confounds_timeseries.tsv"
        app_options["confounds_file"] = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_desc-confounds_timeseries.tsv",
            lookup_table)

        # functional preprocessed data
        app_options["func_file"] = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz",
            lookup_table)


    # elif pipeline == "fmriprep":
    #     highresfile = apply_lookup(
    #         "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*desc-preproc_T1w.nii.gz",
    #         lookup_table)
    #     highresfile = searchfiles()
    #     app_options["highres_file"] = highresfile[0]
    #     space = [s for s in highresfile[0].split("_") if "space" in s]
    #     app_options["output_space"] = space
    #
    #     wm_mask = apply_lookup(
    #         "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*"+space+"*label-WM_probseg.nii.gz",
    #         lookup_table)
    #
    #     gm_mask = apply_lookup(
    #         "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*"+space+"*label-GM_probseg.nii.gz",
    #         lookup_table)
    #
    #     csf_mask = apply_lookup(
    #         "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*"+space+"*label-CSF_probseg.nii.gz",
    #         lookup_table)
    #
    #     app_options["wm_mask_file"] = searchfiles()
    #     app_options["gm_mask_file"] = searchfiles()
    #     app_options["csf_mask_file"] = searchfiles()

    return app_options


def acompcor(gear_options: dict, app_options: dict):
    log.info("ACompCor will be included as confound. Running FSL FAST segmentation now.")

    with tempfile.TemporaryDirectory(dir=gear_options["work-dir"]) as tmpdir:

        try:
            os.chdir(tmpdir)

            # run fast segmentation to generate mask, then use eroded csf and wm for acompcor
            if not "highres_wm_mask_file" in app_options:
                # run fast segmentation to generate mask, then use eroded csf and wm for acompcor
                fastr = fsl.FAST()
                fastr.inputs.in_files = app_options["highres_file"]
                log.info(fastr.cmdline)
                out = fastr.run()

                log.info("Eroding white matter and csf masks now.")
                ero = ErodeImage()
                ero.inputs.in_file = app_options["highres_file"].replace(".nii.gz", "_pve_0.nii.gz")
                ero.inputs.out_file = app_options["highres_file"].replace(".nii.gz", "_csf_mask_eroded.nii.gz")
                ero.inputs.kernel_size = 4
                ero.inputs.kernel_shape = 'box'
                log.info(ero.cmdline)
                out = ero.run()

                ero = ErodeImage()
                ero.inputs.in_file = app_options["highres_file"].replace(".nii.gz", "_pve_2.nii.gz")
                ero.inputs.out_file = app_options["highres_file"].replace(".nii.gz", "_wm_mask_eroded.nii.gz")
                ero.inputs.kernel_size = 6
                ero.inputs.kernel_shape = 'box'
                log.info(ero.cmdline)
                out = ero.run()

                # register to functional space...
                log.info("Resampling to functional space.")
                flt = fsl.FLIRT()
                flt.inputs.in_file = app_options["highres_file"].replace(".nii.gz", "_csf_mask_eroded.nii.gz")
                flt.inputs.out_file = app_options["highres_file"].replace(".nii.gz", "_resamp_csf_mask_eroded.nii.gz")
                flt.inputs.reference = app_options["func_file"]
                flt.inputs.interp = "nearestneighbour"
                flt.inputs.output_type = "NIFTI_GZ"
                flt.inputs.uses_qform = True
                flt.inputs.apply_xfm = True
                log.info(flt.cmdline)
                res = flt.run()

                flt = fsl.FLIRT()
                flt.inputs.in_file = app_options["highres_file"].replace(".nii.gz", "_wm_mask_eroded.nii.gz")
                flt.inputs.out_file = app_options["highres_file"].replace(".nii.gz", "_resamp_wm_mask_eroded.nii.gz")
                flt.inputs.reference = app_options["func_file"]
                flt.inputs.interp = "nearestneighbour"
                flt.inputs.output_type = "NIFTI_GZ"
                flt.inputs.uses_qform = True
                flt.inputs.apply_xfm = True
                log.info(flt.cmdline)
                res = flt.run()

                # create brain mask resampled in func space
                thr = fsl.maths.MathsCommand()
                thr.inputs.in_file = app_options["highres_file"]
                thr.inputs.args = " -thr 1 -bin"
                thr.inputs.out_file = app_options["highres_file"].replace(".nii.gz", "_mask.nii.gz")
                res = thr.run()

                flt = fsl.FLIRT()
                flt.inputs.in_file = app_options["highres_file"].replace(".nii.gz", "_mask.nii.gz")
                flt.inputs.out_file = app_options["highres_file"].replace(".nii.gz", "_resamp_mask.nii.gz")
                flt.inputs.reference = app_options["func_file"]
                flt.inputs.interp = "nearestneighbour"
                flt.inputs.output_type = "NIFTI_GZ"
                flt.inputs.uses_qform = True
                flt.inputs.apply_xfm = True
                log.info(flt.cmdline)
                res = flt.run()

                # store masks for additional acompcor calls
                app_options["highres_csf_mask_file"] = app_options["highres_file"].replace(".nii.gz", "_resamp_csf_mask_eroded.nii.gz")
                app_options["highres_wm_mask_file"] = app_options["highres_file"].replace(".nii.gz", "_resamp_wm_mask_eroded.nii.gz")
                app_options["highres_mask_file"] = app_options["highres_file"].replace(".nii.gz", "_resamp_mask.nii.gz")

            log.info("Computing ACompCor signals now.")
            ccinterface = ACompCor()
            wm_mask = app_options["highres_file"].replace(".nii.gz", "_resamp_wm_mask_eroded.nii.gz")
            vent_mask = app_options["highres_file"].replace(".nii.gz", "_resamp_csf_mask_eroded.nii.gz")
            ccinterface.inputs.realigned_file = app_options["func_file"]
            ccinterface.inputs.mask_files = [wm_mask, vent_mask]
            ccinterface.inputs.merge_method = 'union'
            ccinterface.inputs.num_components = 50
            ccinterface.inputs.pre_filter = 'polynomial'
            ccinterface.inputs.regress_poly_degree = 2
            # ccinterface.inputs.components_file = os.path.join(app_options["funcpath"], "acompcor_signals.txt")  # move this file after removing header and updating delim
            res = ccinterface.run()

            tmpdf = pd.read_csv('components_file.txt', sep='\t')
            numrange = [*range(0,len(tmpdf.columns),1)]
            cols = ["a_comp_cor_"+str(s).zfill(2) for s in numrange]

            tmpdf.columns = cols

            # add confounds to current confound file...
            if not searchfiles(app_options["confounds_file"], dryrun=gear_options["dry-run"]):
                os.makedirs(Path(app_options["confounds_file"]).stem)
                confounds = tmpdf.copy()
            else:
                df = pd.read_csv(app_options["confounds_file"], sep='\t')
                if set(tmpdf.columns).intersection(set(df.columns)):
                    log.warning("Confounds file already contains a_comp_cor regressors... not overwriting")
                    return

                confounds = pd.concat([df, tmpdf], axis=1)

            confounds.to_csv(app_options["confounds_file"], sep='\t', index=False)

            os.chdir(gear_options["work-dir"])

        except KeyboardInterrupt:
            sys.exit()
            pass
        except Exception as e:
            os.chdir(gear_options["work-dir"])
            raise(e)

    return app_options


def motion_confounds(gear_options: dict, app_options: dict):

    # if confounds file contains motion regressors - create motion derivatives
    if not searchfiles(app_options["confounds_file"], dryrun=gear_options["dry-run"]):
        log.warning("no motion parameters provided - skipping.")
        return
    else:
        df = pd.read_csv(app_options["confounds_file"], sep='\t')

    dofs = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

    if all([s in df.columns for s in dofs]):
        for c in dofs:
            if not c+"_derivative1" in df.columns:
                df[c+"_derivative1"] = df[c].diff()
        for c in dofs:
            if not c + "_power2" in df.columns:
                df[c + "_power2"] = np.square(df[c])
        for c in dofs:
            if not c + "_derivative1_power2" in df.columns:
                df[c + "_derivative1_power2"] = np.square(df[c+"_derivative1"])

    # compute framewise displacement and DVARS
    if "framewise_displacement" not in df.columns:
        # create a par file to pass to FD compute
        parfile = op.join(op.dirname(app_options["confounds_file"]), "tmp.par")   #first three columns contain the rotations for the X, Y, and Z voxel axes, in radians. The remaining three columns contain the estimated X, Y, and Z translations
        pardf = df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]].copy()

        pardf.to_csv(parfile, sep=" ", header=None, index=False)
        with tempfile.TemporaryDirectory(dir=gear_options["work-dir"]) as tmpdir:
            try:
                os.chdir(tmpdir)
                fdinterface = FramewiseDisplacement()
                fdinterface.inputs.in_file = parfile
                fdinterface.inputs.parameter_source = "FSL"
                res = fdinterface.run()
                tmpdf = pd.read_csv('fd_power_2012.txt', sep='\t')
                tmpdf.columns = ["framewise_displacement"]

                df = pd.concat([df, tmpdf], axis=1)

                os.chdir(gear_options["work-dir"])

            except KeyboardInterrupt:
                sys.exit()
                pass
            except Exception as e:
                os.chdir(gear_options["work-dir"])
                raise (e)

    if "dvars" not in df.columns:
        with tempfile.TemporaryDirectory(dir=gear_options["work-dir"]) as tmpdir:
            try:
                os.chdir(tmpdir)
                dvarinterface = ComputeDVARS()
                dvarinterface.inputs.in_file = app_options["func_file"]
                dvarinterface.inputs.in_mask = app_options["highres_mask_file"]
                dvarinterface.inputs.save_nstd = True
                dvarinterface.inputs.save_std = True
                res = dvarinterface.run()
                res.outputs.out_all
                tmpdf1 = pd.read_csv(res.outputs.out_nstd, sep='\t')
                tmpdf1.columns = ["dvars"]

                tmpdf2 = pd.read_csv(res.outputs.out_std, sep='\t')
                tmpdf2.columns = ["std_dvars"]

                df = pd.concat([df,tmpdf1,tmpdf2], axis=1)

                os.chdir(gear_options["work-dir"])

            except KeyboardInterrupt:
                sys.exit()
                pass
            except Exception as e:
                os.chdir(gear_options["work-dir"])
                raise(e)

    df.to_csv(app_options["confounds_file"], sep='\t', index=False)


def outliers(gear_options: dict, app_options: dict):
    pass

# -----------------------------------------------
# Support functions
# -----------------------------------------------

def generate_command(
        gear_options: dict,
        app_options: dict,
) -> List[str]:
    """Build the main command line command to run.

    This method should be the same for FW and XNAT instances. It is also BIDS-App
    generic.

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json
    Returns:
        cmd (list of str): command to execute
    """

    cmd = []
    cmd.append(gear_options["feat"]["common_command"])
    cmd.append(app_options["design_file"])

    return cmd


def execute_shell(cmd, dryrun=False, cwd=os.getcwd()):
    log.info("\n %s", cmd)
    if not dryrun:
        terminal = sp.Popen(
            cmd,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
            cwd=cwd
        )
        stdout, stderr = terminal.communicate()
        returnCode = terminal.poll()
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        if returnCode > 0:
            log.error("Error. \n%s\n%s", stdout, stderr)
        return returnCode


def searchfiles(path, dryrun=False, exit_on_errors=True, find_first=False) -> list[str]:
    cmd = "ls -d " + path

    log.debug("\n %s", cmd)

    if not dryrun:
        terminal = sp.Popen(
            cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, stderr = terminal.communicate()
        returnCode = terminal.poll()
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        files = stdout.strip("\n").split("\n")

        if returnCode > 0 and exit_on_errors:
            log.error("Error. \n%s\n%s", stdout, stderr)

        if returnCode > 0 and not exit_on_errors:
            log.warning("Warning. \n%s\n%s", stdout, stderr)

        if find_first:
            files = files[0]

        return files


def sed_inplace(filename, pattern, repl):
    """
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(dir=os.getcwd(), mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)


def locate_by_pattern(filename, pattern):
    """
    Locates all instances that meet pattern and returns value from file.
    Args:
        filename: text file
        pattern: regex

    Returns:

    """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)
    arr = []
    with open(filename) as src_file:
        for line in src_file:
            num = re.findall(pattern_compiled, line)
            if num:
                arr.append(num[0])

    return arr


def replace_line(filename, pattern, repl):
    """
        Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
        `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
        """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(dir=os.getcwd(), mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                if re.findall(pattern_compiled, line):
                    tmp_file.write(repl)
                else:
                    tmp_file.write(line)

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)


def fetch_dummy_volumes(taskname, context):
    # Function generates number of dummy volumes from config or mriqc stored IQMs
    if context.config["DropNonSteadyState"] is False:
        return 0

    acq, f = metadata.find_matching_acq(taskname, context)

    if "DummyVolumes" in context.config:
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        log.info("Set by user....Using %s dummy volumes", context.config['DummyVolumes'])
        return context.config['DummyVolumes']

    if f:
        IQMs = f.info["IQM"]
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        if "dummy_trs_custom" in IQMs:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs_custom"])
            return IQMs["dummy_trs_custom"]
        else:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs"])
            return IQMs["dummy_trs"]

    # if we reach this point there is a problem! return error and exit
    log.error(
        "Option to drop non-steady state volumes selected, no value passed or could be interpreted from session metadata. Quitting...")


def apply_lookup(text, lookup_table):
    if '{' in text and '}' in text:
        for lookup in lookup_table:
            text = text.replace('{' + lookup + '}', lookup_table[lookup])
    return text
