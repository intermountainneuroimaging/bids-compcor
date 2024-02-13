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
import errorhandler
from typing import List, Tuple
import nibabel as nb
from nipype.utils.filemanip import fname_presuffix
from flywheel_gear_toolkit import GearToolkitContext
from functools import reduce

from utils.command_line import exec_command
from nipype.algorithms.confounds import ACompCor, FramewiseDisplacement, ComputeDVARS
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import ErodeImage, ApplyMask
from fw_gear_bids_compcor.metadata import find_matching_acq

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
        if app_options["acompcor"] and os.path.exists(app_options["highres_file"]) and os.path.exists(app_options["func_file"]):
            acompcor(gear_options, app_options, gear_context)

        if app_options["confounds_file"]:
            motion_confounds(gear_options, app_options)

        if app_options["outliers"] and app_options["confounds_file"]:
            # set defaults for spike threshold if none passed
            if "dvars-spike-threshold" not in app_options:
                app_options["dvars-spike-threshold"] = 1.5
            if "fd-spike-threshold" not in app_options:
                app_options["fd-spike-threshold"] = 0.5

            outliers(gear_options, app_options)

        if app_options["confounds_file"]:
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
        newpath = f.replace(str(gear_options["work-dir"]), op.join(gear_options["work-dir"], gear_options["destination-id"]))
        os.makedirs(op.dirname(newpath), exist_ok=True)
        shutil.copy(f, newpath, follow_symlinks=True)

    # zip results
    cmd = "zip -q -r " + os.path.join(gear_options["output-dir"],
                                      "counfounds_" + str(gear_options["destination-id"])) + ".zip " + gear_options[
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
    if pipeline == "bids-hcp" or pipeline == "fmriprep":

        highresfile = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*space*desc-brain_T1w.nii.gz",
            lookup_table)
        app_options["highres_file"] = searchfiles(highresfile, find_first=True) if searchfiles(highresfile, find_first=True) else None
        space = [s for s in Path(app_options["highres_file"]).stem.split("_") if "space" in s][0]
        app_options["output_space"] = space

        brain_mask = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*" + space + "*desc-brain_mask.nii.gz",
            lookup_table)

        app_options["highres_mask_file"] = searchfiles(brain_mask, find_first=True,exit_on_errors=False) if searchfiles(brain_mask, find_first=True, exit_on_errors=False) else None

        if not app_options["highres_file"]:
            highresfile1 = searchfiles(apply_lookup(
                "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*space*desc-preproc_T1w.nii.gz",
                lookup_table), find_first=True)

            # apply mask to highres file...
            mask = ApplyMask()
            mask.inputs.in_file = highresfile1
            mask.inputs.mask_file = app_options["highres_mask_file"]
            mask.inputs.out_file = highresfile1.replace("desc-preproc_T1w.nii.gz","desc-brain_T1w.nii.gz")
            log.info(mask.cmdline)
            out = mask.run()
            app_options["highres_file"] = highresfile1.replace("desc-preproc_T1w.nii.gz","desc-brain_T1w.nii.gz")

        wm_mask = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*"+space+"*label-WM_probseg.nii.gz",
            lookup_table)

        gm_mask = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*"+space+"*label-GM_probseg.nii.gz",
            lookup_table)

        csf_mask = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/anat/*"+space+"*label-CSF_probseg.nii.gz",
            lookup_table)

        app_options["highres_wm_mask_file"] = searchfiles(wm_mask, find_first=True, exit_on_errors=False) if searchfiles(wm_mask, find_first=True, exit_on_errors=False) else None
        app_options["highres_gm_mask_file"] = searchfiles(gm_mask, find_first=True, exit_on_errors=False) if searchfiles(gm_mask, find_first=True, exit_on_errors=False) else None
        app_options["highres_csf_mask_file"] = searchfiles(csf_mask, find_first=True, exit_on_errors=False) if searchfiles(csf_mask, find_first=True, exit_on_errors=False) else None


        # functional preprocessed data
        app_options["func_file"] = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_"+space+"_desc-preproc_bold.nii.gz",
            lookup_table)

        # confounds file
        app_options["confounds_file"] = apply_lookup(
            "{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_{ACQ}_desc-confounds_timeseries.tsv",
            lookup_table)

    return app_options


def acompcor(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    log.info("ACompCor will be included as confound. Running FSL FAST segmentation now.")

    with tempfile.TemporaryDirectory(dir=gear_options["work-dir"]) as tmpdir:

        try:
            os.chdir(tmpdir)

            # run fast segmentation to generate mask, then use eroded csf and wm for acompcor
            if not app_options["highres_wm_mask_file"]:
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

            # add option for non-steady state volume removal
            orig_file = app_options["func_file"]
            if app_options['DropNonSteadyState']:
                app_options['AcqDummyVolumes'] = fetch_dummy_volumes(app_options["acqid"], gear_context)
                app_options["func_file"] = _remove_volumes(app_options["func_file"],
                                                                        app_options['AcqDummyVolumes'])

            log.info("Computing ACompCor signals now.")
            ccinterface = ACompCor()
            wm_mask = app_options["highres_wm_mask_file"]
            vent_mask = app_options["highres_csf_mask_file"]
            ccinterface.inputs.realigned_file = app_options["func_file"]
            ccinterface.inputs.mask_files = [wm_mask, vent_mask]
            ccinterface.inputs.merge_method = 'union'
            ccinterface.inputs.num_components = 50
            ccinterface.inputs.pre_filter = 'polynomial'
            ccinterface.inputs.regress_poly_degree = 2
            res = ccinterface.run()

            tmpdf = pd.read_csv('components_file.txt', sep='\t')

            # add placeholder rows if dummy volumes removed before analysis
            if app_options['DropNonSteadyState'] and app_options['AcqDummyVolumes'] > 0:
                zeros = pd.DataFrame(np.zeros([app_options['AcqDummyVolumes'], tmpdf.shape[1]]), columns=tmpdf.columns)
                tmpdf = pd.concat([zeros, tmpdf], axis=0, ignore_index=True)

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

    spike_criteria = {
        "framewise_displacement": (">", app_options["fd-spike-threshold"]),
        "std_dvars": (">", app_options["dvars-spike-threshold"]),
    }

    confounds_data = pd.read_csv(app_options["confounds_file"], sep="\t")
    confounds_data = spike_regressors(
        data=confounds_data,
        criteria=spike_criteria,
        header_prefix="bids_confounds_outlier",
        lags=[0],
        minimum_contiguous=None,
        concatenate=True,
        output="spikes",
    )

    confounds_data.to_csv(app_options["confounds_file"], sep='\t', index=False)

    return


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


def _remove_volumes(bold_file,n_volumes):
    if n_volumes == 0:
        return bold_file

    out = fname_presuffix(bold_file, suffix='_cut')
    nb.load(bold_file).slicer[..., n_volumes:].to_filename(out)
    return out


def _remove_timepoints(motion_file,n_volumes):
    arr = np.loadtxt(motion_file, ndmin=2)
    arr = arr[n_volumes:,...]

    filename, file_extension = os.path.splitext(motion_file)
    motion_file_new = motion_file.replace(file_extension,"_cut"+file_extension)
    np.savetxt(motion_file_new, arr, delimiter='\t')
    return motion_file_new


def _add_volumes(bold_file, bold_cut_file, n_volumes):
    """prepend n_volumes from bold_file onto bold_cut_file"""
    bold_img = nb.load(bold_file)
    bold_data = bold_img.get_fdata()
    bold_cut_img = nb.load(bold_cut_file)
    bold_cut_data = bold_cut_img.get_fdata()

    # assign everything from n_volumes foward to bold_cut_data
    bold_data[..., n_volumes:] = bold_cut_data

    out = bold_cut_file.replace("_cut","")
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)
    return out


def _add_volumes_melodicmix(melodic_mix_file, n_volumes):

    melodic_mix_arr = np.loadtxt(melodic_mix_file, ndmin=2)

    if n_volumes > 0:
        zeros = np.zeros([n_volumes, melodic_mix_arr.shape[1]])
        melodic_mix_arr = np.vstack([zeros, melodic_mix_arr])

        # save melodic_mix_arr
    np.savetxt(melodic_mix_file, melodic_mix_arr, delimiter='\t')


def fetch_dummy_volumes(taskname, context):
    # Function generates number of dummy volumes from config or mriqc stored IQMs
    if context.config["DropNonSteadyState"] is False:
        return 0

    acq, f = find_matching_acq(taskname, context)

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


def spike_regressors(
    data,
    criteria=None,
    header_prefix="motion_outlier",
    lags=None,
    minimum_contiguous=None,
    concatenate=True,
    output="spikes",
):
    """
    Add spike regressors to a confound/nuisance matrix.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        A tabulation of observations from which spike regressors should be
        estimated.
    criteria : :obj:`dict` of (:obj:`str`, ``'>'`` or ``'<'`` or :obj:`float`)
        Criteria for generating a spike regressor. If, for a given frame, the
        value of the variable corresponding to the key exceeds the threshold
        indicated by the value, then a spike regressor is created for that
        frame. By default, the strategy from [Power2014]_ is implemented: any
        frames with FD greater than 0.5 or standardised DV greater than 1.5
        are flagged for censoring.
    header_prefix : :obj:`str`
        The prefix used to indicate spike regressors in the output data table.
    lags: :obj:`list` of :obj:`int`
        A list indicating the frames to be censored relative to each flag.
        For instance, ``[0]`` censors the flagged frame, while ``[0, 1]`` censors
        both the flagged frame and the following frame.
    minimum_contiguous : :obj:`int` or :obj:`None`
        The minimum number of contiguous frames that must be unflagged for
        spike regression. If any series of contiguous unflagged frames is
        shorter than the specified minimum, then all of those frames will
        additionally have spike regressors implemented.
    concatenate : :obj:`bool`
        Indicates whether the returned object should include only spikes
        (if false) or all input time series and spikes (if true, default).
    output : :obj:`str`
        Indicates whether the output should be formatted as spike regressors
        ('spikes', a separate column for each outlier) or as a temporal mask
        ('mask', a single output column indicating the locations of outliers).

    Returns
    -------
    data : :obj:`~pandas.DataFrame`
        The input DataFrame with a column for each spike regressor.

    References
    ----------
    .. [Power2014] Power JD, et al. (2014)
        Methods to detect, characterize, and remove motion artifact in resting
        state fMRI. NeuroImage. doi:`10.1016/j.neuroimage.2013.08.048
        <https://doi.org/10.1016/j.neuroimage.2013.08.048>`__.

    """
    mask = {}
    indices = range(data.shape[0])
    lags = lags or [0]
    criteria = criteria or {
        "framewise_displacement": (">", 0.5),
        "std_dvars": (">", 1.5),
    }
    for metric, (criterion, threshold) in criteria.items():
        if criterion == "<":
            mask[metric] = set(np.where(data[metric] < threshold)[0])
        elif criterion == ">":
            mask[metric] = set(np.where(data[metric] > threshold)[0])
    mask = reduce((lambda x, y: x | y), mask.values())

    for lag in lags:
        mask = set([m + lag for m in mask]) | mask

    mask = mask.intersection(indices)
    if minimum_contiguous is not None:
        post_final = data.shape[0] + 1
        epoch_length = np.diff(sorted(mask | set([-1, post_final]))) - 1
        epoch_end = sorted(mask | set([post_final]))
        for end, length in zip(epoch_end, epoch_length):
            if length < minimum_contiguous:
                mask = mask | set(range(end - length, end))
        mask = mask.intersection(indices)

    if output == "mask":
        spikes = np.zeros(data.shape[0])
        spikes[list(mask)] = 1
        spikes = pd.DataFrame(data=spikes, columns=[header_prefix])
    else:
        spikes = np.zeros((max(indices) + 1, len(mask)))
        for i, m in enumerate(sorted(mask)):
            spikes[m, i] = 1
        header = ["{:s}{:02d}".format(header_prefix, vol) for vol in range(len(mask))]
        spikes = pd.DataFrame(data=spikes, columns=header)
    if concatenate:
        return pd.concat((data, spikes), axis=1)
    else:
        return spikes