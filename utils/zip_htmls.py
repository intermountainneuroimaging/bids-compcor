"""Compress HTML files."""

import datetime
import glob
import logging
import os
from pathlib import Path
from bs4 import BeautifulSoup
from zipfile import ZIP_DEFLATED, ZipFile


log = logging.getLogger(__name__)


def zip_it_zip_it_good(output_dir, destination_id, name):
    """Compress html file into an appropriately named archive file *.html.zip
    files are automatically shown in another tab in the browser. These are
    saved at the top level of the output folder."""

    name_no_html = name[:-5]  # remove ".html" from end

    dest_zip = os.path.join(
        output_dir, name_no_html + "_" + destination_id + ".html.zip"
    )

    log.info('Creating viewable archive "' + dest_zip + '"')
    
    logging.info("Zipping html at location: %s", os.path.abspath(os.curdir))
    
    # find all references in html and include in zip command
    with open("index.html") as inf:
        txt = inf.read()
    soup = BeautifulSoup(txt, 'html.parser')

    # Find the elements in the file and alter path so its relative
    zipfiles=[]
    for a in soup.find_all('a'):
        if os.path.exists(a['href']):
            zipfiles.append(os.path.relpath(Path(a['href'])))
    
    with ZipFile(dest_zip, "w", ZIP_DEFLATED) as outzip:
        outzip.write("index.html")
        for fl in zipfiles:
            outzip.write(fl)


def zip_htmls(output_dir, destination_id, path):
    """Zip all .html files at the given path so they can be displayed
    on the Flywheel platform.
    Each html file must be converted into an archive individually:
      rename each to be "index.html", then create a zip archive from it.
    """

    log.info("Creating viewable archives for all html files")

    if os.path.exists(path):

        log.info("Found path: " + str(path))

        FWV0 = Path.cwd()
        os.chdir(path)

        html_files = glob.glob("*.html")

        if len(html_files) > 0:

            # if there is an index.html, do it first and re-name it for safe
            # keeping
            save_name = ""
            if os.path.exists("index.html"):
                log.info("Found index.html")
                zip_it_zip_it_good(output_dir, destination_id, "index.html")

                now = datetime.datetime.now()
                save_name = now.strftime("%Y-%m-%d_%H-%M-%S") + "_index.html"
                os.rename("index.html", save_name)

                html_files.remove("index.html")  # don't do this one later

            for h_file in html_files:
                os.rename(h_file, "index.html")
                try:
                    zip_it_zip_it_good(output_dir, destination_id, h_file)
                except:
                    raise
                finally:
                    os.rename("index.html", h_file)

            # reestore if necessary
            if save_name != "":
                os.rename(save_name, "index.html")

        else:
            log.warning("No *.html files at " + str(path))

        os.chdir(FWV0)

    else:

        log.error("Path NOT found: " + str(path))
