"""Step 1: rename ND2 files and crop a 150x150 ROI from every movie.

Copied from SPTnano ROIselector_v2 (the version used for the paper data).

ROISelectorV2 works in two passes:
  1. shows frame 0 of every movie with sliders, you pick the ROI centre and click save;
  2. streams each movie once: dark-frame subtraction, crop, percentile correction, write TIFF.
Outputs: <output_directory>/data/Condition_<name>/<movie>_cropped.tif and saved_data/metadata_summary.csv.
Re-running skips movies that are already cropped.
"""

import gc
import os
import re
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from IPython.display import clear_output, display
from nd2reader import ND2Reader
from skimage import exposure
from tqdm.auto import tqdm


class FlexibleND2Reader:
    """A wrapper around ND2Reader that accepts both .nd2 and .ND2 extensions."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._reader = None
        self._file_handle = None

    def __enter__(self):
        self._file_handle = open(self.filepath, "rb")
        try:
            self._reader = ND2Reader(self._file_handle)
            self._reader.filename = self.filepath
        except Exception:
            self._file_handle.close()
            self._file_handle = None
            raise
        return self._reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._reader:
            self._reader.close()
        if self._file_handle:
            self._file_handle.close()


class ROISelectorV2:
    """
    Two-phase ROI selector.

    Phase 1 opens each source file only long enough to display frame 0 and collect
    ROI metadata. Phase 2 streams each source stack once and writes cropped TIFFs,
    skipping outputs that already exist so interrupted runs can resume.

    Cropped TIFFs are written as one tifffile.imwrite stack ("legacy" mode, the only mode kept).
    """

    def __init__(
        self,
        input_directory,
        output_directory,
        roi_width,
        roi_height,
        split_tiff=False,
        dark_frame_subtraction=False,
        dark_frame_directory=None,
        save_dark_corrected=False,
        ROI_from_metadata=False,
        metadata_path=None,
        percentile_correction=False,
        pmin=1,
        pmax=100,
        tiff_save_mode="legacy",
    ):
        if tiff_save_mode != "legacy":
            raise ValueError('Only tiff_save_mode="legacy" is supported.')

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.split_tiff = split_tiff
        self.dark_frame_subtraction = dark_frame_subtraction
        self.dark_frame_directory = dark_frame_directory
        self.save_dark_corrected = save_dark_corrected
        self.ROI_from_metadata = ROI_from_metadata
        self.metadata_path = metadata_path
        self.metadata_df = None
        self.percentile_correction = percentile_correction
        self.pmin = pmin
        self.pmax = pmax
        self.tiff_save_mode = tiff_save_mode

        if self.ROI_from_metadata and self.metadata_path:
            self.metadata_df = pd.read_csv(self.metadata_path)

        self.metadata = []
        self.conditions = []
        self.condition_paths = {}
        self.jobs = []
        self.pending_roi_jobs = []
        self.pending_index = 0
        self.median_dark_frame = None
        self.median_dark_frames_by_shape = {}
        self.dark_frame_file_shapes = {}
        self.skipped_files = []

    @property
    def metadata_output_path(self):
        return os.path.join(
            self.output_directory,
            "saved_data",
            "metadata_summary.csv",
        )

    @property
    def skipped_files_output_path(self):
        return os.path.join(
            self.output_directory,
            "saved_data",
            "skipped_unreadable_files.csv",
        )

    def prepare_output_folders(self):
        self.condition_paths = self._discover_condition_paths()
        self.conditions = list(self.condition_paths)
        for condition in self.conditions:
            os.makedirs(self._condition_output_directory(condition), exist_ok=True)
        os.makedirs(os.path.join(self.output_directory, "saved_data"), exist_ok=True)

    def process_conditions(self):
        if not self.conditions:
            self.prepare_output_folders()

        self._load_existing_metadata()
        self.jobs = self._discover_jobs()

        if self.ROI_from_metadata and self.metadata_df is not None:
            self._add_external_metadata_for_jobs()

        skipped_keys = {
            (item["Condition"], item["Filename"]) for item in self.skipped_files
        }
        self.pending_roi_jobs = [
            job
            for job in self.jobs
            if not self._has_metadata_for_job(job)
            and not self._output_exists_for_job(job)
            and (job[0], job[1]) not in skipped_keys
        ]
        self.pending_index = 0

        already_cropped_without_metadata = [
            job
            for job in self.jobs
            if not self._has_metadata_for_job(job) and self._output_exists_for_job(job)
        ]
        for condition, file, _input_filepath in already_cropped_without_metadata:
            print(
                "Skipping ROI selection because cropped output already exists "
                f"but metadata is missing: {condition}/{file}",
            )

        if self.pending_roi_jobs:
            self._show_next_roi_selector()
        else:
            print("ROI metadata is complete. Starting resumable crop/save pass.")
            self._print_skipped_files_summary()
            self.process_pending_crops()

    def process_pending_crops(self):
        self.save_metadata()
        self._load_existing_metadata()

        skipped_keys = {
            (item["Condition"], item["Filename"]) for item in self.skipped_files
        }
        for job in tqdm(self.jobs, desc="Cropping files"):
            condition, file, input_filepath = job
            if (condition, file) in skipped_keys:
                print(f"Skipping previously flagged file: {condition}/{file}")
                continue
            if self._output_exists_for_job(job):
                print(f"Skipping existing cropped output: {self._output_path(condition, file)}")
                continue

            row = self._metadata_row_for_job(job)
            if row is None:
                print(f"No ROI metadata found; skipping crop for: {condition}/{file}")
                continue

            try:
                self._crop_and_save_file(condition, file, input_filepath, row)
                gc.collect()
            except Exception as exc:
                gc.collect()
                self._record_skipped_file(condition, file, input_filepath, exc)
                continue

        print("All available ROI crops processed.")
        self._print_skipped_files_summary()

    def subtract_dark_frame(self, image_stack):
        expected_shapes = self._dark_frame_candidate_shapes(image_stack)
        median_dark_frame = self._median_dark_frame_for_shapes(expected_shapes)

        corrected_stack = image_stack - median_dark_frame
        corrected_stack[corrected_stack < 0] = 0
        return corrected_stack

    def extract_median_dark_frame(self, expected_shapes=None):
        if self.dark_frame_directory is None:
            raise ValueError("Dark frame directory is not provided.")

        expected_shapes = self._normalize_expected_shapes(expected_shapes)
        dark_frame_files = sorted(
            f
            for f in os.listdir(self.dark_frame_directory)
            if f.lower().endswith(("nd2", "tif", "tiff"))
        )
        if not dark_frame_files:
            raise FileNotFoundError(
                "No ND2 or TIFF files found in the dark frame directory.",
            )

        dark_frame_path = self._matching_dark_frame_path(
            dark_frame_files,
            expected_shapes,
        )

        if dark_frame_path.lower().endswith("nd2"):
            with FlexibleND2Reader(dark_frame_path) as nd2_dark:
                dark_frames = [np.array(frame) for frame in nd2_dark]
        else:
            with tiff.TiffFile(dark_frame_path) as tif:
                dark_frames = [page.asarray() for page in tif.pages]

        self.median_dark_frame = np.median(np.array(dark_frames), axis=0)
        self.median_dark_frames_by_shape[self.median_dark_frame.shape] = (
            self.median_dark_frame
        )
        del dark_frames
        gc.collect()
        return self.median_dark_frame

    def _median_dark_frame_for_shapes(self, expected_shapes):
        for expected_shape in expected_shapes:
            median_dark_frame = self.median_dark_frames_by_shape.get(expected_shape)
            if median_dark_frame is not None:
                self.median_dark_frame = median_dark_frame
                return median_dark_frame

        return self.extract_median_dark_frame(expected_shapes)

    def _dark_frame_candidate_shapes(self, image_stack):
        image_shape = tuple(image_stack.shape)
        candidate_shapes = [image_shape]
        if image_stack.ndim >= 3 and image_stack.shape[-1] not in (3, 4):
            candidate_shapes.append(tuple(image_stack.shape[-2:]))
        return self._normalize_expected_shapes(candidate_shapes)

    def _normalize_expected_shapes(self, expected_shapes):
        if expected_shapes is None:
            return []

        normalized_shapes = []
        for shape in expected_shapes:
            normalized_shape = tuple(shape)
            if normalized_shape not in normalized_shapes:
                normalized_shapes.append(normalized_shape)
        return normalized_shapes

    def _matching_dark_frame_path(self, dark_frame_files, expected_shapes):
        observed_shapes = []
        for dark_frame_file in dark_frame_files:
            dark_frame_path = os.path.join(self.dark_frame_directory, dark_frame_file)

            dark_frame_shape = self._dark_frame_file_shape(dark_frame_path)
            if not expected_shapes or dark_frame_shape in expected_shapes:
                return dark_frame_path

            observed_shapes.append((dark_frame_file, dark_frame_shape))

        expected_shape_text = ", ".join(str(shape) for shape in expected_shapes)
        observed_shape_text = ", ".join(
            f"{file}: {shape}" for file, shape in observed_shapes
        )
        raise ValueError(
            "No dark frame matched the input frame shape. "
            f"Expected one of: {expected_shape_text}. "
            f"Observed dark frame shapes: {observed_shape_text}",
        )

    def _dark_frame_file_shape(self, dark_frame_path):
        cached_shape = self.dark_frame_file_shapes.get(dark_frame_path)
        if cached_shape is not None:
            return cached_shape

        if dark_frame_path.lower().endswith("nd2"):
            with FlexibleND2Reader(dark_frame_path) as nd2_dark:
                dark_frame_shape = tuple(np.array(nd2_dark[0]).shape)
        else:
            with tiff.TiffFile(dark_frame_path) as tif:
                dark_frame_shape = tuple(tif.pages[0].asarray().shape)

        self.dark_frame_file_shapes[dark_frame_path] = dark_frame_shape
        return dark_frame_shape

    def normalize_mi_ma(self, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        """Normalize an array to the range [0, 1] using specified minimum and maximum values."""
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)
        try:
            import numexpr

            x = numexpr.evaluate("(x - mi) / (ma - mi + eps)")
        except ImportError:
            x = (x - mi) / (ma - mi + eps)
        if clip:
            x = np.clip(x, 0, 1)
        return x

    def apply_percentile_correction(self, image_data):
        global_min = np.percentile(image_data, self.pmin)
        global_max = np.percentile(image_data, self.pmax)
        return self.normalize_mi_ma(
            x=image_data,
            mi=global_min,
            ma=global_max,
            clip=True,
            eps=1e-20,
            dtype=np.float32,
        )

    def save_metadata(self):
        saved_data_dir = os.path.join(self.output_directory, "saved_data")
        os.makedirs(saved_data_dir, exist_ok=True)
        metadata_df = pd.DataFrame(self.metadata, columns=self._metadata_columns())
        if not metadata_df.empty:
            metadata_df = metadata_df.drop_duplicates(
                subset=["Condition", "Filename"],
                keep="last",
            )
            metadata_df = metadata_df.sort_values(["Condition", "Filename"])
            self.metadata = metadata_df.to_dict("records")
        metadata_df.to_csv(self.metadata_output_path, index=False)
        print(f"Metadata saved to CSV: {self.metadata_output_path}")

    def _record_skipped_file(self, condition, file, input_filepath, error):
        record = {
            "Condition": condition,
            "Filename": file,
            "Path": input_filepath,
            "Error": f"{type(error).__name__}: {error}",
        }
        already_logged = any(
            item["Condition"] == condition and item["Filename"] == file
            for item in self.skipped_files
        )
        if not already_logged:
            self.skipped_files.append(record)
            self._save_skipped_files()

        print(
            f"SKIPPED unreadable/failed file: {condition}/{file}\n"
            f"  {record['Error']}\n"
            f"  Moving on to the next file. Log: {self.skipped_files_output_path}",
        )

    def _save_skipped_files(self):
        if not self.skipped_files:
            return
        saved_data_dir = os.path.join(self.output_directory, "saved_data")
        os.makedirs(saved_data_dir, exist_ok=True)
        pd.DataFrame(self.skipped_files).to_csv(
            self.skipped_files_output_path,
            index=False,
        )

    def _print_skipped_files_summary(self):
        if not self.skipped_files:
            return
        print(
            f"Skipped {len(self.skipped_files)} unreadable/failed file(s) "
            f"(also written to {self.skipped_files_output_path}):",
        )
        for item in self.skipped_files:
            print(f"  - {item['Condition']}/{item['Filename']}: {item['Error']}")

    def _discover_jobs(self):
        jobs = []
        for condition in self.conditions:
            condition_path = self.condition_paths[condition]
            files = [
                f
                for f in os.listdir(condition_path)
                if self._is_image_file(f)
            ]
            for file in files:
                jobs.append((condition, file, os.path.join(condition_path, file)))
        return jobs

    def _discover_condition_paths(self):
        condition_paths = {}
        for root, _dirs, files in os.walk(self.input_directory):
            if not any(self._is_image_file(file) for file in files):
                continue

            rel_path = os.path.relpath(root, self.input_directory)
            if rel_path == ".":
                condition = os.path.basename(os.path.abspath(root))
            else:
                condition = self._condition_name_from_relative_path(rel_path)

            if condition in condition_paths and condition_paths[condition] != root:
                condition = self._condition_name_from_relative_path(rel_path)

            condition_paths[condition] = root

        return dict(sorted(condition_paths.items()))

    def _condition_name_from_relative_path(self, rel_path):
        return rel_path.replace("\\", "__").replace("/", "__")

    def _is_image_file(self, file):
        return file.lower().endswith(("tif", "tiff", "nd2"))

    def _external_metadata_row_for_job(self, job):
        condition, file, input_filepath = job
        filename_rows = self.metadata_df[self.metadata_df["Filename"] == file]
        if filename_rows.empty:
            return filename_rows

        condition_matches = filename_rows[filename_rows["Condition"] == condition]
        if not condition_matches.empty:
            return condition_matches

        rel_condition_path = os.path.relpath(
            os.path.dirname(input_filepath),
            self.input_directory,
        )
        candidate_conditions = {
            os.path.basename(os.path.dirname(input_filepath)),
            rel_condition_path,
            rel_condition_path.replace("\\", "/"),
            rel_condition_path.replace("/", "\\"),
        }
        condition_matches = filename_rows[
            filename_rows["Condition"].isin(candidate_conditions)
        ]
        if not condition_matches.empty:
            return condition_matches

        if len(filename_rows) == 1:
            return filename_rows

        return filename_rows.iloc[0:0]

    def _load_existing_metadata(self):
        if os.path.exists(self.metadata_output_path):
            try:
                metadata_df = pd.read_csv(self.metadata_output_path)
            except pd.errors.EmptyDataError:
                return
            if not metadata_df.empty:
                metadata_df = metadata_df.drop_duplicates(
                    subset=["Condition", "Filename"],
                    keep="last",
                )
                self.metadata = metadata_df.to_dict("records")
                print(f"Loaded existing metadata: {self.metadata_output_path}")

    def _add_external_metadata_for_jobs(self):
        for job in self.jobs:
            if self._has_metadata_for_job(job):
                continue

            condition, file, input_filepath = job
            row = self._external_metadata_row_for_job(job)
            if row.empty:
                print(f"No metadata found for file: {condition}/{file}")
                continue

            try:
                first_frame, metadata_info = self._load_first_frame_and_metadata(
                    input_filepath,
                )
            except Exception as exc:
                self._record_skipped_file(condition, file, input_filepath, exc)
                continue
            del first_frame

            self.metadata.append(
                {
                    "Condition": condition,
                    "Filename": file,
                    "ROI_Center_X": int(row["ROI_Center_X"].values[0]),
                    "ROI_Center_Y": int(row["ROI_Center_Y"].values[0]),
                    "ROI_Width": int(row["ROI_Width"].values[0]),
                    "ROI_Height": int(row["ROI_Height"].values[0]),
                    "Date": row["Date"].values[0]
                    if "Date" in row.columns
                    else metadata_info["date"],
                    "Channels": row["Channels"].values[0]
                    if "Channels" in row.columns
                    else metadata_info["channels"],
                    "Pixel_Microns": row["Pixel_Microns"].values[0]
                    if "Pixel_Microns" in row.columns
                    else metadata_info["pixel_microns"],
                    "Num_Frames": row["Num_Frames"].values[0]
                    if "Num_Frames" in row.columns
                    else metadata_info["num_frames"],
                },
            )
        self.save_metadata()

    def _show_next_roi_selector(self):
        while self.pending_index < len(self.pending_roi_jobs):
            condition, file, input_filepath = self.pending_roi_jobs[self.pending_index]
            try:
                first_frame, metadata_info = self._load_first_frame_and_metadata(
                    input_filepath,
                )
                display_frame = self._prepare_display_frame(first_frame)
                del first_frame
            except Exception as exc:
                self._record_skipped_file(condition, file, input_filepath, exc)
                self.pending_index += 1
                gc.collect()
                continue
            break
        else:
            clear_output(wait=True)
            print("All ROI centers specified. Saving metadata before cropping.")
            self._print_skipped_files_summary()
            self.save_metadata()
            self.process_pending_crops()
            return

        height, width = display_frame.shape[:2]

        print(
            f"Selecting ROI {self.pending_index + 1}/{len(self.pending_roi_jobs)}: "
            f"{condition}/{file}",
        )

        preview_output = widgets.Output()

        def draw_preview(_change=None):
            x = x_widget.value
            y = y_widget.value
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(display_frame, cmap="gray")
            ax.set_title(f"Select a central point for ROI - {file}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            roi = plt.Rectangle(
                (x - self.roi_width // 2, y - self.roi_height // 2),
                self.roi_width,
                self.roi_height,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(roi)
            with preview_output:
                clear_output(wait=True)
                plt.show()
            plt.close(fig)

        x_widget = widgets.IntSlider(
            value=width // 2,
            min=0,
            max=width,
            step=1,
            description="X Center:",
            continuous_update=False,
        )
        y_widget = widgets.IntSlider(
            value=height // 2,
            min=0,
            max=height,
            step=1,
            description="Y Center:",
            continuous_update=False,
        )

        button = widgets.Button(description="Set ROI")

        def on_button_clicked(_button):
            center_x = x_widget.value
            center_y = y_widget.value
            clear_output(wait=True)
            print(
                f"ROI center set to: ({center_x}, {center_y}), "
                f"size: ({self.roi_width}, {self.roi_height}) for file: {file}",
            )
            self._print_skipped_files_summary()
            self.metadata.append(
                {
                    "Condition": condition,
                    "Filename": file,
                    "ROI_Center_X": center_x,
                    "ROI_Center_Y": center_y,
                    "ROI_Width": self.roi_width,
                    "ROI_Height": self.roi_height,
                    "Date": metadata_info["date"],
                    "Channels": metadata_info["channels"],
                    "Pixel_Microns": metadata_info["pixel_microns"],
                    "Num_Frames": metadata_info["num_frames"],
                },
            )
            self.save_metadata()
            self.pending_index += 1
            gc.collect()
            self._show_next_roi_selector()

        button.on_click(on_button_clicked)
        x_widget.observe(draw_preview, names="value")
        y_widget.observe(draw_preview, names="value")
        draw_preview()
        display(widgets.VBox([preview_output, x_widget, y_widget, button]))

    def _load_first_frame_and_metadata(self, input_filepath):
        metadata_info = {
            "date": "",
            "channels": "",
            "pixel_microns": "",
            "num_frames": "",
        }

        if input_filepath.lower().endswith(("tif", "tiff")):
            with tiff.TiffFile(input_filepath) as tif:
                first_frame = tif.pages[0].asarray()
                metadata_info["num_frames"] = len(tif.pages)
            return first_frame, metadata_info

        with FlexibleND2Reader(input_filepath) as nd2_file:
            first_frame = np.array(nd2_file[0])
            num_frames = nd2_file.sizes.get("t", len(nd2_file))
            nd2_metadata = nd2_file.metadata
            if "date" in nd2_metadata:
                formatted_date = nd2_metadata["date"].strftime("%Y-%m-%d")
            else:
                formatted_date = "No date available"
            metadata_info = {
                "date": formatted_date,
                "channels": ", ".join(nd2_metadata.get("channels", [])),
                "pixel_microns": nd2_metadata.get(
                    "pixel_microns",
                    "No pixel size available",
                ),
                "num_frames": num_frames,
            }
        return first_frame, metadata_info

    def _prepare_display_frame(self, first_frame):
        first_frame_array = np.array(first_frame)
        p2, p98 = np.percentile(first_frame_array, (2, 98))
        first_frame_array = exposure.rescale_intensity(
            first_frame_array,
            in_range=(p2, p98),
        )
        if np.issubdtype(first_frame_array.dtype, np.integer) or np.issubdtype(
            first_frame_array.dtype,
            np.bool_,
        ):
            return np.invert(first_frame_array)

        display_min = np.nanmin(first_frame_array)
        display_max = np.nanmax(first_frame_array)
        return display_max + display_min - first_frame_array

    def _crop_and_save_file(self, condition, file, input_filepath, metadata_row):
        output_filepath = self._output_path(condition, file)
        temp_output_filepath = output_filepath + ".tmp.tif"
        if os.path.exists(temp_output_filepath):
            os.remove(temp_output_filepath)

        center_x = int(metadata_row["ROI_Center_X"])
        center_y = int(metadata_row["ROI_Center_Y"])
        roi_width = int(metadata_row["ROI_Width"])
        roi_height = int(metadata_row["ROI_Height"])
        box = (
            center_x - roi_width // 2,
            center_y - roi_height // 2,
            center_x + roi_width // 2,
            center_y + roi_height // 2,
        )

        print(f"Cropping file: {input_filepath}")
        self._crop_and_save_file_legacy(
            file,
            input_filepath,
            output_filepath,
            temp_output_filepath,
            box,
        )

    def _crop_and_save_file_legacy(
        self,
        file,
        input_filepath,
        output_filepath,
        temp_output_filepath,
        box,
    ):
        cropped_frames = []
        dark_corrected_frames = [] if self.save_dark_corrected else None
        original_min = None
        original_max = None
        output_dtype = None

        try:
            iterator = self._iter_frames(input_filepath)
            total = self._frame_count(input_filepath)
            for frame in tqdm(iterator, desc=f"Saving {file}", total=total):
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                original_min = frame_min if original_min is None else min(original_min, frame_min)
                original_max = frame_max if original_max is None else max(original_max, frame_max)

                if self.dark_frame_subtraction:
                    frame = self.subtract_dark_frame(frame)

                if output_dtype is None:
                    output_dtype = frame.dtype

                if dark_corrected_frames is not None:
                    dark_corrected_frames.append(frame)

                cropped_frames.append(self._crop_array_with_padding(frame, box))

            if input_filepath.lower().endswith("nd2") and self.percentile_correction:
                cropped_frames = [
                    self.apply_percentile_correction(frame) for frame in cropped_frames
                ]
                print("Percentile correction applied to cropped frames.")
                cropped_frames = [
                    (frame * (original_max - original_min) + original_min).astype(output_dtype)
                    for frame in cropped_frames
                ]

            if dark_corrected_frames is not None:
                dark_corrected_output = (
                    os.path.splitext(input_filepath)[0] + "_dark_corrected.tif"
                )
                tiff.imwrite(
                    dark_corrected_output,
                    dark_corrected_frames,
                    photometric="minisblack",
                )
                print(f"Dark-corrected file saved to: {dark_corrected_output}")

            tiff.imwrite(temp_output_filepath, cropped_frames, photometric="minisblack")
            os.replace(temp_output_filepath, output_filepath)
            print(f"Finished saving {output_filepath}.")
        except Exception:
            if os.path.exists(temp_output_filepath):
                os.remove(temp_output_filepath)
            raise
        finally:
            cropped_frames.clear()
            if dark_corrected_frames is not None:
                dark_corrected_frames.clear()
            gc.collect()

    def _iter_frames(self, input_filepath):
        if input_filepath.lower().endswith(("tif", "tiff")):
            with tiff.TiffFile(input_filepath) as tif:
                for page in tif.pages:
                    yield page.asarray()
            return

        with FlexibleND2Reader(input_filepath) as nd2_file:
            for frame in nd2_file:
                yield np.array(frame)

    def _frame_count(self, input_filepath):
        if input_filepath.lower().endswith(("tif", "tiff")):
            with tiff.TiffFile(input_filepath) as tif:
                return len(tif.pages)

        with FlexibleND2Reader(input_filepath) as nd2_file:
            return nd2_file.sizes.get("t", len(nd2_file))

    def _raw_min_max_and_dtype(self, input_filepath):
        original_min = None
        original_max = None
        original_dtype = None
        for frame in self._iter_frames(input_filepath):
            if original_dtype is None:
                original_dtype = frame.dtype
            frame_min = np.min(frame)
            frame_max = np.max(frame)
            original_min = frame_min if original_min is None else min(original_min, frame_min)
            original_max = frame_max if original_max is None else max(original_max, frame_max)
        return original_min, original_max, original_dtype

    def _crop_array_with_padding(self, frame, box):
        left, upper, right, lower = box
        crop_width = right - left
        crop_height = lower - upper
        if frame.ndim == 2:
            cropped = np.zeros((crop_height, crop_width), dtype=frame.dtype)
        else:
            cropped = np.zeros((crop_height, crop_width, *frame.shape[2:]), dtype=frame.dtype)

        frame_height, frame_width = frame.shape[:2]
        src_left = max(left, 0)
        src_upper = max(upper, 0)
        src_right = min(right, frame_width)
        src_lower = min(lower, frame_height)

        if src_left >= src_right or src_upper >= src_lower:
            return cropped

        dst_left = src_left - left
        dst_upper = src_upper - upper
        dst_right = dst_left + (src_right - src_left)
        dst_lower = dst_upper + (src_lower - src_upper)
        cropped[dst_upper:dst_lower, dst_left:dst_right] = frame[
            src_upper:src_lower,
            src_left:src_right,
        ]
        return cropped

    def _metadata_columns(self):
        return [
            "Condition",
            "Filename",
            "ROI_Center_X",
            "ROI_Center_Y",
            "ROI_Width",
            "ROI_Height",
            "Date",
            "Channels",
            "Pixel_Microns",
            "Num_Frames",
        ]

    def _output_path(self, condition, file):
        condition_output_directory = self._condition_output_directory(condition)
        return os.path.join(
            condition_output_directory,
            os.path.splitext(file)[0] + "_cropped.tif",
        )

    def _condition_output_directory(self, condition):
        return os.path.join(self.output_directory, "data", f"Condition_{condition}")

    def _output_exists_for_job(self, job):
        condition, file, _input_filepath = job
        output_filepath = self._output_path(condition, file)
        return os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0

    def _has_metadata_for_job(self, job):
        return self._metadata_row_for_job(job) is not None

    def _metadata_row_for_job(self, job):
        condition, file, _input_filepath = job
        filename_matches = []
        for row in reversed(self.metadata):
            if row.get("Condition") == condition and row.get("Filename") == file:
                return row
            if row.get("Filename") == file:
                filename_matches.append(row)
        if len(filename_matches) == 1:
            return filename_matches[0]
        return None


TRAILING_NUMBER = re.compile(r"^(.*)_(\d+)$")


def insert_suffix_before_number(folder, suffix, dry_run=True):
    """MovieName_035.nd2 + "_dday-0" -> MovieName_dday-0_035.nd2, for every .nd2 in one folder (not subfolders)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    already_tagged = re.compile(re.escape(suffix) + r"_\d+$", re.IGNORECASE)

    planned, skipped = [], []
    for nd2_path in sorted(folder.iterdir()):
        if not nd2_path.is_file() or nd2_path.suffix.lower() != ".nd2":
            continue
        stem = nd2_path.stem
        if already_tagged.search(stem):
            skipped.append((nd2_path.name, "already has suffix"))
            continue
        match = TRAILING_NUMBER.match(stem)
        if not match:
            skipped.append((nd2_path.name, "no trailing _number before .nd2"))
            continue
        prefix, number = match.groups()
        new_path = nd2_path.with_name(f"{prefix}{suffix}_{number}{nd2_path.suffix}")
        if new_path.exists():
            skipped.append((nd2_path.name, f"target already exists: {new_path.name}"))
            continue
        planned.append((nd2_path, new_path))

    _apply_renames(planned, skipped, dry_run)
    return planned


def replace_in_filenames(folder, old, new, dry_run=True):
    """Replace text in every filename in one folder (not subfolders), e.g. fix a wrong genotype."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    planned, skipped = [], []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if old not in path.name:
            skipped.append((path.name, f"no {old}"))
            continue
        new_path = path.with_name(path.name.replace(old, new))
        if new_path.exists():
            skipped.append((path.name, f"target already exists: {new_path.name}"))
            continue
        planned.append((path, new_path))

    _apply_renames(planned, skipped, dry_run)
    return planned


def _apply_renames(planned, skipped, dry_run):
    print(f"Mode: {'DRY RUN' if dry_run else 'RENAME'}   planned: {len(planned)}   skipped: {len(skipped)}")
    for old_path, new_path in planned:
        print(f"{old_path.name}  ->  {new_path.name}")
        if not dry_run:
            old_path.rename(new_path)
    for name, reason in skipped:
        print(f"  skipped {name}  ({reason})")
    if dry_run:
        print("Dry run only. Set dry_run=False to apply.")
