"""Step 2a: per-frame and per-window track features (ParticleMetrics).

Copied unchanged from SPTnano features.py, apart from the default frame interval.
Used twice:
  calculate_all_features(calculate_time_windowed=False, calculate_time_averaged=False) -> instant_df.csv
  calculate_time_windowed_metrics(...)  (run in parallel by sptnano.windows)           -> window table
"""

import re

import numpy as np
import pandas as pd
import scipy.optimize

# from scipy.stats import vonmises
from scipy.stats import kurtosis, skew, vonmises
from tqdm.auto import tqdm

DEFAULT_FRAME_INTERVAL_S = 0.01


class ParticleMetrics:
    def __init__(self, df, time_between_frames=None, tolerance=None):
        self.df = df.copy()
        # self.df['location'] = self.df['filename'].apply(self.extract_location)  # Add Location column
        # self.df['molecule'] = self.df['filename'].apply(self.extract_molecule)  # Add Molecule column
        # self.df['genotype'] = self.df['filename'].apply(self.extract_genotype)  # Add Genotype column
        # self.df['cell_type'] = self.df['filename'].apply(self.extract_cell_type)  # Add Cell Type column
        self.df["location"] = self.df["filename"].apply(
            self.extract_location
        )  # Add Location column
        self.df["molecule"] = self.df["condition"].apply(
            self.extract_molecule
        )  # Add Molecule column
        self.df["genotype"] = self.df["condition"].apply(
            self.extract_genotype
        )  # Add Genotype column
        self.df["cell_type"] = self.df["condition"].apply(
            self.extract_cell_type
        )  # Add Cell Type column
        self.metrics_df = self.df.copy()

        # Use the default frame interval if not provided
        self.time_between_frames = (
            time_between_frames
            if time_between_frames is not None
            else DEFAULT_FRAME_INTERVAL_S
        )

        self.time_averaged_df = pd.DataFrame(
            columns=[
                "x_um_start",
                "y_um_start",
                "x_um_end",
                "y_um_end",
                "particle",
                "condition",
                "filename",
                "file_id",
                "unique_id",
                "avg_msd",
                "n_frames",
                "total_time_s",
                "location",
                "molecule",
                "genotype",
                "cell_type",
                "diffusion_coefficient",
                "anomalous_exponent",
                "motion_class",
                "avg_speed_um_s",
                "avg_acceleration_um_s2",
                "avg_jerk_um_s3",
                "avg_normalized_curvature",
                "avg_angle_normalized_curvature",
                "kappa_turning",
                "kappa_absolute",
                "net_displacement_um",
                "cum_displacement_um",
                # NEW features in time-averaged summary
                "straightness_index",
                "radius_of_gyration",
                "convex_hull_area",
                "directional_entropy",
                "speed_variability",
                "direction_autocorrelation",
                "eccentricity",
                "turning_angle_variance",
                "turning_angle_skew",
                "turning_angle_kurtosis",
                "steplength_mean",
                "steplength_std",
                "steplength_skew",
                "steplength_kurtosis",
                "diffusivity_cv",
                "fractal_dimension",
                "psd_slope_speed",
                "self_intersections",
                "pausing_fraction",
            ]
        )

        self.time_windowed_df = pd.DataFrame(
            columns=[
                "time_window",
                "x_um_start",
                "y_um_start",
                "x_um_end",
                "y_um_end",
                "particle",
                "condition",
                "filename",
                "file_id",
                "unique_id",
                "frame_start",
                "frame_end",
                "e_uid",
                "window_uid",
                "split_count",
                "avg_msd",
                "n_frames",
                "total_time_s",
                "location",
                "molecule",
                "genotype",
                "cell_type",
                "diffusion_coefficient",
                "anomalous_exponent",
                "motion_class",
                "avg_speed_um_s",
                "avg_acceleration_um_s2",
                "avg_jerk_um_s3",
                "avg_normalized_curvature",
                "avg_angle_normalized_curvature",
                "persistence_length",
                "net_displacement_um",
                "cum_displacement_um",
                "kappa_turning",
                "kappa_absolute",
                # NEW features in time-windowed summary:
                "straightness_index",
                "radius_of_gyration",
                "convex_hull_area",
                "directional_entropy",
                "speed_variability",
                "direction_autocorrelation",
                "eccentricity",
                "turning_angle_variance",
                "turning_angle_skew",
                "turning_angle_kurtosis",
                "steplength_mean",
                "steplength_std",
                "steplength_skew",
                "steplength_kurtosis",
                "diffusivity_cv",
                "fractal_dimension",
                "psd_slope_speed",
                "self_intersections",
                "pausing_fraction",
                "bad_fit_flag",
            ]
        )

        self.msd_lagtime_df = pd.DataFrame(
            columns=[
                "unique_id",
                "time_window",
                "lag_time",
                "msd",
                "diffusion_coefficient",
                "anomalous_exponent",
            ]
        )

        # Calculate tolerance if not provided
        self.tolerance = tolerance or self.calculate_tolerance()

    def calculate_tolerance(self):
        # Determine tolerance from existing data if available
        if len(self.msd_lagtime_df) > 0:
            alpha_std = self.msd_lagtime_df["anomalous_exponent"].std()
            return alpha_std / 2  # Use half of the standard deviation
        else:
            return 0.1  # Default value if no previous data is available

    @staticmethod
    def extract_location(filename):
        match = re.search(r"loc-(\w+)(?:_|$)", filename)
        if match:
            return match.group(1).upper()  # Convert to uppercase
        return "Unknown"  # Default value if no location is found

    @staticmethod
    def extract_molecule(filename):
        match = re.search(r"mol-(\w+)(?:_|$)", filename)
        if match:
            return match.group(1)
        return "Unknown"  # Default value if no molecule is found

    @staticmethod
    def extract_genotype(filename):
        match = re.search(r"geno-(\w+)(?:_|$)", filename)
        if match:
            return match.group(1)
        return "Unknown"  # Default value if no genotype is found

    @staticmethod
    def extract_cell_type(filename):
        match = re.search(r"type-(\w+)(?:_|$)", filename)
        if match:
            return match.group(1)
        return "Unknown"  # Default value if no cell type is found

    @staticmethod
    def msd_model(t, D, alpha):
        return 4 * D * t**alpha

    def calculate_distances(self):
        """
        Calculate the distances between consecutive frames for each particle in micrometers.
        Skip calculation if 'segment_len_um' already exists.
        """
        if "segment_len_um" not in self.metrics_df.columns:
            self.metrics_df = self.metrics_df.sort_values(by=["unique_id", "frame"])
            self.metrics_df[["x_um_prev", "y_um_prev"]] = self.metrics_df.groupby(
                "unique_id"
            )[["x_um", "y_um"]].shift(1)
            self.metrics_df["segment_len_um"] = np.sqrt(
                (self.metrics_df["x_um"] - self.metrics_df["x_um_prev"]) ** 2
                + (self.metrics_df["y_um"] - self.metrics_df["y_um_prev"]) ** 2
            )
            self.metrics_df["segment_len_um"] = self.metrics_df[
                "segment_len_um"
            ].fillna(0)
        return self.metrics_df

    def calculate_speeds(self):
        """
        Calculate the speed between consecutive frames for each particle in micrometers per second.
        """
        self.metrics_df[["time_s_prev"]] = self.metrics_df.groupby("unique_id")[
            ["time_s"]
        ].shift(1)
        self.metrics_df["delta_time_s"] = (
            self.metrics_df["time_s"] - self.metrics_df["time_s_prev"]
        )
        self.metrics_df["speed_um_s"] = (
            self.metrics_df["segment_len_um"] / self.metrics_df["delta_time_s"]
        )
        # Fill NaN and infinite values with 0
        self.metrics_df["speed_um_s"] = (
            self.metrics_df["speed_um_s"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        return self.metrics_df

    def calculate_directions(self):
        """
        Calculate the direction of motion between consecutive frames for each particle in radians.
        """
        self.metrics_df["direction_rad"] = np.arctan2(
            self.metrics_df["y_um"] - self.metrics_df["y_um_prev"],
            self.metrics_df["x_um"] - self.metrics_df["x_um_prev"],
        )
        # Fill NaN values with 0
        self.metrics_df["direction_rad"] = self.metrics_df["direction_rad"].fillna(0)
        return self.metrics_df

    def calculate_accelerations(self):
        """
        Calculate the acceleration between consecutive frames for each particle in micrometers per second squared.
        """
        self.metrics_df[["speed_um_s_prev"]] = self.metrics_df.groupby("unique_id")[
            ["speed_um_s"]
        ].shift(1)
        self.metrics_df["acceleration_um_s2"] = (
            self.metrics_df["speed_um_s"] - self.metrics_df["speed_um_s_prev"]
        ) / self.metrics_df["delta_time_s"]
        # Fill NaN and infinite values with 0
        self.metrics_df["acceleration_um_s2"] = (
            self.metrics_df["acceleration_um_s2"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        return self.metrics_df

    def calculate_jerk(self):
        """
        Calculate the jerk between consecutive frames for each particle in micrometers per second cubed.
        """
        self.metrics_df[["acceleration_um_s2_prev"]] = self.metrics_df.groupby(
            "unique_id"
        )[["acceleration_um_s2"]].shift(1)
        self.metrics_df["jerk_um_s3"] = (
            self.metrics_df["acceleration_um_s2"]
            - self.metrics_df["acceleration_um_s2_prev"]
        ) / self.metrics_df["delta_time_s"]
        # Fill NaN and infinite values with 0
        self.metrics_df["jerk_um_s3"] = (
            self.metrics_df["jerk_um_s3"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        return self.metrics_df

    def calculate_normalized_curvature(self):
        """
        Calculate the curvature normalized by distance between consecutive frames for each particle.
        """
        self.metrics_df[["direction_rad_prev"]] = self.metrics_df.groupby("unique_id")[
            ["direction_rad"]
        ].shift(1)
        self.metrics_df["normalized_curvature"] = (
            self.metrics_df["direction_rad"] - self.metrics_df["direction_rad_prev"]
        ) / self.metrics_df["segment_len_um"]
        # Fill NaN and infinite values with 0
        self.metrics_df["normalized_curvature"] = (
            self.metrics_df["normalized_curvature"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        # self.metrics_df['normalized_curvature_deg'] = np.degrees(self.metrics_df['normalized_curvature'])
        return self.metrics_df

    def calculate_angle_normalized_curvature(self):
        """
        Calculate the curvature (change in direction) normalized to the range [-pi, pi] between consecutive frames for each particle.
        """
        self.metrics_df[["direction_rad_prev"]] = self.metrics_df.groupby("unique_id")[
            ["direction_rad"]
        ].shift(1)
        self.metrics_df["angle_normalized_curvature"] = (
            self.metrics_df["direction_rad"] - self.metrics_df["direction_rad_prev"]
        )
        # Normalize curvature to the range [-pi, pi]
        self.metrics_df["angle_normalized_curvature"] = (
            self.metrics_df["angle_normalized_curvature"] + np.pi
        ) % (2 * np.pi) - np.pi
        # Fill NaN values with 0
        self.metrics_df["angle_normalized_curvature"] = self.metrics_df[
            "angle_normalized_curvature"
        ].fillna(0)
        # Convert columns from radians to degrees

        # self.metrics_df['angle_normalized_curvature_deg'] = np.degrees(self.metrics_df['angle_normalized_curvature'])
        return self.metrics_df

    def calculate_vonmises_kappa(self, window_data):
        """
        Fit von Mises distributions to the turning angles and absolute directions
        in the given window data.

        Parameters
        - window_data: DataFrame containing tracking data for the window.

        Returns
        - kappa_turning: Estimated concentration parameter for turning angles.
        - kappa_absolute: Estimated concentration parameter for absolute directions.

        """
        # Use the pre-computed columns:
        # For turning angles, we are using "angle_normalized_curvature"
        turning_angles = window_data["angle_normalized_curvature"].values
        # For absolute directions, use "direction_rad"
        absolute_angles = window_data["direction_rad"].values

        # Fit the von Mises distribution for turning angles.
        try:
            # Optionally you can set fscale=1 to fix the scale parameter.
            params_turning = vonmises.fit(turning_angles, fscale=1)
            kappa_turning = params_turning[1]
        except Exception as e:
            print(f"Von Mises fit for turning angles failed: {e}")
            kappa_turning = np.nan

        # Fit the von Mises distribution for absolute directions.
        try:
            params_absolute = vonmises.fit(absolute_angles, fscale=1)
            kappa_absolute = params_absolute[1]
        except Exception as e:
            print(f"Von Mises fit for absolute angles failed: {e}")
            kappa_absolute = np.nan

        return kappa_turning, kappa_absolute

    # def calculate_persistence_length(self, track_data):
    #     """
    #     Calculate the persistence length for a given track.
    #     Parameters:
    #     - track_data: DataFrame containing the track data.
    #     Returns:
    #     - persistence_length: Calculated persistence length for the track.
    #     """
    #     directions = track_data['direction_rad']
    #     # Calculate directional correlation: <cos(theta_i - theta_j)>
    #     direction_diffs = directions.diff().dropna()
    #     correlation = np.cos(direction_diffs).mean()
    #         # Check if correlation is greater than zero before calculating log
    #     if correlation > 0:
    #         persistence_length = -1 / np.log(correlation)
    #     else:
    #         persistence_length = np.nan  # Assign NaN if the correlation is zero or negative

    #     # persistence_length = -1 / np.log(correlation) if correlation != 0 else np.nan
    #     return persistence_length

    def calculate_persistence_length(self, track_data):  # updated 3-6-2025
        """
        Calculate the persistence length for a given track (or time window)
        in physical units. If the directional correlation is non-positive, returns 0.

        Parameters
        - track_data: DataFrame containing the track data (or time window data).

        Returns
        - persistence_length: Calculated persistence length for the track in physical units.

        """
        directions = track_data["direction_rad"]
        # Compute the differences in direction between consecutive frames.
        direction_diffs = directions.diff().dropna()
        # Calculate the average cosine of these differences.
        correlation = np.cos(direction_diffs).mean()

        if correlation > 0:
            # Persistence length in "step" units.
            persistence_length_steps = -1 / np.log(correlation)
            # Calculate the average segment length (in physical units, e.g., micrometers) over this window.
            avg_segment_length = track_data["segment_len_um"].mean()
            # Scale to obtain the persistence length in physical units.
            persistence_length = persistence_length_steps * avg_segment_length
        else:
            # Instead of returning NaN, return 0.
            persistence_length = 0

        return persistence_length

    def calculate_net_displacement(self):
        """
        Calculate the net displacement from the starting point for each particle in micrometers.
        """
        self.metrics_df[["x_um_start", "y_um_start"]] = self.metrics_df.groupby(
            "unique_id"
        )[["x_um", "y_um"]].transform("first")
        self.metrics_df["net_displacement_um"] = np.sqrt(
            (self.metrics_df["x_um"] - self.metrics_df["x_um_start"]) ** 2
            + (self.metrics_df["y_um"] - self.metrics_df["y_um_start"]) ** 2
        )
        return self.metrics_df

    def calculate_cum_displacement(self):  # ADDED 3_19_2025
        # Ensure that segment lengths are computed
        self.calculate_distances()
        # Compute cumulative (total) displacement for each particle
        self.metrics_df["cum_displacement_um"] = self.metrics_df.groupby("unique_id")[
            "segment_len_um"
        ].cumsum()
        return self.metrics_df

    # def calculate_cumulative_displacement(df):
    #     """
    #     Calculates the cumulative displacement (total distance traveled) for each particle.

    #     Parameters:
    #     - df: DataFrame containing columns 'unique_id', 'frame', 'x_um', and 'y_um'.

    #     Returns:
    #     - df: DataFrame with a new column 'cumulative_displacement_um', which is the
    #         cumulative sum of distances (segment lengths) between consecutive frames.
    #     """
    #     # Ensure the data is sorted by unique_id and frame
    #     df = df.sort_values(by=['unique_id', 'frame']).copy()

    #     # Compute segment lengths if not already present
    #     if 'segment_len_um' not in df.columns:
    #         # Calculate previous positions for each particle
    #         df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
    #         # Compute the Euclidean distance between consecutive positions
    #         df['segment_len_um'] = ((df['x_um'] - df['x_um_prev'])**2 +
    #                                 (df['y_um'] - df['y_um_prev'])**2)**0.5
    #         # Replace NaN for the first frame of each particle with 0
    #         df['segment_len_um'] = df['segment_len_um'].fillna(0)

    #     # Compute the cumulative sum of segment lengths for each particle
    #     df['cumulative_displacement_um'] = df.groupby('unique_id')['segment_len_um'].cumsum()
    #     return df

    def calculate_instantaneous_diffusion_coefficient(self):
        """
        Calculate the instantaneous diffusion coefficient for each particle in square micrometers per second.
        """
        self.metrics_df["instant_diff_coeff"] = self.metrics_df[
            "segment_len_um"
        ] ** 2 / (4 * self.metrics_df["delta_time_s"])
        # Fill NaN and infinite values with 0
        self.metrics_df["instant_diff_coeff"] = (
            self.metrics_df["instant_diff_coeff"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        return self.metrics_df

    def calculate_instantaneous_velocity(self):
        """
        Calculate the instantaneous velocity between consecutive frames for each particle in micrometers per second.
        """
        self.metrics_df["instant_velocity_x_um_s"] = (
            self.metrics_df["x_um"] - self.metrics_df["x_um_prev"]
        ) / self.metrics_df["delta_time_s"]
        self.metrics_df["instant_velocity_y_um_s"] = (
            self.metrics_df["y_um"] - self.metrics_df["y_um_prev"]
        ) / self.metrics_df["delta_time_s"]
        # Fill NaN and infinite values with 0
        self.metrics_df["instant_velocity_x_um_s"] = (
            self.metrics_df["instant_velocity_x_um_s"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        self.metrics_df["instant_velocity_y_um_s"] = (
            self.metrics_df["instant_velocity_y_um_s"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        return self.metrics_df

    ############# ADDITIONAL METRICS FUNCTIONS APRIL 2025 ################################
    def calculate_straightness_index(self, data):
        """
        Compute straightness as the ratio of net displacement to cumulative displacement.
        """
        net_disp = data["net_displacement_um"].iloc[-1]
        cum_disp = data["cum_displacement_um"].iloc[-1]
        return net_disp / cum_disp if cum_disp > 0 else np.nan

    def calculate_radius_of_gyration(self, data):
        """
        Compute the radius of gyration as the root mean square distance from the centroid.
        """
        x = data["x_um"].values
        y = data["y_um"].values
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        rg = np.sqrt(np.mean((x - centroid_x) ** 2 + (y - centroid_y) ** 2))
        return rg

    def calculate_convex_hull_area(self, data):
        """
        Compute the convex hull area of the points.
        """
        from scipy.spatial import ConvexHull

        points = data[["x_um", "y_um"]].values
        if len(points) < 3:
            return 0.0  # Not enough points for a hull
        hull = ConvexHull(points)
        # For 2D, hull.volume gives the area
        return hull.volume

    def calculate_directional_entropy(self, data, n_bins=18):
        """
        Compute Shannon entropy from the distribution of turning angles.
        n_bins controls the resolution: default 18 bins (approx. 20° per bin over [-π, π]).
        """
        angles = data["angle_normalized_curvature"].dropna().values
        # Compute a histogram with the specified number of bins over the range [-pi, pi]
        hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi), density=True)
        # Remove zero bins to avoid log(0)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        return entropy

    def calculate_speed_variability(self, data):
        """
        Compute the coefficient of variation of the instantaneous speeds.
        """
        speeds = data["speed_um_s"].values
        if np.mean(speeds) > 0:
            return np.std(speeds) / np.mean(speeds)
        return np.nan

    def calculate_directional_autocorrelation(self, data):
        """
        Compute the lag-1 directional autocorrelation using cosine similarity of differences.
        """
        directions = data["direction_rad"].values
        if len(directions) < 2:
            return np.nan
        autocorr = np.mean(np.cos(np.diff(directions)))
        return autocorr

    # 1. Trajectory Eccentricity and Orientation (we compute eccentricity here)
    def calculate_trajectory_eccentricity(self, data):
        """
        Calculate eccentricity as a measure of elongation of the distribution of points.
        Uses the covariance matrix of x and y coordinates.
        Returns a single scalar value.
        """
        x = data["x_um"].values
        y = data["y_um"].values
        if len(x) < 2:
            return np.nan
        cov_mat = np.cov(x, y)
        eig_vals, _ = np.linalg.eig(cov_mat)
        eig_vals = np.sort(eig_vals)[::-1]  # Ensure first eigenvalue is largest
        if eig_vals[0] > 0:
            eccentricity = np.sqrt(1 - eig_vals[1] / eig_vals[0])
        else:
            eccentricity = np.nan
        return eccentricity

    # 2. Turning Angle Distribution Moments (variance, skewness, kurtosis)
    def calculate_turning_angle_moments(self, data):
        """
        Compute the variance, skewness, and kurtosis of the turning angles.
        Uses the 'angle_normalized_curvature' column.
        """
        angles = data["angle_normalized_curvature"].dropna().values
        if len(angles) < 2:
            return np.nan, np.nan, np.nan
        variance = np.var(angles)
        skewness = skew(angles)
        kurt = kurtosis(angles)
        return variance, skewness, kurt

    # 3. Step-Length Distribution Statistics
    def calculate_steplength_distribution_stats(self, data):
        """
        Compute mean, standard deviation, skewness, and kurtosis of step lengths.
        Uses the 'segment_len_um' column.
        """
        if "segment_len_um" not in data.columns:
            # Ensure distances are computed
            data = self.calculate_distances()
        steps = data["segment_len_um"].values
        if len(steps) < 2:
            return np.nan, np.nan, np.nan, np.nan
        step_mean = np.mean(steps)
        step_std = np.std(steps)
        step_skew = skew(steps)
        step_kurt = kurtosis(steps)
        return step_mean, step_std, step_skew, step_kurt

    # 4. Instantaneous Diffusivity Fluctuations
    def calculate_instantaneous_diffusivity_variability(self, data):
        """
        Compute the coefficient of variation (CV) of instantaneous diffusivity.
        Instantaneous diffusivity D_inst = (segment_len_um)^2 / (4 * delta_time_s).
        Assumes 'delta_time_s' and 'segment_len_um' are computed.
        """
        if "delta_time_s" not in data.columns:
            data = self.calculate_speeds()
        # Avoid division by zero:
        valid = data["delta_time_s"] > 0
        if not valid.any():
            return np.nan
        D_inst = (data.loc[valid, "segment_len_um"] ** 2) / (
            4 * data.loc[valid, "delta_time_s"]
        )
        if np.mean(D_inst) > 0:
            return np.std(D_inst) / np.mean(D_inst)
        else:
            return np.nan

    # 5. Fractal Dimension (Box-Counting Method)
    def calculate_fractal_dimension(self, data):
        """
        Estimate the fractal dimension using a simple box-counting algorithm.
        """
        x = data["x_um"].values
        y = data["y_um"].values
        points = np.vstack((x, y)).T
        if len(points) < 2:
            return np.nan

        # Define box sizes as a range from the minimum to maximum distance
        min_box = np.min(np.ptp(points, axis=0)) / 10  # small fraction of the spread
        max_box = np.max(np.ptp(points, axis=0))
        if min_box <= 0:
            return np.nan
        n_boxes = np.logspace(np.log10(min_box), np.log10(max_box), num=10)
        counts = []
        for box_size in n_boxes:
            # Shift points to positive quadrant
            shifted = points - points.min(axis=0)
            # Divide the plane into boxes of size box_size and count non-empty boxes.
            bins_x = np.ceil(shifted[:, 0] / box_size).astype(int)
            bins_y = np.ceil(shifted[:, 1] / box_size).astype(int)
            boxes = set(zip(bins_x, bins_y, strict=False))
            counts.append(len(boxes))

        # Perform a linear fit on log-log data; the slope estimates the fractal dimension.
        try:
            coeffs = np.polyfit(np.log(n_boxes), np.log(counts), 1)
            fractal_dim = -coeffs[0]
        except Exception:
            fractal_dim = np.nan
        return fractal_dim

    # 6. Power Spectral Density (PSD) Slope of Speed
    def calculate_psd_slope(self, data, signal="speed"):
        """
        Compute the slope of the power spectral density (PSD) in log–log space.
        For example, if signal == 'speed', use the instantaneous speed values.
        """
        if signal == "speed":
            if "speed_um_s" not in data.columns:
                data = self.calculate_speeds()
            s = data["speed_um_s"].dropna().values
        elif signal == "acceleration":
            if "acceleration_um_s2" not in data.columns:
                data = self.calculate_accelerations()
            s = data["acceleration_um_s2"].dropna().values
        else:
            return np.nan

        if len(s) < 4:
            return np.nan

        # Compute Fourier transform and PSD.
        fft_vals = np.fft.fft(s)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.fftfreq(len(s))
        # Consider only positive frequencies.
        positive = freqs > 0
        if positive.sum() < 2:
            return np.nan
        log_freqs = np.log10(freqs[positive])
        log_psd = np.log10(psd[positive])
        # Linear fit to get the slope.
        try:
            slope, _ = np.polyfit(log_freqs, log_psd, 1)
        except Exception:
            slope = np.nan
        return slope

    # 7. Self-Intersection Count (Path Crossing Frequency)
    def calculate_self_intersections(self, data):
        """
        Count the number of self-intersections in a trajectory.
        A simple (but potentially inefficient) algorithm that checks if line segments
        intersect. This is usually feasible for small windows.
        """
        # Extract positions.
        points = data[["x_um", "y_um"]].values
        n_points = len(points)
        if n_points < 4:
            return 0
        intersections = 0
        # Check every pair of segments (i, i+1) and (j, j+1) with j > i+1.
        for i in range(n_points - 1):
            p1 = points[i]
            p2 = points[i + 1]
            for j in range(i + 2, n_points - 1):
                p3 = points[j]
                p4 = points[j + 1]
                if self.segments_intersect(p1, p2, p3, p4):
                    intersections += 1
        return intersections

    def segments_intersect(self, p1, p2, p3, p4):
        """Determine if line segments (p1, p2) and (p3, p4) intersect."""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (
            ccw(p1, p2, p3) != ccw(p1, p2, p4)
        )

    # 8. Pausing/Intermittency Metrics
    def calculate_pausing_metric(self, data, speed_threshold=0.1):
        """
        Compute the fraction of frames in which the instantaneous speed is below the threshold.
        """
        if "speed_um_s" not in data.columns:
            data = self.calculate_speeds()
        speeds = data["speed_um_s"].dropna().values
        if len(speeds) == 0:
            return np.nan
        pauses = speeds < speed_threshold
        return np.sum(pauses) / len(speeds)

    ############# V4 below ###############################

    def msd_normal(self, t, D):
        """
        Normal diffusion MSD model in 2D.
        For 2-dimensional diffusion:
            MSD(t) = 4 * D * t
        """
        return 4 * D * t

    def calculate_msd_for_track(
        self,
        track_data,
        store_msd=False,
        time_window=None,
        use_bounds=False,
        allow_partial_window=True,
        min_window_size=60,
        **kwargs,
    ):
        """
        Calculate the MSD for a given track (or window) and fit the MSD model.

        Parameters
        - track_data: DataFrame for the track or window.
        - store_msd: If True, store the per-lag MSD records.
        - time_window: Identifier for the time window (if applicable).
        - use_bounds: (Legacy; unused) Whether to enforce parameter bounds.
        - allow_partial_window: If False, skip windows with fewer than min_window_size frames.
        - min_window_size: Minimum number of frames required in a window.

        Additional keyword arguments:
        - fit_method: "r2_threshold" (default) or "model_selection".
                        In "r2_threshold" mode the goodness-of-fit is assessed via an R² threshold.
                        In "model_selection" mode both a normal model (α fixed = 1) and an anomalous model
                        (α free) are fitted and compared via AIC.
        - r2_threshold: R² threshold for a good fit for the normal model (default: 0.95).
        - anomalous_r2_threshold: R² threshold for the anomalous model (default: 0.80).
        - bad_fit_strategy: How to handle a poor fit:
                "remove_track" (remove the entire track),
                "excise_window" (skip the offending window and, if it's the first such instance in the track,
                                    later windows will get a new unique ID),
                "flag" (keep the window but mark it as flagged).
                Default is "flag".
        - use_ci: (Boolean) If True, use the confidence interval for α (based on parameter uncertainty)
                    rather than a fixed tolerance for diffusion type classification. Default is False.
        - ci_multiplier: Multiplier for the standard error for constructing a confidence interval (default: 1.96).

        Returns
        A tuple: (avg_msd, D, alpha, motion_class)
                where motion_class is a string (e.g. "normal", "subdiffusive", "superdiffusive", or flagged/excised notes).

        """
        import numpy as np
        import scipy.optimize

        n_frames = len(track_data)

        # Check that there are enough frames.
        if time_window is None:
            if n_frames < 3:
                print(
                    f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient frames ({n_frames})."
                )
                return np.nan, np.nan, np.nan, "unlabeled"
        elif not allow_partial_window and n_frames < min_window_size:
            print(
                f"Skipping unique_id {track_data['unique_id'].iloc[0]} for time window {time_window}: insufficient frames ({n_frames} < {min_window_size})."
            )
            return np.nan, np.nan, np.nan, "unlabeled"

        # Compute lag times and observed MSD values.
        lag_times = np.arange(1, n_frames) * self.time_between_frames
        msd_values = np.zeros(len(lag_times))
        for lag in range(1, len(lag_times) + 1):
            displacements = (
                track_data[["x_um", "y_um"]].iloc[lag:].values
                - track_data[["x_um", "y_um"]].iloc[:-lag].values
            ) ** 2
            msd_values[lag - 1] = np.mean(np.sum(displacements, axis=1))

        if len(lag_times) < 3:
            print(
                f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient lag times ({len(lag_times)})."
            )
            return np.nan, np.nan, np.nan, "unlabeled"

        # Get options from kwargs.
        fit_method = kwargs.get("fit_method", "r2_threshold")
        r2_threshold = kwargs.get("r2_threshold", 0.95)
        anomalous_r2_threshold = kwargs.get("anomalous_r2_threshold", 0.80)
        bad_fit_strategy = kwargs.get("bad_fit_strategy", "flag")
        use_ci_flag = kwargs.get("use_ci", False)
        ci_multiplier = kwargs.get(
            "ci_multiplier", 1.96
        )  # Approximately 95% CI if errors are normal.

        # Initialize variables that will be returned
        motion_class = "unlabeled"  # Default value
        avg_msd = np.nan
        D = np.nan
        alpha = np.nan

        log_obs = np.log10(msd_values)
        n = len(lag_times)

        # ----- MODEL_SELECTION BRANCH -----
        if fit_method == "model_selection":
            # Fit the anomalous model (free α) first.
            try:
                popt_anom, pcov_anom = scipy.optimize.curve_fit(
                    self.msd_model, lag_times, msd_values
                )
                D_anom, alpha_anom = popt_anom[0], popt_anom[1]
                predicted_anom = self.msd_model(lag_times, D_anom, alpha_anom)
                log_pred_anom = np.log10(predicted_anom)
                rss_anom = np.sum((log_obs - log_pred_anom) ** 2)
            except RuntimeError:
                print(
                    f"Curve fitting (anomalous) failed for unique_id {track_data['unique_id'].iloc[0]}."
                )
                return np.nan, np.nan, np.nan, "bad_fit_removed_track"

            # Fit the normal model (α fixed at 1) using msd_normal.
            try:
                popt_norm, _ = scipy.optimize.curve_fit(
                    self.msd_normal, lag_times, msd_values
                )
                D_norm = popt_norm[0]
                predicted_norm = self.msd_normal(lag_times, D_norm)
                log_pred_norm = np.log10(predicted_norm)
                rss_norm = np.sum((log_obs - log_pred_norm) ** 2)
            except RuntimeError:
                print(
                    f"Curve fitting (normal) failed for unique_id {track_data['unique_id'].iloc[0]}."
                )
                # Use the anomalous model if normal fit fails.
                D_norm, rss_norm = np.nan, np.inf

            # Compute AIC values.
            # For the anomalous model, k = 2; for normal, k = 1.
            aic_anom = 2 * 2 + n * np.log(rss_anom / n)
            aic_norm = 2 * 1 + n * np.log(rss_norm / n)

            # Calculate Akaike weights.
            aic_values = {"anomalous": aic_anom, "normal": aic_norm}
            min_aic = min(aic_values.values())
            delta_aics = {model: (aic_values[model] - min_aic) for model in aic_values}
            weights = {model: np.exp(-0.5 * delta_aics[model]) for model in delta_aics}
            sum_weights = sum(weights.values())
            akaike_weights = {model: weights[model] / sum_weights for model in weights}
            # For debugging:
            print(f"Akaike weights: {akaike_weights}")

            # Choose the model with the lower AIC.
            if aic_norm < aic_anom:
                chosen_model = "normal"
                D = D_norm
                alpha = 1.0
                chosen_rss = rss_norm
                chosen_covar = None  # α is fixed.
            else:
                chosen_model = "anomalous"
                D = D_anom
                alpha = alpha_anom
                chosen_rss = rss_anom
                chosen_covar = pcov_anom

            # Compute R² in log–log space.
            ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
            chosen_r2 = 1 - chosen_rss / ss_tot

            # Decide which R² threshold applies.
            threshold = (
                r2_threshold if chosen_model == "normal" else anomalous_r2_threshold
            )

            # Check if the chosen model's fit is acceptable.
            if chosen_r2 < threshold:
                if bad_fit_strategy == "remove_track":
                    print(
                        f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} (R² = {chosen_r2:.3f}). Removing track."
                    )
                    return np.nan, np.nan, np.nan, "bad_fit_removed_track"
                elif bad_fit_strategy == "excise_window":
                    print(
                        f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Excising window."
                    )
                    return np.nan, np.nan, np.nan, "excised_bad_fit"
                elif bad_fit_strategy == "flag":
                    print(
                        f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Flagging window."
                    )
                    motion_class = (
                        "bad_fit_flagged_normal"
                        if chosen_model == "normal"
                        else (
                            "bad_fit_flagged_subdiffusive"
                            if alpha < 1 - (self.tolerance if not use_ci_flag else 0)
                            else "bad_fit_flagged_superdiffusive"
                        )
                    )
                    return np.mean(msd_values), D, alpha, motion_class

            # Now, determine diffusion type using either fixed tolerance or CI.
            if chosen_model == "normal":
                motion_class = "normal"
            elif use_ci_flag and (chosen_covar is not None):
                # Calculate standard error of α.
                try:
                    alpha_err = np.sqrt(np.diag(chosen_covar))[1]
                except Exception as e:
                    print(
                        f"Could not extract uncertainty: {e}; reverting to fixed tolerance."
                    )
                    use_ci_flag = False
                if use_ci_flag:
                    alpha_lower = alpha - ci_multiplier * alpha_err
                    alpha_upper = alpha + ci_multiplier * alpha_err
                    print(
                        f"Confidence interval for α: [{alpha_lower:.3f}, {alpha_upper:.3f}]"
                    )
                    if alpha_lower <= 1 <= alpha_upper:
                        motion_class = "normal"
                    elif alpha_upper < 1:
                        motion_class = "subdiffusive"
                    elif alpha_lower > 1:
                        motion_class = "superdiffusive"
                    else:
                        motion_class = "ambiguous"
                elif alpha < 1 - self.tolerance:
                    motion_class = "subdiffusive"
                elif alpha > 1 + self.tolerance:
                    motion_class = "superdiffusive"
                else:
                    motion_class = "normal"

            avg_msd = np.mean(msd_values)

        # -------- FALLBACK: r2_threshold METHOD --------
        else:
            try:
                popt, pcov = scipy.optimize.curve_fit(
                    self.msd_model, lag_times, msd_values
                )
                D, alpha = popt[0], popt[1]
                predicted = self.msd_model(lag_times, D, alpha)
            except RuntimeError:
                print(
                    f"Curve fitting failed for unique_id {track_data['unique_id'].iloc[0]}."
                )
                return np.nan, np.nan, np.nan, "bad_fit_removed_track"

            log_pred = np.log10(predicted)
            ss_res = np.sum((log_obs - log_pred) ** 2)
            ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
            chosen_r2 = 1 - ss_res / ss_tot

            threshold = (
                anomalous_r2_threshold
                if (alpha < 1 - self.tolerance or alpha > 1 + self.tolerance)
                else r2_threshold
            )

            if chosen_r2 < threshold:
                if bad_fit_strategy == "remove_track":
                    # print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} (R² = {chosen_r2:.3f}). Removing track.")
                    return np.nan, np.nan, np.nan, "bad_fit_removed_track"
                elif bad_fit_strategy == "excise_window":
                    # print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Excising window.")
                    return np.nan, np.nan, np.nan, "excised_bad_fit"
                elif bad_fit_strategy == "flag":
                    # print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Flagging window.")
                    if alpha < 1 - self.tolerance:
                        motion_class = "bad_fit_flagged_subdiffusive"
                    elif alpha > 1 + self.tolerance:
                        motion_class = "bad_fit_flagged_superdiffusive"
                    else:
                        motion_class = "bad_fit_flagged_normal"
                    return np.mean(msd_values), D, alpha, motion_class
            elif alpha < 1 - self.tolerance:
                motion_class = "subdiffusive"
            elif alpha > 1 + self.tolerance:
                motion_class = "superdiffusive"
            else:
                motion_class = "normal"

            avg_msd = np.mean(msd_values)

            # Optionally store per-lag MSD records.
            if store_msd and (time_window is not None):
                import pandas as pd

                msd_records = [
                    {
                        "unique_id": track_data["unique_id"].iloc[0],
                        "time_window": time_window,
                        "lag_time": lag,
                        "msd": msd,
                        "diffusion_coefficient": D,
                        "anomalous_exponent": alpha,
                    }
                    for lag, msd in zip(lag_times, msd_values, strict=False)
                ]
                if msd_records:
                    msd_records_df = pd.DataFrame(msd_records)
                    self.msd_lagtime_df = pd.concat(
                        [self.msd_lagtime_df, msd_records_df], ignore_index=True
                    )

            return avg_msd, D, alpha, motion_class # test #

    def calculate_time_windowed_metrics(
        self, window_size=None, overlap=None, filter_metrics_df=False, **kwargs
    ):
        """
        Calculate metrics for time windows across all tracks.

        Any additional keyword arguments are passed to calculate_msd_for_track,
        including those controlling fit_method (e.g. "r2_threshold" vs "model_selection")
        and the bad-fit handling options.

        For bad fits, three strategies are available via 'bad_fit_strategy':
        - "remove_track": Remove all windows from the track.
        - "excise_window": A new set of unique ids called e_uid are made. Bad fits
        are subsequent windows are given a new unique ID with the suffix '_s'.
        - "flag": DEFAULT. Keep the window but flag it (e.g. 'bad_fit_flagged_normal').

        Parameters
        - window_size: Number of frames per window.
        - overlap: Number of overlapping frames.
        - filter_metrics_df: If True, update self.metrics_df to only include frames in windows.
        - **kwargs: Additional options to pass to calculate_msd_for_track.

        """
        if window_size is None:
            window_size = self.calculate_default_window_size()

        if overlap is None:
            overlap = int(window_size / 2)

        # ─── determine step and strategy ─── # NEW
        step = window_size - overlap
        use_majority = overlap > window_size / 2

        # ─── NEW: Set default bad_fit_strategy to "flag" ───
        if "bad_fit_strategy" not in kwargs:
            kwargs["bad_fit_strategy"] = "flag"
            print("ℹ️  Using default bad_fit_strategy='flag' (no track splitting)")

        bad_fit_strategy = kwargs.get("bad_fit_strategy", "flag")
        print(f"ℹ️  Bad fit strategy: {bad_fit_strategy}")

        if use_majority:
            # we'll collect *all* e_uid candidates per frame
            self.metrics_df["e_uid_candidates"] = [
                [] for _ in range(len(self.metrics_df))
            ]
            # NEW: Also collect window_uid candidates for majority vote
            self.metrics_df["window_uid_candidates"] = [
                [] for _ in range(len(self.metrics_df))
            ]
        else:
            # simple overwrite
            self.metrics_df["e_uid"] = None
            # NEW: Initialize window_uid column
            self.metrics_df["window_uid"] = None

        print(
            "Note: Partial windows (below the window size) are not included in the time-windowed metrics."
        )
        windowed_list = []
        included_frames = set()

        for unique_id, track_data in tqdm(
            self.metrics_df.groupby("unique_id"),
            desc="Calculating Time-Windowed Metrics",
        ):
            n_frames = len(track_data)
            track_removed = False

            # ─── Initialize split counter (only used if bad_fit_strategy="excise_window") ───
            split_count = 0

            for start in range(0, n_frames - window_size + 1, window_size - overlap):
                end = start + window_size
                window_data = track_data.iloc[start:end].copy()
                included_frames.update(window_data["frame"].values)

                # Compute displacements.
                start_row = window_data.iloc[0]
                end_row = window_data.iloc[-1]
                net_disp = np.sqrt(
                    (end_row["x_um"] - start_row["x_um"]) ** 2
                    + (end_row["y_um"] - start_row["y_um"]) ** 2
                )

                window_data["x_um_prev"] = window_data["x_um"].shift(1)
                window_data["y_um_prev"] = window_data["y_um"].shift(1)
                window_data["segment_len_um"] = np.sqrt(
                    (window_data["x_um"] - window_data["x_um_prev"]) ** 2
                    + (window_data["y_um"] - window_data["y_um_prev"]) ** 2
                ).fillna(0)
                cum_disp = window_data["segment_len_um"].sum()

                # Call the MSD fitting function.
                avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
                    window_data,
                    store_msd=True,
                    time_window=start // (window_size - overlap),
                    **kwargs,
                )

                if motion_class == "bad_fit_removed_track":
                    print(
                        f"Removing entire track for unique_id {unique_id} due to a bad fit."
                    )
                    track_removed = True
                    break

                # ─── NEW: Simplified excision logic ───
                if (
                    bad_fit_strategy == "excise_window"
                    and motion_class == "excised_bad_fit"
                ):
                    # Only create new e_uid if explicitly using excision strategy
                    e_uid = f"{unique_id}_excised"
                    split_count += 1
                    # For subsequent windows after excision, use split numbering
                elif bad_fit_strategy == "excise_window" and split_count > 0:
                    # After excision has occurred, use split numbering
                    e_uid = f"{unique_id}_s{split_count}"
                else:
                    # Default behavior: keep original unique_id
                    # This covers both 'flag' strategy and normal windows in 'excise_window' strategy
                    e_uid = unique_id

                # NEW: Create window_uid with format: unique_id_timewindow_framestart_frameend
                time_window_num = start // (window_size - overlap)
                frame_start = start_row["frame"]
                frame_end = end_row["frame"]
                window_uid = f"{unique_id}_{time_window_num}_{frame_start}_{frame_end}"

                # # ─── assign per-frame, either by append or overwrite ─── #im removing this for now - I just want to use most recent window
                if use_majority:
                    # append to our per-frame candidate lists
                    self.metrics_df.loc[window_data.index, "e_uid_candidates"] = (
                        self.metrics_df.loc[
                            window_data.index, "e_uid_candidates"
                        ].apply(lambda lst: lst + [e_uid])
                    )
                    # NEW: append window_uid candidates
                    self.metrics_df.loc[window_data.index, "window_uid_candidates"] = (
                        self.metrics_df.loc[
                            window_data.index, "window_uid_candidates"
                        ].apply(lambda lst: lst + [window_uid])
                    )
                else:
                    # later window simply overwrites
                    self.metrics_df.loc[window_data.index, "e_uid"] = e_uid
                    # NEW: assign window_uid to frames
                    self.metrics_df.loc[window_data.index, "window_uid"] = window_uid

                total_time_s = (
                    window_data["time_s"].iloc[-1] - window_data["time_s"].iloc[0]
                )
                avg_speed = window_data["speed_um_s"].mean()
                avg_acceleration = window_data["acceleration_um_s2"].mean()
                avg_jerk = window_data["jerk_um_s3"].mean()
                avg_norm_curvature = window_data["normalized_curvature"].mean()
                avg_angle_norm_curvature = window_data[
                    "angle_normalized_curvature"
                ].mean()
                persistence_length = self.calculate_persistence_length(window_data)

                # --- New: Compute additional (window-specific) features --- April 2025
                straightness = self.calculate_straightness_index(window_data)
                rg = self.calculate_radius_of_gyration(window_data)
                hull_area = self.calculate_convex_hull_area(window_data)
                dir_entropy = self.calculate_directional_entropy(window_data)
                speed_cv = self.calculate_speed_variability(window_data)
                direction_autocorr = self.calculate_directional_autocorrelation(
                    window_data
                )
                eccentricity = self.calculate_trajectory_eccentricity(window_data)
                tvar, tskew, tkurt = self.calculate_turning_angle_moments(window_data)
                step_stats = self.calculate_steplength_distribution_stats(window_data)
                if step_stats is not None:
                    (
                        steplength_mean,
                        steplength_std,
                        steplength_skew,
                        steplength_kurt,
                    ) = step_stats
                else:
                    (
                        steplength_mean,
                        steplength_std,
                        steplength_skew,
                        steplength_kurt,
                    ) = (np.nan, np.nan, np.nan, np.nan)
                diffusivity_cv = self.calculate_instantaneous_diffusivity_variability(
                    window_data
                )
                fractal_dim = self.calculate_fractal_dimension(window_data)
                psd_slope_speed = self.calculate_psd_slope(window_data, signal="speed")
                intersections = self.calculate_self_intersections(window_data)
                pausing_fraction = self.calculate_pausing_metric(
                    window_data, speed_threshold=0.1
                )

                # --- NEW: Compute von Mises kappa values for this window ---
                kappa_turning, kappa_absolute = self.calculate_vonmises_kappa(
                    window_data
                )

                window_summary = pd.DataFrame(
                    {
                        "time_window": [time_window_num],
                        "x_um_start": [start_row["x_um"]],
                        "y_um_start": [start_row["y_um"]],
                        "x_um_end": [end_row["x_um"]],
                        "y_um_end": [end_row["y_um"]],
                        "particle": [start_row["particle"]],
                        "condition": [start_row["condition"]],
                        "filename": [start_row["filename"]],
                        "file_id": [start_row["file_id"]],
                        "unique_id": [
                            unique_id
                        ],  # ← Always uses original unique_id now
                        "frame_start": [
                            frame_start
                        ],  # I added these to vectorize later mapping functions
                        "frame_end": [frame_end],
                        "e_uid": [e_uid],
                        "window_uid": [
                            window_uid
                        ],  # NEW: Add window_uid to windowed dataframe
                        "split_count": [
                            split_count
                        ],  # NEW: track how many splits this window has
                        "avg_msd": [avg_msd],
                        "n_frames": [window_size],
                        "total_time_s": [total_time_s],
                        "location": [start_row["location"]],
                        "molecule": [start_row["molecule"]],
                        "genotype": [start_row["genotype"]],
                        "cell_type": [start_row["cell_type"]],
                        "diffusion_coefficient": [D],
                        "anomalous_exponent": [alpha],
                        "motion_class": [motion_class],
                        "avg_speed_um_s": [avg_speed],
                        "avg_acceleration_um_s2": [avg_acceleration],
                        "avg_jerk_um_s3": [avg_jerk],
                        "avg_normalized_curvature": [avg_norm_curvature],
                        "avg_angle_normalized_curvature": [avg_angle_norm_curvature],
                        "persistence_length": [persistence_length],
                        "net_displacement_um": [net_disp],
                        "cum_displacement_um": [cum_disp],
                        # New metrics:
                        "straightness_index": [straightness],
                        "radius_of_gyration": [rg],
                        "convex_hull_area": [hull_area],
                        "directional_entropy": [dir_entropy],
                        "speed_variability": [speed_cv],
                        "direction_autocorrelation": [direction_autocorr],
                        "kappa_turning": [kappa_turning],
                        "kappa_absolute": [kappa_absolute],
                        "eccentricity": [eccentricity],
                        "turning_angle_variance": [tvar],
                        "turning_angle_skew": [tskew],
                        "turning_angle_kurtosis": [tkurt],
                        "steplength_mean": [steplength_mean],
                        "steplength_std": [steplength_std],
                        "steplength_skew": [steplength_skew],
                        "steplength_kurtosis": [steplength_kurt],
                        "diffusivity_cv": [diffusivity_cv],
                        "fractal_dimension": [fractal_dim],
                        "psd_slope_speed": [psd_slope_speed],
                        "self_intersections": [intersections],
                        "pausing_fraction": [pausing_fraction],
                        "bad_fit_flag": [motion_class.startswith("bad_fit_flagged")],
                    }
                )

                windowed_list.append(window_summary)

            if track_removed:
                print(
                    f"Skipping unique_id {unique_id} entirely due to a bad fit in one or more windows."
                )
                continue

        # ─── after all windows, collapse majority if needed ───
        if use_majority:
            from collections import Counter

            def pick_majority(cands):
                if not cands:
                    return None
                return Counter(cands).most_common(1)[0][0]

            self.metrics_df["e_uid"] = self.metrics_df["e_uid_candidates"].apply(
                pick_majority
            )
            # NEW: Apply majority vote for window_uid as well
            self.metrics_df["window_uid"] = self.metrics_df[
                "window_uid_candidates"
            ].apply(pick_majority)
            
            self.metrics_df.drop(
                columns=["e_uid_candidates", "window_uid_candidates"], inplace=True
            )

        self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)

        if filter_metrics_df:
            print("Filtering metrics_df to only include frames within time windows...")
            print(f"Initial number of frames: {len(self.metrics_df)}")
            self.metrics_df = self.metrics_df[
                self.metrics_df["frame"].isin(included_frames)
            ].copy()
            print(f"Remaining frames after filtering: {len(self.metrics_df)}")

    def produce_time_averaged_df(self, max_lagtime=None):
        """
        Produce the time-averaged DataFrame.

        Parameters
        - max_lagtime: maximum number of frames to consider for lag times

        """
        if max_lagtime is None:
            max_lagtime = self.calculate_default_max_lagtime()

        time_averaged_list = []

        for unique_id, track_data in tqdm(
            self.metrics_df.groupby("unique_id"),
            desc="Producing Time-Averaged DataFrame",
        ):
            avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
                track_data, max_lagtime
            )

            # Calculate total time in seconds for the track
            total_time_s = track_data["time_s"].iloc[-1] - track_data["time_s"].iloc[0]

            # Calculate average instantaneous metrics for the track
            avg_speed = track_data["speed_um_s"].mean()
            avg_acceleration = track_data["acceleration_um_s2"].mean()
            avg_jerk = track_data["jerk_um_s3"].mean()
            avg_norm_curvature = track_data["normalized_curvature"].mean()
            avg_angle_norm_curvature = track_data["angle_normalized_curvature"].mean()
            # NEW: Compute von Mises kappa values for the entire track.
            kappa_turning, kappa_absolute = self.calculate_vonmises_kappa(track_data)

            # NEW: Retrieve net displacement and cumulative displacement.
            # Assumes these columns were computed earlier via calculate_net_displacement() and calculate_cum_displacement()
            net_disp = track_data["net_displacement_um"].iloc[-1]
            cum_disp = track_data["cum_displacement_um"].iloc[-1]

            # NEW: Compute additional descriptors
            straightness = self.calculate_straightness_index(track_data)
            rg = self.calculate_radius_of_gyration(track_data)
            hull_area = self.calculate_convex_hull_area(track_data)
            dir_entropy = self.calculate_directional_entropy(track_data)
            speed_cv = self.calculate_speed_variability(track_data)
            direction_autocorr = self.calculate_directional_autocorrelation(track_data)
            eccentricity = self.calculate_trajectory_eccentricity(track_data)
            tvar, tskew, tkurt = self.calculate_turning_angle_moments(track_data)
            steplength_mean, steplength_std, steplength_skew, steplength_kurt = (
                self.calculate_steplength_distribution_stats(track_data)
            )
            diffusivity_cv = self.calculate_instantaneous_diffusivity_variability(
                track_data
            )
            fractal_dim = self.calculate_fractal_dimension(track_data)
            psd_slope_speed = self.calculate_psd_slope(track_data, signal="speed")
            intersections = self.calculate_self_intersections(track_data)
            pausing_fraction = self.calculate_pausing_metric(
                track_data, speed_threshold=0.1
            )
            # kappa_turning, kappa_absolute = self.calculate_vonmises_kappa(track_data)

            # Add track-level summary information to time_averaged_df
            start_row = track_data.iloc[0]
            end_row = track_data.iloc[-1]
            track_summary = pd.DataFrame(
                {
                    "x_um_start": [start_row["x_um"]],
                    "y_um_start": [start_row["y_um"]],
                    "x_um_end": [end_row["x_um"]],
                    "y_um_end": [end_row["y_um"]],
                    "particle": [start_row["particle"]],
                    "condition": [start_row["condition"]],
                    "filename": [start_row["filename"]],
                    "file_id": [start_row["file_id"]],
                    "unique_id": [unique_id],
                    "avg_msd": [avg_msd],  # Add the average MSD (units: μm²)
                    "n_frames": [len(track_data)],  # Add the number of frames
                    "total_time_s": [total_time_s],  # Add the total time in seconds
                    "location": [start_row["location"]],  # Add the Location
                    "molecule": [start_row["molecule"]],
                    "genotype": [start_row["genotype"]],
                    "cell_type": [start_row["cell_type"]],  # Add the Cell Type
                    "diffusion_coefficient": [D],  # Add the diffusion coefficient
                    "anomalous_exponent": [alpha],  # Add the anomalous exponent
                    "motion_class": [motion_class],  # Add the motion class
                    "avg_speed_um_s": [avg_speed],  # Add average speed
                    "avg_acceleration_um_s2": [
                        avg_acceleration
                    ],  # Add average acceleration
                    "avg_jerk_um_s3": [avg_jerk],  # Add average jerk
                    "avg_normalized_curvature": [
                        avg_norm_curvature
                    ],  # Add average normalized curvature
                    "avg_angle_normalized_curvature": [
                        avg_angle_norm_curvature
                    ],  # Add average angle normalized curvature
                    "kappa_turning": [
                        kappa_turning
                    ],  # NEW: von Mises concentration for turning angles
                    "kappa_absolute": [
                        kappa_absolute
                    ],  # NEW: von Mises concentration for absolute directions
                    "net_displacement_um": [
                        net_disp
                    ],  # NEW: Net displacement over the track (straight-line distance from start to end)
                    "cum_displacement_um": [
                        cum_disp
                    ],  # NEW: Cumulative displacement over the track (total path length)
                    ### below new april 2025
                    "straightness_index": [straightness],
                    "radius_of_gyration": [rg],
                    "convex_hull_area": [hull_area],
                    "directional_entropy": [dir_entropy],
                    "speed_variability": [speed_cv],
                    "direction_autocorrelation": [direction_autocorr],
                    "eccentricity": [eccentricity],
                    "turning_angle_variance": [tvar],
                    "turning_angle_skew": [tskew],
                    "turning_angle_kurtosis": [tkurt],
                    "steplength_mean": [steplength_mean],
                    "steplength_std": [steplength_std],
                    "steplength_skew": [steplength_skew],
                    "steplength_kurtosis": [steplength_kurt],
                    "diffusivity_cv": [diffusivity_cv],
                    "fractal_dimension": [fractal_dim],
                    "psd_slope_speed": [psd_slope_speed],
                    "self_intersections": [intersections],
                    "pausing_fraction": [pausing_fraction],
                }
            )

            time_averaged_list.append(track_summary)

        self.time_averaged_df = pd.concat(time_averaged_list).reset_index(drop=True)

    def calculate_metrics_for_window(self, window_data):
        """
        Calculate metrics for a given window of data.
        """
        n_frames = len(window_data)
        max_lagtime = min(
            100, int(n_frames / 2)
        )  # Example: use half the length of the window or 100, whichever is smaller

        msd_values = np.zeros(max_lagtime)

        for lag in range(1, max_lagtime + 1):
            if lag < n_frames:
                displacements = (
                    window_data[["x_um", "y_um"]].iloc[lag:].values
                    - window_data[["x_um", "y_um"]].iloc[:-lag].values
                ) ** 2
                squared_displacements = np.sum(displacements, axis=1)
                msd_values[lag - 1] = np.mean(squared_displacements)
            else:
                break

        avg_msd = np.mean(
            msd_values
        )  # Calculate the average MSD for the window (units: μm²)

        # Fit MSD data to determine the type of motion and extract parameters
        total_time_s = window_data["time_s"].iloc[-1] - window_data["time_s"].iloc[0]
        lag_times = np.arange(1, max_lagtime + 1) * (total_time_s / (n_frames - 1))
        popt, _ = scipy.optimize.curve_fit(
            self.msd_model, lag_times, msd_values[:max_lagtime]
        )
        D, alpha = popt[0], popt[1]

        # Classify the type of motion
        if alpha < 1:
            motion_class = "subdiffusive"
        elif alpha > 1:
            motion_class = "superdiffusive"
        else:
            motion_class = "normal"

        return avg_msd, D, alpha, motion_class

    def calculate_default_window_size(self):
        """
        Calculate the default window size based on the average length of tracks in the dataset.
        """
        avg_track_length = self.metrics_df.groupby("unique_id").size().mean()
        default_window_size = int(
            avg_track_length / 2
        )  # Example: use half the average length of the tracks
        return default_window_size

    def calculate_all_features(
        self,
        max_lagtime=None,
        calculate_time_windowed=False,
        calculate_time_averaged=True,
    ):
        """
        Calculate all features for the particle tracking data.
        This method will call all individual feature calculation methods.

        Parameters
        ----------
        max_lagtime : int, optional
            Maximum number of frames to consider for lag times
        calculate_time_windowed : bool, default False
            Whether to calculate time-windowed metrics
        calculate_time_averaged : bool, default True
            Whether to calculate time-averaged metrics (MSD analysis per track)
            Set to False for faster processing when only instantaneous features are needed

        """
        print("🔧 Feature calculation options:")
        print("   • Instantaneous features: ✓ Always calculated")
        print(
            f"   • Time-windowed features: {'✓ Enabled' if calculate_time_windowed else '✗ Disabled'}"
        )
        print(
            f"   • Time-averaged features: {'✓ Enabled' if calculate_time_averaged else '✗ Disabled'}"
        )

        # Calculate default max lag time if not provided (only needed for time-averaged)
        if calculate_time_averaged and max_lagtime is None:
            max_lagtime = self.calculate_default_max_lagtime()

        # Calculate distances between consecutive frames
        self.calculate_distances()

        # Calculate speeds between consecutive frames
        self.calculate_speeds()

        # Calculate directions of motion between consecutive frames
        self.calculate_directions()

        # Calculate accelerations between consecutive frames
        self.calculate_accelerations()

        # Calculate jerks between consecutive frames
        self.calculate_jerk()

        # Calculate curvatures between consecutive frames
        self.calculate_normalized_curvature()
        self.calculate_angle_normalized_curvature()

        # Calculate instantaneous diffusion coefficient between consecutive frames
        self.calculate_instantaneous_diffusion_coefficient()

        # Calculate net displacement
        self.calculate_net_displacement()

        self.calculate_cum_displacement()  # ADDED 3_19_2025

        # Calculate MSD for each track and aggregate (OPTIONAL)
        if calculate_time_averaged:
            print("📊 Calculating time-averaged metrics (MSD analysis)...")
            self.calculate_msd_for_all_tracks(max_lagtime)
        else:
            print("⏭️  Skipping time-averaged metrics calculation")
            # Initialize empty time_averaged_df for consistency
            self.time_averaged_df = pd.DataFrame()

        # Calculate instantaneous velocity
        self.calculate_instantaneous_velocity()

        # Calculate time-windowed metrics if requested (OPTIONAL)
        if calculate_time_windowed:
            print("📊 Calculating time-windowed metrics...")
            self.calculate_time_windowed_metrics()
        else:
            print("⏭️  Skipping time-windowed metrics calculation")
            # Initialize empty time_windowed_df for consistency
            self.time_windowed_df = pd.DataFrame()

        # Cleanup step to remove temporary columns
        self.cleanup()

        print("✅ Feature calculation completed:")
        print(f"   • Instantaneous features: {len(self.metrics_df)} trajectory points")
        print(f"   • Time-averaged features: {len(self.time_averaged_df)} tracks")
        print(f"   • Time-windowed features: {len(self.time_windowed_df)} windows")

        return self.metrics_df

    def get_time_averaged_df(self):
        """
        Return the DataFrame with time-averaged metrics.
        """
        return self.time_averaged_df

    def get_time_windowed_df(self):
        """
        Return the DataFrame with time-windowed metrics.
        """
        return self.time_windowed_df

    def calculate_default_max_lagtime(self):
        """
        Calculate the default maximum lag time based on the shortest track in the dataset.
        """
        min_track_length = self.metrics_df.groupby("unique_id").size().min()
        default_max_lagtime = min(
            100, int(min_track_length / 2)
        )  # Example: use half the length of the shortest track or 100, whichever is smaller
        return default_max_lagtime

    def calculate_msd_for_all_tracks(self, max_lagtime):
        """
        Calculate the MSD for all tracks and produce the time-averaged DataFrame.

        Parameters
        - max_lagtime: maximum number of frames to consider for lag times

        """
        self.produce_time_averaged_df(max_lagtime)

    def cleanup(self):
        """
        Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
        """
        self.metrics_df.drop(
            columns=[
                "x_um_prev",
                "y_um_prev",
                "time_s_prev",
                "delta_time_s",
                "speed_um_s_prev",
                "acceleration_um_s2_prev",
                "direction_rad_prev",
                "instant_velocity_x_um_s",
                "instant_velocity_y_um_s",
            ],
            inplace=True,
        )


################################### DEV VERSION BELOW #############################################
