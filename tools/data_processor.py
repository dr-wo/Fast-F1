from typing import Union
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd

import fastf1
import fastf1.plotting
from fastf1.core import Session as FastF1Session, Laps as FastF1Laps

F1_TEAM_COLORS = {
    'Ferrari': '#dc0000',
    'McLaren': '#ff8700',
    'Red Bull Racing': "#3029ed",
    'Red Bull': '#3029ed',
    'Mercedes': "#00fbe2",
    'Aston Martin': '#006f62',
    'Alpine': '#ff87bc',
    'Williams': "#1f6af4ff",
    'Haas F1 Team': '#6e6e6e',
    'Haas': '#6e6e6e',
    'Cadillac': "#82d8f4",
    'Kick Sauber': '#00e701',
    'Sauber': '#00e701',
    'Audi': '#00e701',
    'Racing Bulls': "#5e59ee",
    'RB': "#5e59ee",
    'Audi': "#000000",
}

DRIVER_LINESTYLES_BY_RANK = {
    0: '-',
    1: '--',
    2: ':',
}

DRIVER_BAR_HATCHES_BY_LINESTYLE = {
    '-': None,
    '--': 'xx',
    ':': '//',
}


def _build_team_style_maps(
    laps: pd.DataFrame,
) -> tuple[dict[str, object], dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Build shared team color, line style, and hatch maps."""
    team_names = sorted(laps['Team'].dropna().unique())
    fallback_cmap = plt.get_cmap('tab20')
    fallback_team_colors = {
        team: fallback_cmap(idx % fallback_cmap.N)
        for idx, team in enumerate(team_names)
    }
    team_color_map = {
        team: F1_TEAM_COLORS.get(team, fallback_team_colors[team])
        for team in team_names
    }

    driver_team_summary = (
        laps.groupby(['Team', 'Driver'])
        .size()
        .reset_index(name='total_laps')
        .sort_values(['Team', 'total_laps', 'Driver'], ascending=[True, False, True])
    )
    team_driver_linestyle_map = {}
    for team, team_group in driver_team_summary.groupby('Team', sort=False):
        drivers = team_group['Driver'].tolist()
        for idx, driver in enumerate(drivers):
            team_driver_linestyle_map[(team, driver)] = DRIVER_LINESTYLES_BY_RANK.get(idx, ':')

    team_driver_hatch_map = {
        key: DRIVER_BAR_HATCHES_BY_LINESTYLE.get(linestyle)
        for key, linestyle in team_driver_linestyle_map.items()
    }

    return team_color_map, team_driver_linestyle_map, team_driver_hatch_map


class Session:
    """Wrapper for FastF1 Session with chainable filter methods.
    
    This class provides a convenient interface to FastF1 session data with
    chainable filter methods for common operations.
    
    Examples:
        # Chain multiple filters
        session = Session(2023, 1, 'R')
        filtered_laps = session.drivers(['HAM', 'VER']).quicklaps(1.05).compounds(['SOFT']).get()
        
        # Or access filtered laps directly
        session.drivers(['HAM']).quicklaps()
        ham_quick_laps = session.laps
        
        # Reset filters
        session.reset()  # Back to all laps
        
        # Access the underlying FastF1 session
        fastf1_session = session.data  # or session()
    """
    
    def __init__(self, year: int, round: Union[int, str], session_name: str, test: bool = False, **kwargs):
        self.year = year
        self.round = round
        self.session_name = session_name
        self.test = test
        self._session = self._load_session(**kwargs)
        self._original_laps = self._session.laps  # Keep original for reset
        self.laps = self._session.laps  # Current filtered laps
        self.lap_distance = self._session.get_circuit_info().marshal_sectors['Distance'].max()
        self.race_lap_number = max(ceil(3e5 / self.lap_distance), 78)  # Approximate lap number for 300 km

        # Data enhancements
        self._add_stint_lap_number()
    
    def _load_session(self, **kwargs) -> FastF1Session:
        """Load the session data."""
        # Get the session object and load the data
        # if 'mpl_timedelta_support' not in kwargs:
        #     kwargs['mpl_timedelta_support'] = True
        # if 'color_scheme' not in kwargs:
        #     kwargs['color_scheme'] = 'fastf1'
        if not self.test:
            _data = fastf1.get_session(self.year, self.round, self.session_name, **kwargs)
        else:
            _data = fastf1.get_testing_event(self.year, self.round).get_session(self.session_name, **kwargs)
        _data.load()
        self._fill_missing_outlap_laptimes(_data)
        return _data

    def _fill_missing_outlap_laptimes(self, session: FastF1Session) -> None:
        """Backfill missing out-lap times from the next lap start when possible."""
        laps = session.laps
        required_columns = {'Driver', 'LapTime', 'PitOutTime', 'LapStartTime'}
        if not required_columns.issubset(laps.columns):
            return

        for driver in laps['Driver'].dropna().unique():
            driver_laps = laps.loc[laps['Driver'] == driver].sort_values('LapStartTime')
            next_lap_start = driver_laps['LapStartTime'].shift(-1)
            fill_mask = (
                driver_laps['LapTime'].isna()
                & driver_laps['PitOutTime'].notna()
                & next_lap_start.notna()
            )
            if not fill_mask.any():
                continue

            filled_laptimes = next_lap_start[fill_mask] - driver_laps.loc[fill_mask, 'PitOutTime']
            filled_laptimes = filled_laptimes[filled_laptimes > pd.Timedelta(0)]
            if filled_laptimes.empty:
                continue

            session.laps.loc[filled_laptimes.index, 'LapTime'] = filled_laptimes
    
    def _add_stint_lap_number(self) -> None:
        """Add StintLapNumber column to laps data."""
        stint_lap_numbers = []
        for driver in self.laps['Driver'].unique():
            driver_laps = self.laps[self.laps['Driver'] == driver]
            stints = driver_laps['Stint'].unique()
            for stint in stints:
                stint_laps = driver_laps[driver_laps['Stint'] == stint]
                lap_numbers = range(1, len(stint_laps) + 1)
                stint_lap_numbers.extend(lap_numbers)
        self.laps['StintLapNumber'] = stint_lap_numbers
        
    @property
    def data(self) -> FastF1Session:
        """Access the loaded FastF1 session data."""
        return self._session

    def __call__(self):
        """Allow calling the instance to get the data."""
        return self._session

    def reset(self) -> "Session":
        """Reset filters and return to original laps data.
        
        Returns:
            Session: Self for method chaining
            
        Example:
            >>> session = Session(2023, 1, 'R')
            >>> filtered = session.drivers(['HAM']).quicklaps()
            >>> session.reset()  # Reset to all laps
        """
        self.laps = self._original_laps
        return self

    def drivers(self, drivers: list[str]) -> "Session":
        """Filter the session data by drivers.
        
        Args:
            drivers: List of driver abbreviations or numbers
            
        Returns:
            Session: Self for method chaining
            
        Example:
            >>> session.drivers(['HAM', 'VER'])
        """
        self.laps = self.laps.pick_drivers(drivers)
        return self
    
    def quicklaps(self, threshold: float = 1.07) -> "Session":
        """Filter the session data by quick laps.
        
        Args:
            threshold: Threshold for quick lap detection (default 1.07 = 107%)
            
        Returns:
            Session: Self for method chaining
            
        Example:
            >>> session.quicklaps(threshold=1.05)
        """
        self.laps = self.laps.pick_quicklaps(threshold)
        return self
    
    def compounds(self, compounds: list[str]) -> "Session":
        """Filter the session data by compounds.
        
        Args:
            compounds: List of compound names (e.g., ['SOFT', 'MEDIUM'])
            
        Returns:
            Session: Self for method chaining
            
        Example:
            >>> session.compounds(['SOFT', 'MEDIUM'])
        """
        self.laps = self.laps.pick_compounds(compounds)
        return self
    
    def track_status(self, status: str) -> "Session":
        """Filter the session data by track status.
        
        Args:
            status: Track status (e.g., '1', '2', '4', '5', '6')
            
        Returns:
            Session: Self for method chaining
            
        Example:
            >>> session.track_status('1')  # All clear
        """
        self.laps = self.laps.pick_track_status(status)
        return self

    def stint(self, stint: int) -> "Session":
        """Filter the session data by driver and stint number.
        
        Args:
            stint: Stint number
            
        Returns:
            Session: Self for method chaining
            
        Example:
            >>> session.stint(1)  # First stint
        """
        self.laps = self.laps[self.laps['Stint'] == stint]
        return self
    
    def clean_laps(self, car_ahead: float) -> "Session":
        """Filter the session data by delta from the car ahead"""
        tel = self.laps.get_car_data()
        # Find all laps, and for each lap, mark if at any sampled time there's a car less than car_ahead seconds ahead
        # The DistanceToDriverAhead is in meters, but we want time gap.
        # Use Speed (in km/h) to convert: distance / (speed/3.6) = time gap in seconds
        # We want only laps where for ALL samples: (distance_to_ahead / speed) > car_ahead
        # Handle edge cases where DistanceToDriverAhead or Speed are nan/zero.

        # Need for each lap: get all samples, compute time gap to car ahead, see if all are > car_ahead
        mask = []
        for lap_number in self.laps['LapNumber']:
            lap_tel = tel[tel['LapNumber'] == lap_number]
            # Defensive: exclude cases where no data for lap
            if lap_tel.empty:
                mask.append(False)
                continue
            dist = lap_tel.get('DistanceToDriverAhead', None)
            spd = lap_tel.get('Speed', None)
            # If not present, skip lap
            if dist is None or spd is None:
                mask.append(False)
                continue
            # Avoid divide by zero: only compute timegap where spd > 1 km/h
            valid = (spd > 1) & dist.notna() & (dist > 0)
            timegap = pd.Series(float('inf'), index=lap_tel.index)
            # Calculate only for valid points
            timegap.loc[valid] = dist[valid] / (spd[valid] / 3.6)
            # For rest (empty/NaN/zero): treat as inf (=no car ahead very close; will not falsely fail)
            # Want all samples to have a gap > car_ahead seconds, else mark lap as "not clean"
            if (timegap > car_ahead).all():
                mask.append(True)
            else:
                mask.append(False)
        self.laps = self.laps[mask]
        return self
    
    def effective_stint(self) -> "Session":
        """Filter the session data to get long run distance by combining stints if they are close enough."""
        # Compute effective stints independently for each driver.
        self.laps['EffectiveStint'] = pd.NA
        self.laps['EffectiveStintLapNumber'] = pd.NA
        gap_threshold = pd.Timedelta(seconds=50)

        for driver in self.laps['Driver'].unique():
            print(f"Processing driver {driver}")
            driver_mask = self.laps['Driver'] == driver
            driver_laps = self.laps.loc[driver_mask]
            if driver_laps.empty:
                continue

            # Keep stint processing order based on first lap start time.
            stint_order = (
                driver_laps.groupby('Stint', sort=False)['LapStartTime']
                .min()
                .sort_values()
                .index
                .tolist()
            )
            print(f"  Stints found: {stint_order}")

            # Identify quali-sim stints for this driver only.
            print(f"  Get rid of quali-sim stints based on lap info")
            quali_sim_stints = set()
            for stint in stint_order:
                print(f"    Analyzing stint {stint} to check if it is quali lap, total lap count: {len(driver_laps.loc[driver_laps['Stint'] == stint])}")
                stint_laps = driver_laps.loc[driver_laps['Stint'] == stint]
                lap_count = len(stint_laps)

                if lap_count <= 3:
                    quali_sim_stints.add(stint)
                    print(f"      Marking stint {stint} as quali-sim (only {lap_count} laps)")
                    continue

                valid_lap_times = stint_laps['LapTime'].dropna()
                if valid_lap_times.empty:
                    print(f"      No valid lap times for stint {stint}, marking as quali-sim")
                    continue

                best_lap_time = valid_lap_times.min()
                slow_laps = valid_lap_times[valid_lap_times > 1.07 * best_lap_time]
                if len(slow_laps) / len(valid_lap_times) > 0.33:
                    quali_sim_stints.add(stint)
                    print(f"      Marking stint {stint} as quali-sim ({len(slow_laps)}/{len(valid_lap_times)} slow laps)")

            effective_stint_number = 1
            prev_stint_end_time = None
            prev_stint_quali_sim = False
            prev_effective_stint = None
            in_lap_correct = False
            out_lap_correct = None
            effective_lap_offset = {}

            for stint in stint_order:
                # print(f"    Processing stint {stint}")

                stint_mask = driver_mask & (self.laps['Stint'] == stint)
                stint_laps = self.laps.loc[stint_mask]
                if stint_laps.empty:
                    continue

                # if out_lap_correct:
                #     # Remove 20s from the LapTime for the first lap of the stint
                #     print(f"      Applying out lap correction to stint {stint}")
                #     first_lap_mask = stint_laps['StintLapNumber'] == 1
                #     second_lap_mask = stint_laps['StintLapNumber'] == 2
                #     self.laps.loc[driver_laps[first_lap_mask].index, 'LapTime'] = (
                #         self.laps.loc[driver_laps[second_lap_mask].index, 'LapStartTime'] + pd.Timedelta(seconds=20)
                #     )
                #     out_lap_correct = None  # Only apply to the first stint after the small gap


                is_quali_sim = stint in quali_sim_stints
                stint_start = stint_laps['PitOutTime'].min()
                stint_end = stint_laps['PitInTime'].max()
                tyre = stint_laps['Compound'].mode().iloc[0] if not stint_laps['Compound'].mode().empty else 'Unknown'
                print(f"      Stint {stint} start: {stint_start}, end: {stint_end}, duration: {stint_end - stint_start}, previous_stint_end: {prev_stint_end_time}, is_quali_sim: {is_quali_sim}, tyre: {tyre}")

                start_tyre_life = stint_laps['TyreLife'].min()

                combine_with_previous = (
                    prev_stint_end_time is not None
                    and not prev_stint_quali_sim
                    and not is_quali_sim
                    and (stint_start - prev_stint_end_time) < gap_threshold
                    and start_tyre_life == 1 # Only combine if a new tyre is put on
                )

                if prev_stint_end_time and ((stint_start - prev_stint_end_time) < gap_threshold):
                    # Sometimes teams will play tricks by just call the car through pitlane. If the gap is very small, we should combine regardless of quali-sim status, otherwise we may end up with many very short stints.
                    stop_time = stint_start - prev_stint_end_time
                    in_lap_correct = True
                    out_lap_correct = True
                    print(f"        Stint {stint} has very small gap to previous stint ({stint_start - prev_stint_end_time}, pre_end: {prev_stint_end_time}, start: {stint_start}), combining regardless of quali-sim status")
                    

                # print(f"      Stint {stint}, lap count: {len(stint_laps)}, start: {stint_start}, end: {stint_end}, duration: {stint_end - stint_start}, is_quali_sim: {is_quali_sim}, type: {tyre}， prev_stint_end_time: {prev_stint_end_time}, combine_with_previous: {combine_with_previous}")
                # print(f"          combine_with_previous: {combine_with_previous}, prev_stint_end_time: {prev_stint_end_time}, stint_start: {stint_start}, gap: {stint_start - prev_stint_end_time if prev_stint_end_time is not None else 'N/A'}, is_quali_sim: {is_quali_sim}, tyre: {tyre}")

                only_gap_throushold = (
                    prev_stint_end_time is not None
                    and not prev_stint_quali_sim
                    and not is_quali_sim
                    and (stint_start - prev_stint_end_time) > gap_threshold
                )
                if prev_stint_end_time and only_gap_throushold:
                    print(f"        Stint {stint} is NOT quali-sim but gap to previous stint is too large ({stint_start - prev_stint_end_time}), not combining")

                # if in_lap_correct:
                #     # Remove 3s from the LapTime for the last lap of the stint
                #     print(f"      Applying in lap correction to stint {stint}")
                #     last_lap_mask = stint_laps['StintLapNumber'] == stint_laps['StintLapNumber'].max()
                #     self.laps.loc[stint_laps[last_lap_mask].index, 'LapTime'] = (
                #         self.laps.loc[stint_laps[last_lap_mask].index, 'LapTime'] - pd.Timedelta(seconds=3)
                #     )
                #     in_lap_correct = False  # Only apply to the first stint after the small gap

                if combine_with_previous:
                    effective_stint = prev_effective_stint
                    lap_offset = effective_lap_offset[effective_stint]
                else:
                    effective_stint = effective_stint_number
                    effective_stint_number += 1
                    lap_offset = 0

                self.laps.loc[stint_mask, 'EffectiveStint'] = effective_stint
                self.laps.loc[stint_mask, 'EffectiveStintLapNumber'] = (
                    stint_laps['StintLapNumber'] + lap_offset
                )

                effective_lap_offset[effective_stint] = lap_offset + len(stint_laps)
                prev_stint_end_time = stint_laps['PitInTime'].max()
                prev_stint_quali_sim = is_quali_sim
                prev_effective_stint = effective_stint

        self.laps['EffectiveStint'] = self.laps['EffectiveStint'].astype('Int64')
        self.laps['EffectiveStintLapNumber'] = self.laps['EffectiveStintLapNumber'].astype('Int64')
        return self

def export_long_effective_stints(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    output_csv: str,
    min_laps: int = 40,
    test: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Export all laps from effective stints longer than `min_laps` into one CSV."""
    all_laps = []

    for round_number in rounds:
        for session_name in session_names:
            print(f"Loading {year} round={round_number} session={session_name}")
            try:
                session = Session(
                    year=year,
                    round=round_number,
                    session_name=session_name,
                    test=test,
                    **kwargs
                )
                session.effective_stint()
            except Exception as exc:
                print(
                    f"Skipping year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                continue

            if session.laps.empty:
                continue

            max_laps_per_effective_stint = (
                session.laps.groupby(['Driver', 'EffectiveStint'])['EffectiveStintLapNumber']
                .max()
                .reset_index(name='MaxEffectiveStintLapNumber')
            )
            valid_effective_stints = max_laps_per_effective_stint[
                max_laps_per_effective_stint['MaxEffectiveStintLapNumber'] > min_laps
            ][['Driver', 'EffectiveStint']]

            if valid_effective_stints.empty:
                continue

            session_long_stints = session.laps.merge(
                valid_effective_stints,
                on=['Driver', 'EffectiveStint'],
                how='inner'
            ).copy()
            session_long_stints['Year'] = year
            session_long_stints['Round'] = round_number
            session_long_stints['SessionName'] = session_name
            all_laps.append(session_long_stints)

    if all_laps:
        result = pd.concat(all_laps, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                'Year', 'Round', 'SessionName', 'Driver',
                'EffectiveStint', 'EffectiveStintLapNumber'
            ]
        )
    
    # Print out record of which team and which driver has done a long distance test, print the driver, team, number of laps, and when it happens. Order in team / dirver / number of laps / day
    if not result.empty:
        summary = (
            result.groupby(['Driver', 'Team'])['EffectiveStintLapNumber']
            .max()
            .reset_index(name='MaxEffectiveStintLapNumber')
            .sort_values(by=['MaxEffectiveStintLapNumber', 'Driver'], ascending=[False, True])
        )
        print("Summary of long effective stints:")
        print(summary.to_string(index=False))

    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} laps to {output_csv}")
    return result


def load_all_session_laps(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    test: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Load all laps for the requested sessions without long-stint filtering."""
    all_laps = []

    for round_number in rounds:
        for session_name in session_names:
            print(f"Loading full laps for {year} round={round_number} session={session_name}")
            try:
                session = Session(
                    year=year,
                    round=round_number,
                    session_name=session_name,
                    test=test,
                    **kwargs
                )
            except Exception as exc:
                print(
                    f"Skipping full laps for year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                continue

            if session.laps.empty:
                continue

            session_laps = session.laps.copy()
            session_laps['Year'] = year
            session_laps['Round'] = round_number
            session_laps['SessionName'] = session_name
            all_laps.append(session_laps)

    if all_laps:
        return pd.concat(all_laps, ignore_index=True)

    return pd.DataFrame(columns=['Year', 'Round', 'SessionName', 'Driver', 'Team'])

def plot_race_sim(
    laps: pd.DataFrame,
    min_laps: int = 57,
    reference_laps: int = 57,
    correction_map: dict[tuple[Union[int, str], Union[int, str]], float] = None,
    session_offset_map: dict[tuple[Union[int, str], Union[int, str]], float] = None,
    benchmark_session_key: tuple[Union[int, str], Union[int, str]] = None,
    title: str = None,
    output_path: str = None,
):
    """
    Draw race-simulation plots for long effective stints.

    The reference pace is built from the quickest effective stint with more than
    `min_laps` laps, using laps 2 through `reference_laps`. The plot shows
    cumulative delta to that constant reference pace:

        n * reference_avg_lap_time - cumulative_time_to_n_laps

    Each line represents one driver and uses the team color when available.
    Two plots are generated:
    - uncorrected
    - corrected by subtracting the day-specific delta from each afternoon lap
    """
    required_columns = {
        'Driver', 'Team', 'LapTime', 'EffectiveStint', 'EffectiveStintLapNumber',
        'Year', 'Round', 'SessionName', 'LapStartTime'
    }
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for race sim plot: {sorted(missing_columns)}"
        )

    plot_laps = laps.dropna(
        subset=['LapTime', 'EffectiveStint', 'EffectiveStintLapNumber', 'Driver']
    ).copy()
    if plot_laps.empty:
        raise ValueError("No valid laps available for plotting")

    plot_laps['LapTimeSeconds'] = plot_laps['LapTime'].dt.total_seconds()

    stint_summary = (
        plot_laps.groupby(['Year', 'Round', 'SessionName', 'Driver', 'Team', 'EffectiveStint'])
        .agg(
            lap_count=('EffectiveStintLapNumber', 'max')
        )
        .reset_index()
    )
    long_stints = stint_summary[stint_summary['lap_count'] > min_laps].copy()
    if long_stints.empty:
        raise ValueError(f"No effective stints longer than {min_laps} laps found")

    plot_laps = plot_laps.merge(
        long_stints[['Year', 'Round', 'SessionName', 'Driver', 'Team', 'EffectiveStint', 'lap_count']],
        on=['Year', 'Round', 'SessionName', 'Driver', 'Team', 'EffectiveStint'],
        how='inner'
    )
    if reference_laps < 1:
        raise ValueError("reference_laps must be at least 1")

    reference_lap_count = reference_laps - 1
    correction_map = correction_map or {}
    session_offset_map = session_offset_map or {}
    cutoff_seconds = 4.5 * 3600
    plot_laps['LapStartSeconds'] = plot_laps['LapStartTime'].dt.total_seconds()
    plot_laps['Period'] = plot_laps['LapStartSeconds'].map(
        lambda seconds: 'forenoon' if pd.notna(seconds) and seconds < cutoff_seconds else 'afternoon'
    )
    plot_laps['DayCorrectionSeconds'] = [
        correction_map.get((row.Round, row.SessionName), 0.0)
        for row in plot_laps[['Round', 'SessionName']].itertuples(index=False)
    ]
    plot_laps['CorrectedLapTimeSeconds'] = plot_laps['LapTimeSeconds']
    corrected_afternoon_mask = plot_laps['Period'] == 'afternoon'
    plot_laps.loc[corrected_afternoon_mask, 'CorrectedLapTimeSeconds'] = (
        plot_laps.loc[corrected_afternoon_mask, 'CorrectedLapTimeSeconds']
        - plot_laps.loc[corrected_afternoon_mask, 'DayCorrectionSeconds']
    )
    plot_laps['SessionOffsetSeconds'] = [
        session_offset_map.get((row.Round, row.SessionName), 0.0)
        for row in plot_laps[['Round', 'SessionName']].itertuples(index=False)
    ]
    plot_laps['AlignedCorrectedLapTimeSeconds'] = (
        plot_laps['CorrectedLapTimeSeconds'] - plot_laps['SessionOffsetSeconds']
    )
    representative_laps = plot_laps[
        (plot_laps['EffectiveStintLapNumber'] >= 2)
        & (plot_laps['EffectiveStintLapNumber'] <= reference_laps)
    ].copy()
    if representative_laps.empty:
        raise ValueError("No representative laps available after excluding lap 1")
    team_color_map, team_driver_linestyle_map, _ = _build_team_style_maps(representative_laps)

    print("Team-driver line style map:")
    print(team_driver_linestyle_map)

    def _make_plot(apply_correction: bool):
        lap_time_column = 'AlignedCorrectedLapTimeSeconds' if apply_correction else 'LapTimeSeconds'
        reference_candidates = representative_laps.copy()
        if benchmark_session_key is not None:
            reference_candidates = reference_candidates[
                (reference_candidates['Round'] == benchmark_session_key[0])
                & (reference_candidates['SessionName'] == benchmark_session_key[1])
            ].copy()
        reference_summary = (
            reference_candidates.groupby(
                ['Year', 'Round', 'SessionName', 'Driver', 'Team', 'EffectiveStint', 'lap_count']
            )
            .agg(
                reference_total_time_seconds=(lap_time_column, 'sum'),
                reference_completed_laps=('EffectiveStintLapNumber', 'nunique'),
                reference_first_lap=('EffectiveStintLapNumber', 'min'),
                reference_last_lap=('EffectiveStintLapNumber', 'max')
            )
            .reset_index()
        )
        reference_summary = reference_summary[
            (reference_summary['reference_completed_laps'] == reference_lap_count)
            & (reference_summary['reference_first_lap'] == 2)
            & (reference_summary['reference_last_lap'] == reference_laps)
        ].copy()
        if reference_summary.empty:
            raise ValueError(
                f"No effective stint has a complete representative window from lap 2 to lap {reference_laps}"
            )

        reference_stint = reference_summary.loc[
            reference_summary['reference_total_time_seconds'].idxmin()
        ]
        reference_avg_lap_time = (
            reference_stint['reference_total_time_seconds'] / reference_lap_count
        )
        reference_label = (
            f"Reference: {reference_stint['Driver']} ({reference_stint['Team']}), "
            f"stint {reference_stint['EffectiveStint']}, "
            f"laps 2-{reference_laps} in "
            f"{reference_stint['reference_total_time_seconds']:.3f}s, "
            f"avg {reference_avg_lap_time:.3f}s"
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        for (year, round_number, session_name, driver, team, effective_stint), stint_laps in representative_laps.groupby(
            ['Year', 'Round', 'SessionName', 'Driver', 'Team', 'EffectiveStint'], sort=False
        ):
            color = team_color_map.get(team)
            linestyle = team_driver_linestyle_map.get((team, driver), '-')

            stint_laps = stint_laps.sort_values('EffectiveStintLapNumber').copy()
            has_forenoon = (stint_laps['Period'] == 'forenoon').any()
            has_afternoon = (stint_laps['Period'] == 'afternoon').any()
            period_label = (
                'AM+PM'
                if has_forenoon and has_afternoon else
                'AM only'
                if has_forenoon else
                'PM only'
            )
            label = (
                f"R{round_number} S{session_name} {driver} ({team}, stint {effective_stint}, "
                f"{int(stint_laps['lap_count'].iloc[0])} laps)"
            )
            if apply_correction:
                session_correction = float(stint_laps['DayCorrectionSeconds'].iloc[0])
                session_offset = float(stint_laps['SessionOffsetSeconds'].iloc[0])
                cumulative_time = stint_laps['AlignedCorrectedLapTimeSeconds'].cumsum()
                label = (
                    f"{label} | {period_label} | corr {session_correction:.3f}s on PM"
                    f" | sess {session_offset:.3f}s"
                )
            else:
                cumulative_time = stint_laps['LapTimeSeconds'].cumsum()

            stint_laps['ReferenceElapsedSeconds'] = (
                (stint_laps['EffectiveStintLapNumber'] - 1) * reference_avg_lap_time
            )
            stint_laps['CumulativeDeltaSeconds'] = (
                stint_laps['ReferenceElapsedSeconds'] - cumulative_time
            )
            x_values = pd.concat(
                [
                    pd.Series([1], dtype='int64'),
                    stint_laps['EffectiveStintLapNumber'].reset_index(drop=True)
                ],
                ignore_index=True
            )
            y_values = pd.concat(
                [
                    pd.Series([0.0], dtype='float64'),
                    stint_laps['CumulativeDeltaSeconds'].reset_index(drop=True)
                ],
                ignore_index=True
            )
            ax.plot(
                x_values,
                y_values,
                label=label,
                color=color,
                linewidth=2,
                linestyle=linestyle
            )

        correction_label = (
            f"Corrected by per-session afternoon delta after 4.5h: {correction_map}; "
            f"session offsets: {session_offset_map}"
            if apply_correction else
            "Uncorrected"
        )
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Effective Stint Lap Number')
        ax.set_ylabel('Delta to Reference Pace (s)')
        ax.set_title(title or f'Race Sim Delta Plot\n{reference_label}\n{correction_label}')
        ax.grid(True, which='major', alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        return fig, ax

    raw_fig, raw_ax = _make_plot(apply_correction=False)
    corrected_fig, corrected_ax = _make_plot(apply_correction=True)

    if output_path:
        if '.' in output_path:
            base, ext = output_path.rsplit('.', 1)
            raw_fig.savefig(f"{base}_uncorrected.{ext}", dpi=150, bbox_inches='tight')
            corrected_fig.savefig(f"{base}_corrected.{ext}", dpi=150, bbox_inches='tight')
        else:
            raw_fig.savefig(f"{output_path}_uncorrected", dpi=150, bbox_inches='tight')
            corrected_fig.savefig(f"{output_path}_corrected", dpi=150, bbox_inches='tight')

    return {
        'uncorrected': (raw_fig, raw_ax),
        'corrected': (corrected_fig, corrected_ax),
        'correction_map': correction_map,
        'session_offset_map': session_offset_map,
    }


def plot_cumulative_laps_by_day(
    laps: pd.DataFrame,
    title: str = None,
    output_path: str = None,
):
    """Plot cumulative laps by session-day for drivers and teams."""
    required_columns = {'Round', 'SessionName', 'Driver', 'Team'}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for cumulative laps plot: {sorted(missing_columns)}"
        )

    plot_laps = laps.dropna(subset=['Round', 'SessionName', 'Driver', 'Team']).copy()
    if plot_laps.empty:
        raise ValueError("No valid laps available for cumulative laps plot")

    day_counts = (
        plot_laps.groupby(['Round', 'SessionName', 'Driver', 'Team'])
        .size()
        .reset_index(name='LapCount')
        .sort_values(['Round', 'SessionName', 'Driver'])
    )
    day_order = (
        day_counts[['Round', 'SessionName']]
        .drop_duplicates()
        .sort_values(['Round', 'SessionName'])
        .reset_index(drop=True)
    )
    day_order['DayIndex'] = range(1, len(day_order) + 1)
    day_order['DayLabel'] = [
        f"R{row.Round} S{row.SessionName}"
        for row in day_order.itertuples(index=False)
    ]

    cumulative_counts = day_counts.merge(
        day_order,
        on=['Round', 'SessionName'],
        how='left'
    )
    driver_cumulative_counts = cumulative_counts.copy()
    driver_cumulative_counts['CumulativeLapCount'] = (
        driver_cumulative_counts.sort_values(['Driver', 'DayIndex'])
        .groupby('Driver')['LapCount']
        .cumsum()
    )
    team_day_counts = (
        plot_laps.groupby(['Round', 'SessionName', 'Team'])
        .size()
        .reset_index(name='LapCount')
        .sort_values(['Round', 'SessionName', 'Team'])
    )
    team_cumulative_counts = team_day_counts.merge(
        day_order,
        on=['Round', 'SessionName'],
        how='left'
    )
    team_cumulative_counts['CumulativeLapCount'] = (
        team_cumulative_counts.sort_values(['Team', 'DayIndex'])
        .groupby('Team')['LapCount']
        .cumsum()
    )

    team_color_map, team_driver_linestyle_map, _ = _build_team_style_maps(plot_laps)

    driver_fig, driver_ax = plt.subplots(figsize=(12, 7))
    for (driver, team), driver_counts in driver_cumulative_counts.groupby(['Driver', 'Team'], sort=False):
        driver_counts = driver_counts.sort_values('DayIndex')
        driver_ax.plot(
            driver_counts['DayIndex'],
            driver_counts['CumulativeLapCount'],
            label=f"{driver} ({team})",
            color=team_color_map.get(team),
            linestyle=team_driver_linestyle_map.get((team, driver), '-'),
            linewidth=2,
            marker='o'
        )

    driver_ax.set_xlabel('Test Day')
    driver_ax.set_ylabel('Cumulative Lap Count')
    driver_ax.set_title(title or 'Cumulative Driver Laps by Day')
    driver_ax.set_xticks(day_order['DayIndex'])
    driver_ax.set_xticklabels(day_order['DayLabel'], rotation=45, ha='right')
    driver_ax.grid(True, alpha=0.3)
    driver_ax.legend(loc='best', fontsize=9)
    driver_fig.tight_layout()

    team_fig, team_ax = plt.subplots(figsize=(12, 7))
    for team, team_counts in team_cumulative_counts.groupby('Team', sort=False):
        team_counts = team_counts.sort_values('DayIndex')
        team_ax.plot(
            team_counts['DayIndex'],
            team_counts['CumulativeLapCount'],
            label=team,
            color=team_color_map.get(team),
            linewidth=2.5,
            marker='o'
        )

    team_ax.set_xlabel('Test Day')
    team_ax.set_ylabel('Cumulative Lap Count')
    team_ax.set_title('Cumulative Team Laps by Day')
    team_ax.set_xticks(day_order['DayIndex'])
    team_ax.set_xticklabels(day_order['DayLabel'], rotation=45, ha='right')
    team_ax.grid(True, alpha=0.3)
    team_ax.legend(loc='best', fontsize=9)
    team_fig.tight_layout()

    if output_path:
        if '.' in output_path:
            base, ext = output_path.rsplit('.', 1)
            driver_fig.savefig(f"{base}_drivers.{ext}", dpi=150, bbox_inches='tight')
            team_fig.savefig(f"{base}_teams.{ext}", dpi=150, bbox_inches='tight')
        else:
            driver_fig.savefig(f"{output_path}_drivers", dpi=150, bbox_inches='tight')
            team_fig.savefig(f"{output_path}_teams", dpi=150, bbox_inches='tight')

    return {
        'drivers': (driver_fig, driver_ax, driver_cumulative_counts),
        'teams': (team_fig, team_ax, team_cumulative_counts),
        'day_order': day_order,
    }


def plot_single_lap_comparison(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    output_path: str = None,
    correction_map: dict[tuple[Union[int, str], Union[int, str]], float] = None,
    test: bool = False,
    **kwargs
):
    """Plot best-lap deltas to the overall quickest lap as a bar chart."""
    all_laps = []
    correction_map = correction_map or {}

    for round_number in rounds:
        for session_name in session_names:
            print(f"Loading single-lap data for {year} round={round_number} session={session_name}")
            try:
                session = Session(
                    year=year,
                    round=round_number,
                    session_name=session_name,
                    test=test,
                    **kwargs
                )
            except Exception as exc:
                print(
                    f"Skipping single-lap data for year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                continue

            session_laps = session.laps.copy()
            session_laps['Year'] = year
            session_laps['Round'] = round_number
            session_laps['SessionName'] = session_name
            all_laps.append(session_laps)

    if not all_laps:
        raise ValueError("No laps were loaded for single-lap comparison")

    laps = pd.concat(all_laps, ignore_index=True)
    laps = laps.dropna(subset=['LapTime', 'Driver', 'Team', 'LapStartTime']).copy()
    if laps.empty:
        raise ValueError("No valid laps available for single-lap comparison")

    # Exclude in-laps and out-laps to keep representative single-lap attempts only.
    laps = laps[laps['PitOutTime'].isna() & laps['PitInTime'].isna()].copy()
    if laps.empty:
        raise ValueError("No representative laps remain after removing in-laps/out-laps")

    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    cutoff_seconds = 4.5 * 3600
    laps['LapStartSeconds'] = laps['LapStartTime'].dt.total_seconds()
    laps['Period'] = laps['LapStartSeconds'].map(
        lambda seconds: 'forenoon' if pd.notna(seconds) and seconds < cutoff_seconds else 'afternoon'
    )
    laps['CorrectionSeconds'] = [
        correction_map.get((row.Round, row.SessionName), 0.0)
        for row in laps[['Round', 'SessionName']].itertuples(index=False)
    ]
    laps['AdjustedLapTimeSeconds'] = laps['LapTimeSeconds']
    afternoon_mask = laps['Period'] == 'afternoon'
    laps.loc[afternoon_mask, 'AdjustedLapTimeSeconds'] = (
        laps.loc[afternoon_mask, 'AdjustedLapTimeSeconds']
        - laps.loc[afternoon_mask, 'CorrectionSeconds']
    )

    best_laps = (
        laps.sort_values('AdjustedLapTimeSeconds')
        .groupby('Driver', as_index=False)
        .first()
        .sort_values('AdjustedLapTimeSeconds')
        .reset_index(drop=True)
    )
    if best_laps.empty:
        raise ValueError("No best laps available for single-lap comparison")

    quickest_lap = float(best_laps['AdjustedLapTimeSeconds'].iloc[0])
    best_laps['DeltaToQuickestSeconds'] = best_laps['AdjustedLapTimeSeconds'] - quickest_lap

    team_color_map, _, team_driver_hatch_map = _build_team_style_maps(
        best_laps[['Team', 'Driver']].copy()
    )
    bar_colors = [team_color_map.get(team) for team in best_laps['Team']]
    bar_hatches = [
        team_driver_hatch_map.get((team, driver))
        for team, driver in best_laps[['Team', 'Driver']].itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        best_laps['Driver'],
        best_laps['DeltaToQuickestSeconds'],
        color=bar_colors,
        edgecolor='black',
        linewidth=0.8
    )
    for bar, hatch in zip(bars, bar_hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Driver')
    ax.set_ylabel('Delta to Quickest Lap (s)')
    ax.set_title('Single-Lap Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, ax, best_laps


def run_two_day_benchmark_race_sim(
    year: int,
    round_number: Union[int, str],
    benchmark_session: Union[int, str],
    comparison_session: Union[int, str],
    output_prefix: str,
    min_laps: int = 57,
    reference_laps: int = 57,
    test: bool = False,
    **kwargs
):
    """
    Run a dedicated two-day race-sim analysis.

    Workflow:
    1. Load only the benchmark and comparison sessions.
    2. Compute AM/PM correction for both sessions.
    3. Compute an additional AM-to-AM session offset relative to the benchmark.
    4. Plot race sim using the benchmark session as the fixed reference.
    """
    session_names = [benchmark_session, comparison_session]
    correction_map = forenoon_afternoon_delta(
        year=year,
        rounds=[round_number],
        session_names=session_names,
        output_csv=f"{output_prefix}_ampm_delta.csv",
        min_laps=min_laps,
        test=test,
        **kwargs
    )

    long_laps = export_long_effective_stints(
        year=year,
        rounds=[round_number],
        session_names=session_names,
        output_csv=f"{output_prefix}_long_stints.csv",
        min_laps=min_laps,
        test=test,
        **kwargs
    )
    if long_laps.empty:
        raise ValueError("No long-stint laps available for the two-day benchmark race-sim workflow")

    aligned_laps = long_laps.dropna(
        subset=['LapTime', 'LapStartTime', 'Round', 'SessionName', 'EffectiveStintLapNumber']
    ).copy()
    aligned_laps['LapTimeSeconds'] = aligned_laps['LapTime'].dt.total_seconds()
    cutoff_seconds = 4.5 * 3600
    aligned_laps['LapStartSeconds'] = aligned_laps['LapStartTime'].dt.total_seconds()
    aligned_laps['Period'] = aligned_laps['LapStartSeconds'].map(
        lambda seconds: 'forenoon' if pd.notna(seconds) and seconds < cutoff_seconds else 'afternoon'
    )
    aligned_laps['CorrectionSeconds'] = [
        correction_map.get((row.Round, row.SessionName), 0.0)
        for row in aligned_laps[['Round', 'SessionName']].itertuples(index=False)
    ]
    aligned_laps['SessionCorrectedLapTimeSeconds'] = aligned_laps['LapTimeSeconds']
    afternoon_mask = aligned_laps['Period'] == 'afternoon'
    aligned_laps.loc[afternoon_mask, 'SessionCorrectedLapTimeSeconds'] = (
        aligned_laps.loc[afternoon_mask, 'SessionCorrectedLapTimeSeconds']
        - aligned_laps.loc[afternoon_mask, 'CorrectionSeconds']
    )

    morning_laps = aligned_laps[
        (aligned_laps['Period'] == 'forenoon')
        & (aligned_laps['EffectiveStintLapNumber'] >= 2)
        & (aligned_laps['EffectiveStintLapNumber'] <= reference_laps)
    ].copy()
    morning_session_summary = (
        morning_laps.groupby(['Round', 'SessionName'])
        .agg(avg_lap_time_seconds=('SessionCorrectedLapTimeSeconds', 'mean'))
        .reset_index()
    )
    benchmark_key = (round_number, benchmark_session)
    comparison_key = (round_number, comparison_session)
    benchmark_am = morning_session_summary[
        (morning_session_summary['Round'] == benchmark_key[0])
        & (morning_session_summary['SessionName'] == benchmark_key[1])
    ]
    comparison_am = morning_session_summary[
        (morning_session_summary['Round'] == comparison_key[0])
        & (morning_session_summary['SessionName'] == comparison_key[1])
    ]
    if benchmark_am.empty or comparison_am.empty:
        raise ValueError("Unable to compute AM-to-AM session delta for the selected two-day workflow")

    session_offset = float(
        comparison_am['avg_lap_time_seconds'].iloc[0]
        - benchmark_am['avg_lap_time_seconds'].iloc[0]
    )
    session_offset_map = {
        benchmark_key: 0.0,
        comparison_key: session_offset,
    }
    print(f"Two-day session offset map relative to benchmark {benchmark_key}: {session_offset_map}")

    plots = plot_race_sim(
        long_laps,
        min_laps=min_laps,
        reference_laps=reference_laps,
        correction_map=correction_map,
        session_offset_map=session_offset_map,
        benchmark_session_key=benchmark_key,
        title=f"Two-Day Race Sim Benchmark R{round_number} S{benchmark_session} vs S{comparison_session}",
        output_path=f"{output_prefix}_race_sim.png"
    )

    return {
        'correction_map': correction_map,
        'session_offset_map': session_offset_map,
        'long_laps': long_laps,
        'plots': plots,
        'morning_session_summary': morning_session_summary,
    }


def forenoon_afternoon_delta(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    output_csv: str,
    min_laps: int = 40,
    test: bool = False,
    **kwargs
) -> dict[tuple[Union[int, str], Union[int, str]], float]:
    """
    Calculate the average lap-time delta between forenoon and afternoon.

    Filtering rules:
    1. Remove isolated quali-sim laps: a single lap whose previous and next lap
       are both slower than 107% of the day's fastest lap.
    2. Remove all laps slower than 115% of the day's fastest lap.
    3. Split remaining laps by `LapStartTime` at 4.5 hours from session start
       (11:30 local for a 07:00-16:00 test day) and compare average lap times
       between forenoon and afternoon.
    """
    all_laps = []

    for round_number in rounds:
        for session_name in session_names:
            print(f"Loading {year} round={round_number} session={session_name}")
            try:
                session = Session(
                    year=year,
                    round=round_number,
                    session_name=session_name,
                    test=test,
                    **kwargs
                )
                session.effective_stint()
            except Exception as exc:
                print(
                    f"Skipping year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                continue

            if session.laps.empty:
                continue

            session_laps = session.laps.copy()
            session_laps['Year'] = year
            session_laps['Round'] = round_number
            session_laps['SessionName'] = session_name
            all_laps.append(session_laps)

    if not all_laps:
        raise ValueError("No laps were loaded for forenoon/afternoon comparison")

    laps = pd.concat(all_laps, ignore_index=True)
    laps = laps.dropna(subset=['LapTime', 'LapNumber', 'Driver', 'LapStartDate']).copy()
    if laps.empty:
        raise ValueError("No valid laps available after dropping missing values")

    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['DayKey'] = list(zip(laps['Round'], laps['SessionName']))

    day_fastest_lap = laps.groupby('DayKey')['LapTimeSeconds'].transform('min')
    laps['Slow107'] = laps['LapTimeSeconds'] > (1.07 * day_fastest_lap)
    laps['Slow115'] = laps['LapTimeSeconds'] > (1.15 * day_fastest_lap)

    laps = laps.sort_values(['Round', 'SessionName', 'Driver', 'LapNumber']).copy()
    laps['PrevSlow107'] = laps.groupby(['Round', 'SessionName', 'Driver'])['Slow107'].shift(1)
    laps['NextSlow107'] = laps.groupby(['Round', 'SessionName', 'Driver'])['Slow107'].shift(-1)

    isolated_quali_sim_mask = (
        ~laps['Slow107']
        & laps['PrevSlow107'].fillna(False)
        & laps['NextSlow107'].fillna(False)
    )
    laps = laps.loc[~isolated_quali_sim_mask].copy()
    laps = laps.loc[~laps['Slow115']].copy()
    if laps.empty:
        raise ValueError("No laps remain after applying the lap-time filters")

    cutoff_seconds = 4.5 * 3600
    laps['LapStartSeconds'] = laps['LapStartTime'].dt.total_seconds()
    laps['Period'] = laps['LapStartSeconds'].map(
        lambda seconds: 'forenoon' if pd.notna(seconds) and seconds < cutoff_seconds else 'afternoon'
    )
    print("Lap counts by period after filtering:")
    print(laps['Period'].value_counts())

    paired_summary = (
        laps.groupby(['Round', 'SessionName', 'Period'])
        .agg(
            avg_lap_time_seconds=('LapTimeSeconds', 'mean'),
            sample_count=('LapTimeSeconds', 'size')
        )
        .reset_index()
        .pivot_table(
            index=['Round', 'SessionName'],
            columns='Period',
            values=['avg_lap_time_seconds', 'sample_count']
        )
        .reset_index()
    )
    paired_summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in paired_summary.columns.to_flat_index()
    ]
    for period in ('forenoon', 'afternoon'):
        avg_col = f'avg_lap_time_seconds_{period}'
        count_col = f'sample_count_{period}'
        if avg_col not in paired_summary.columns:
            paired_summary[avg_col] = pd.NA
        if count_col not in paired_summary.columns:
            paired_summary[count_col] = 0

    paired_summary = paired_summary.dropna(
        subset=['avg_lap_time_seconds_forenoon', 'avg_lap_time_seconds_afternoon']
    ).copy()

    if paired_summary.empty:
        raise ValueError("No sessions have both forenoon and afternoon data")

    paired_summary['DeltaSeconds'] = (
        paired_summary['avg_lap_time_seconds_afternoon']
        - paired_summary['avg_lap_time_seconds_forenoon']
    )
    paired_summary.to_csv(output_csv, index=False)
    print(f"Saved paired forenoon/afternoon summary to {output_csv}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        paired_summary['avg_lap_time_seconds_forenoon'],
        paired_summary['avg_lap_time_seconds_afternoon'],
        color='#1f77b4',
        s=70,
        alpha=0.8
    )

    min_time = min(
        paired_summary['avg_lap_time_seconds_forenoon'].min(),
        paired_summary['avg_lap_time_seconds_afternoon'].min()
    )
    max_time = max(
        paired_summary['avg_lap_time_seconds_forenoon'].max(),
        paired_summary['avg_lap_time_seconds_afternoon'].max()
    )
    ax.plot([min_time, max_time], [min_time, max_time], linestyle='--', color='black')
    ax.set_xlabel('Forenoon Average Lap Time (s)')
    ax.set_ylabel('Afternoon Average Lap Time (s)')
    ax.set_title('Forenoon vs Afternoon Pace')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_plot = output_csv.rsplit('.', 1)[0] + '.png'
    fig.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"Saved forenoon/afternoon scatter plot to {output_plot}")

    correction_map = {
        (row.Round, row.SessionName): row.DeltaSeconds
        for row in paired_summary[['Round', 'SessionName', 'DeltaSeconds']].itertuples(index=False)
    }
    print(f"Per-session afternoon - forenoon delta map: {correction_map}")
    return correction_map

if __name__ == "__main__":    # Example usage
    full_laps = load_all_session_laps(
        year=2026,
        rounds=[1, 2],
        session_names=[1, 2, 3],
        test=True
    )
    
    cumulative_laps_plots = plot_cumulative_laps_by_day(
        full_laps,
        output_path='/Users/zhxutong/dr-wo/temp/cumulative_laps_by_day.png'
    )

    two_day_race_sim = run_two_day_benchmark_race_sim(
        year=2026,
        round_number=2,
        benchmark_session=2,
        comparison_session=3,
        output_prefix='/Users/zhxutong/dr-wo/temp/r2s2_r2s3',
        min_laps=30,
        reference_laps=57,
        test=True
    )

    single_lap_plot = plot_single_lap_comparison(
        year=2026,
        rounds=[2],
        session_names=[2, 3],
        output_path='/Users/zhxutong/dr-wo/temp/single_lap_comparison.png',
        correction_map=two_day_race_sim['correction_map'],
        test=True
    )

    two_day_race_sim['plots']['uncorrected'][0].show()
    two_day_race_sim['plots']['corrected'][0].show()
    cumulative_laps_plots['drivers'][0].show()
    cumulative_laps_plots['teams'][0].show()
    single_lap_plot[0].show()
