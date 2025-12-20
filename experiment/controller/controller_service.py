from dataclasses import dataclass
from enum import Enum
import numpy as np


def celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    return kelvin - 273.15


@dataclass
class ThermalSystemParams:
    heating_rate_k_per_step: float  # Temperature increase per step when heater is ON (K/step)
    cooling_coefficient: float  # Cooling rate coefficient (fraction of temp difference lost per step)
    ambient_temp_k: float  # Ambient temperature (Kelvin)


@dataclass
class ControllerServiceConfig:
    temp_min: float  
    temp_max: float 
    steps_per_hour: int
    grid_step: float = 0.1
    grid_pad: float = 3.0
    penalty_scale: float = 1_000.0 


class Action(Enum):
    OFF = 0
    ON = 1


@dataclass
class TrajectoryStep:
    action: Action
    predicted_temperature: float  # Kelvin
    predicted_cost: float  # EUR for this step
    spot_price: float  # EUR/kWh for this step
    fcr_revenue: float  # EUR for this step


@dataclass
class ControllerServiceResult:
    predicted_power: float
    trajectory: list[TrajectoryStep]


class ControllerService:
    def __init__(
        self,
        initial_measurements: ThermalSystemParams,
        config: ControllerServiceConfig,
    ):
        self.thermal_system = initial_measurements
        self.config = config

    def _build_temperature_grid(self, ambient_temp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temperature grid and precompute transitions for OFF/ON actions.
        
        Args:
            ambient_temp: Current ambient temperature (Kelvin)
            
        Returns:
            Tuple of (temps_grid, next_idx_off, next_idx_on, penalty_lut)
            - temps_grid: Array of discrete temperature values (Kelvin)
            - next_idx_off: Next state index for OFF action at each grid point
            - next_idx_on: Next state index for ON action at each grid point
            - penalty_lut: Penalty value for each temperature state
        """
        grid_min = self.config.temp_min - self.config.grid_pad
        grid_max = self.config.temp_max + self.config.grid_pad
        temps_grid = np.arange(grid_min, grid_max + self.config.grid_step, self.config.grid_step)
        S = len(temps_grid)
        
        def clamp_to_idx(temp: float) -> int:
            """Convert temperature to nearest grid index, clamped to valid range."""
            j = int(np.round((temp - grid_min) / self.config.grid_step))
            return max(0, min(S - 1, j))
        
        # Precompute transitions for each grid point
        next_idx_off = np.empty(S, dtype=np.int32)
        next_idx_on = np.empty(S, dtype=np.int32)
        
        for s in range(S):
            t_now = temps_grid[s]
            temp_diff = t_now - ambient_temp
            
            # OFF: only cooling (temperature-dependent)
            cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff
            next_temp_off = t_now + cooling_delta
            next_idx_off[s] = clamp_to_idx(next_temp_off)
            
            # ON: cooling + heating
            heating_delta = self.thermal_system.heating_rate_k_per_step
            next_temp_on = t_now + cooling_delta + heating_delta
            next_idx_on[s] = clamp_to_idx(next_temp_on)
        
        # Penalty lookup table: cubic penalty for constraint violations
        low = np.maximum(0.0, self.config.temp_min - temps_grid)
        high = np.maximum(0.0, temps_grid - self.config.temp_max)
        penalty_lut = self.config.penalty_scale * (low**3 + high**3)
        
        return temps_grid, next_idx_off, next_idx_on, penalty_lut

    def _dp_solve(
        self,
        current_temp: float,
        ambient_temp: float,
        future_prices: list[float],
        watts_on: float,
        optimization_steps: int,
        fcr_d_down_price: float,
        fcr_d_up_price: float
    ) -> list[Action]:
        """
        Solve optimal ON/OFF heating schedule via dynamic programming.
        
        Returns:
            List of optimal actions for each time step
        """
        # Build temperature grid and transition tables
        temps_grid, next_idx_off, next_idx_on, penalty_lut = self._build_temperature_grid(ambient_temp)
        S = len(temps_grid)
        T = optimization_steps
        
        # Value function and policy tables
        V = np.zeros((T + 1, S))
        pi = np.zeros((T, S), dtype=np.uint8)  # 0=OFF, 1=ON
        
        # Backward pass: compute optimal value and policy
        for t in range(T - 1, -1, -1):
            # Get price for this time step (convert step index to hour index)
            hour_idx = t // self.config.steps_per_hour
            if hour_idx < len(future_prices):
                price = future_prices[hour_idx]
            else:
                price = future_prices[-1] if future_prices else 0.0
            
            # Energy cost per step when ON
            energy_kwh = (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)
            energy_cost = energy_kwh * price
            
            # Compute cost for OFF action at each state
            cost_off = penalty_lut[next_idx_off] + V[t + 1, next_idx_off]
            # Up-regulating is the same as turning off to release electricity.
            # So we subtract the FCR-D up price.
            cost_off -= fcr_d_up_price * (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)
            
            # Compute cost for ON action at each state
            cost_on = energy_cost + penalty_lut[next_idx_on] + V[t + 1, next_idx_on]
            # Down-regulating is the same as turning on to use electricity.
            # So we subtract the FCR-D down price.
            cost_on -= fcr_d_down_price * (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)
            
            # Choose better action at each state
            better_is_on = cost_on < cost_off
            V[t] = np.where(better_is_on, cost_on, cost_off)
            pi[t] = better_is_on.astype(np.uint8)
        
        # Find starting state index (closest grid point to current temperature)
        grid_min = temps_grid[0]
        s0 = int(np.clip(np.round((current_temp - grid_min) / self.config.grid_step), 0, S - 1))
        
        # Forward pass: extract optimal action sequence
        actions = []
        s = s0
        for t in range(T):
            a = pi[t, s]
            action = Action.ON if a else Action.OFF
            actions.append(action)
            # Move to next state
            s = next_idx_on[s] if a else next_idx_off[s]
        
        return actions

    def get_next_action(
        self,
        current_temp: float,
        future_prices: list[float],
        ambient_temp: float,
        watts_on: float,
        fcr_d_down_price: float,
        fcr_d_up_price: float
    ) -> ControllerServiceResult:
        # Limit optimization horizon (max 24 hours look-ahead) 
        optimization_horizon_hours = min(24, len(future_prices))
        optimization_steps = optimization_horizon_hours * self.config.steps_per_hour

        # Solve using dynamic programming
        actions = self._dp_solve(
            current_temp=current_temp,
            ambient_temp=ambient_temp,
            future_prices=future_prices,
            watts_on=watts_on,
            optimization_steps=optimization_steps,
            fcr_d_down_price=fcr_d_down_price,
            fcr_d_up_price=fcr_d_up_price
        )

        # Build detailed trajectory with temperature and cost predictions
        trajectory = self._build_trajectory_with_details(
            actions=actions,
            initial_temp=current_temp,
            ambient_temp=ambient_temp,
            future_prices=future_prices,
            watts_on=watts_on,
            fcr_d_down_price=fcr_d_down_price,
            fcr_d_up_price=fcr_d_up_price
        )

        return ControllerServiceResult(
            predicted_power=watts_on if actions[0] == Action.ON else 0.0,
            trajectory=trajectory,
        )
    

    def _predict_future_temperature(
        self, action: Action, current_temp: float, ambient_temp: float
    ) -> float:
        """
        Predict next temperature.

        Note: Temperature is in Kelvin.
        """
        # Temperature difference from ambient
        temp_diff = current_temp - ambient_temp

        # Heat loss (always occurs)
        cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff

        # Heating (only when ON)
        heating_delta = self.thermal_system.heating_rate_k_per_step if action == Action.ON else 0.0

        # Total temperature change
        delta_temp = heating_delta + cooling_delta

        return current_temp + delta_temp

    def _simulate_temperature_trajectory(
        self, action_sequence: list[Action], initial_temp: float, ambient_temp: float
    ) -> list[float]:
        """
        Simulate the temperature trajectory over the entire action sequence.

        Note: Temperature is in Kelvin.

        Returns:
            List of temperatures at each step (Kelvin)
        """
        trajectory = []
        current_temp = initial_temp

        for action in action_sequence:
            # Predict next temperature based on action
            current_temp = self._predict_future_temperature(action, current_temp, ambient_temp)
            trajectory.append(current_temp)

        return trajectory

    def _build_trajectory_with_details(
        self,
        actions: list[Action],
        initial_temp: float,
        ambient_temp: float,
        future_prices: list[float],
        watts_on: float,
        fcr_d_down_price: float,
        fcr_d_up_price: float
    ) -> list[TrajectoryStep]:
        """
        Build a detailed trajectory with temperature, cost, and price for each step.

        Returns:
            List of TrajectoryStep objects with detailed predictions
        """
        trajectory = []
        current_temp = initial_temp

        for step_idx, action in enumerate(actions):
            # Get price for this step (hourly prices)
            hour_idx = step_idx // self.config.steps_per_hour
            if hour_idx < len(future_prices):
                price = future_prices[hour_idx]
            else:
                price = future_prices[-1] if future_prices else 0.0

            # Predict next temperature
            next_temp = self._predict_future_temperature(action, current_temp, ambient_temp)

            # Calculate cost for this step
            fcr_revenue = 0.0
            if action == Action.ON:
                energy_kwh = (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)
                cost = energy_kwh * price
                fcr_revenue = fcr_d_down_price * (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)
            else:
                cost = 0.0
                fcr_revenue = fcr_d_up_price * (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)

            # Store trajectory step
            trajectory.append(TrajectoryStep(
                action=action,
                predicted_temperature=next_temp,
                predicted_cost=cost,
                spot_price=price,
                fcr_revenue=fcr_revenue,
            ))

            # Update for next iteration
            current_temp = next_temp

        return trajectory
