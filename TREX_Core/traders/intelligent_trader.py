import asyncio
from datetime import datetime
import TREX_Core.utils as utils
import math
import statistics

class Trader:
    """An intelligent trader that balances economic optimization with battery management.
    
    This trader uses value-based decision making, adaptive pricing, and battery state awareness
    to optimize market participation and energy use. It works with the existing TREX infrastructure
    without requiring external ML libraries.
    
    Key features:
    - Calculates energy value based on time of day and battery state
    - Adjusts bid/ask prices based on market conditions
    - Manages battery charging/discharging to balance market opportunities with local needs
    - Tracks market history to improve future decisions
    - Handles both market participation and battery scheduling
    """
    
    def __init__(self, **kwargs):
        # Initialize participant reference
        self.__participant = kwargs['trader_fns']
        
        # Configuration parameters with defaults
        self.min_bid_price = kwargs.get('min_bid_price', 0.05)
        self.max_bid_price = kwargs.get('max_bid_price', 0.25)
        self.min_ask_price = kwargs.get('min_ask_price', 0.05)
        self.max_ask_price = kwargs.get('max_ask_price', 0.25)
        
        # Battery management parameters
        self.min_reserve_soc = kwargs.get('min_reserve_soc', 0.2)  # Minimum state of charge
        self.target_peak_soc = kwargs.get('target_peak_soc', 0.9)  # Target SoC during peak hours
        self.target_offpeak_soc = kwargs.get('target_offpeak_soc', 0.5)  # Target SoC during off-peak
        
        # Market history and tracking
        self.settlement_history = {}
        self.price_history = {}
        self.action_scenario_history = {}
        
        # Time patterns (hours in 24h format)
        self.peak_demand_hours = (16, 17, 18, 19, 20, 21, 22)  # 4PM-10PM
        self.solar_production_hours = (8, 9, 10, 11, 12, 13, 14, 15, 16)  # 8AM-4PM
        
        # Adaptative parameters
        self.price_adjustment_factor = 1.0
        self.success_count = 0
        self.attempt_count = 0
    
    def _is_peak_hour(self, timestamp):
        """Determine if a timestamp falls within peak demand hours."""
        hour = utils.timestamp_to_local(timestamp, self.__participant['timing']['timezone']).hour
        return hour in self.peak_demand_hours
    
    def _is_solar_production_hour(self, timestamp):
        """Determine if a timestamp falls within peak solar production hours."""
        hour = utils.timestamp_to_local(timestamp, self.__participant['timing']['timezone']).hour
        return hour in self.solar_production_hours
    
    def _calculate_energy_value(self, timestamp, battery_soc):
        """Calculate the marginal value of energy for the given time."""
        # Base value depends on time of day
        if self._is_peak_hour(timestamp):
            base_value = 0.20  # Higher value during peak demand
        elif self._is_solar_production_hour(timestamp):
            base_value = 0.10  # Lower value during high solar production
        else:
            base_value = 0.15  # Medium value during other times
        
        # Adjust based on battery state - energy more valuable when battery low
        battery_factor = 1.0 + max(0, self.min_reserve_soc - battery_soc) * 2
        
        # If we have price history, incorporate it
        time_key = utils.timestamp_to_local(timestamp, self.__participant['timing']['timezone']).hour
        if time_key in self.price_history and len(self.price_history[time_key]) > 0:
            avg_price = statistics.mean(self.price_history[time_key])
            # Blend historical prices with calculated value
            base_value = (base_value + avg_price) / 2
        
        return min(self.max_ask_price, max(self.min_ask_price, base_value * battery_factor))
    
    def _determine_prices(self, energy_value, is_buying):
        """Set bid/ask prices based on value and market conditions."""
        if is_buying:
            # Bid price: lower than energy value (willing to buy below value)
            price = max(self.min_bid_price, energy_value * 0.85 * self.price_adjustment_factor)
            return min(self.max_bid_price, price)
        else:
            # Ask price: higher than energy value (willing to sell above value)
            price = max(self.min_ask_price, energy_value * 1.15 * self.price_adjustment_factor)
            return min(self.max_ask_price, price)
    
    def _determine_battery_strategy(self, current_soc, timestamp, residual_load, residual_gen, max_charge, max_discharge):
        """Decide optimal battery action based on current state and time patterns."""
        hour = utils.timestamp_to_local(timestamp, self.__participant['timing']['timezone']).hour
        
        # Determine target SoC based on time of day
        if self._is_peak_hour(timestamp):
            # During peak demand, prefer to discharge to meet load and sell to market
            target_soc = self.min_reserve_soc
        elif self._is_solar_production_hour(timestamp):
            # During solar production, prefer to charge for later use
            target_soc = self.target_peak_soc
        else:
            # Default target
            target_soc = self.target_offpeak_soc
        
        # Calculate SoC gap
        soc_gap = target_soc - current_soc
        
        # Battery actions are in Wh, so we need battery capacity
        capacity = self.__participant['storage']['info']()['capacity']
        
        if residual_load > 0:  # We need energy
            if current_soc > self.min_reserve_soc:
                # Discharge but preserve minimum reserve
                max_discharge_amount = abs(max_discharge)  # max_discharge is negative
                available_energy = (current_soc - self.min_reserve_soc) * capacity
                return -min(max_discharge_amount, available_energy, residual_load)
            else:
                # Battery too low, don't discharge
                return 0
        elif residual_gen > 0:  # We have excess energy
            if soc_gap > 0:
                # Charge toward target
                available_capacity = soc_gap * capacity
                return min(max_charge, available_capacity, residual_gen)
            else:
                # Above target, prioritize market selling
                return 0
        else:
            # If balanced, adjust toward target soc
            adjustment_rate = 0.1  # Adjust by up to 10% of capacity per period
            adjustment = soc_gap * adjustment_rate * capacity
            if adjustment > 0:
                return min(adjustment, max_charge)
            else:
                return max(adjustment, max_discharge)
    
    def _update_price_history(self, timestamp, price):
        """Track price history by hour of day for future reference."""
        hour = utils.timestamp_to_local(timestamp, self.__participant['timing']['timezone']).hour
        if hour not in self.price_history:
            self.price_history[hour] = []
        
        # Keep a limited history (last 10 prices)
        if len(self.price_history[hour]) >= 10:
            self.price_history[hour].pop(0)
        
        self.price_history[hour].append(price)
    
    def _update_success_rate(self, success):
        """Track success rate of market orders and adjust strategy."""
        self.attempt_count += 1
        if success:
            self.success_count += 1
        
        # Calculate success rate and adjust price factor
        if self.attempt_count >= 10:
            success_rate = self.success_count / self.attempt_count
            
            # Adjust price factor based on success rate
            if success_rate < 0.5:
                # Low success rate: make bids more attractive, asks less attractive
                self.price_adjustment_factor = min(1.5, self.price_adjustment_factor * 1.05)
            elif success_rate > 0.8:
                # High success rate: we can be more aggressive with prices
                self.price_adjustment_factor = max(0.8, self.price_adjustment_factor * 0.98)
            
            # Reset counters periodically
            if self.attempt_count >= 20:
                self.attempt_count = 10
                self.success_count = int(success_rate * 10)
    
    async def _process_last_settlement(self, actions):
        """Process settlement results from the last period to inform future decisions."""
        last_settle = self.__participant['timing']['last_settle']
        
        if last_settle in self.action_scenario_history:
            last_settle_info = await self.__participant['ledger'].get_settled_info(last_settle)
            
            # Record settlement data for analysis
            if 'bids' in last_settle_info and last_settle_info['bids']['quantity'] > 0:
                self._update_price_history(last_settle[1], last_settle_info['bids']['price'])
                self._update_success_rate(True)
            elif 'asks' in last_settle_info and last_settle_info['asks']['quantity'] > 0:
                self._update_price_history(last_settle[1], last_settle_info['asks']['price'])
                self._update_success_rate(True)
            else:
                # No settlement occurred
                self._update_success_rate(False)
            
            # Handle battery actions based on settlement results
            scenario = self.action_scenario_history[last_settle]['scenario']
            
            if scenario == 'buy_and_charge':
                # We bought energy to use and charge
                settled_quantity = last_settle_info.get('bids', {}).get('quantity', 0)
                residual_load = self.action_scenario_history[last_settle]['residual_load']
                max_charge = self.action_scenario_history[last_settle]['max_charge']
                
                # First satisfy load, then charge with remaining energy
                charge_amount = max(0, min(max_charge, settled_quantity - residual_load))
                if charge_amount > 0:
                    actions['bess'] = {str(last_settle): charge_amount}
            
            elif scenario == 'discharge_and_sell':
                # We discharged battery and sold energy
                settled_quantity = last_settle_info.get('asks', {}).get('quantity', 0)
                max_discharge = self.action_scenario_history[last_settle]['max_discharge']
                
                # Only discharge what was actually sold
                if settled_quantity > 0:
                    actions['bess'] = {str(last_settle): -min(abs(max_discharge), settled_quantity)}
            
            elif scenario == 'discharge_for_load':
                # We used battery to satisfy local load
                residual_load = self.action_scenario_history[last_settle]['residual_load']
                max_discharge = self.action_scenario_history[last_settle]['max_discharge']
                
                # Discharge to cover load
                actions['bess'] = {str(last_settle): -min(abs(max_discharge), residual_load)}
            
            elif scenario == 'charge_from_generation':
                # We charged battery from local generation
                residual_gen = self.action_scenario_history[last_settle]['residual_gen']
                max_charge = self.action_scenario_history[last_settle]['max_charge']
                
                # Charge from generation
                actions['bess'] = {str(last_settle): min(max_charge, residual_gen)}
        
        # Clean up old history entries
        stale_round = self.__participant['timing']['stale_round']
        self.action_scenario_history.pop(stale_round, None)
        
        return actions
    
    async def act(self, **kwargs):
        """Main decision function creating market and battery actions."""
        actions = {}
        
        # Get timing information
        last_settle = self.__participant['timing']['last_settle']
        next_settle = self.__participant['timing']['next_settle']
        timezone = self.__participant['timing']['timezone']
        next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)
        
        # Process previous settlement results
        actions = await self._process_last_settlement(actions)
        
        # Get current state information
        generation, load = await self.__participant['read_profile'](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load
        
        # Battery information
        if 'storage' in self.__participant:
            storage_info = self.__participant['storage']['info']()
            current_soc = storage_info['state_of_charge'] / 100.0  # Convert percentage to decimal
            capacity = storage_info['capacity']
            
            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            max_charge = storage_schedule[next_settle]['energy_potential'][1]
            max_discharge = storage_schedule[next_settle]['energy_potential'][0]  # Negative value
            
            # Calculate energy value for decision making
            energy_value = self._calculate_energy_value(next_settle[1], current_soc)
            
            # Determine optimal battery action for next settlement
            battery_action = self._determine_battery_strategy(
                current_soc, next_settle[1], residual_load, residual_gen, max_charge, max_discharge)
            
            # Process local energy balance after battery action
            if battery_action > 0:  # Charging
                # If charging, reduce excess generation or increase load
                residual_gen = max(0, residual_gen - battery_action)
                residual_load = max(0, residual_load + battery_action)
            elif battery_action < 0:  # Discharging
                # If discharging, increase excess generation or reduce load
                discharge_amount = abs(battery_action)
                residual_gen = max(0, residual_gen + discharge_amount)
                residual_load = max(0, residual_load - discharge_amount)
            
            # Decide market actions based on remaining energy balance
            if residual_load > 0:
                # Need to buy energy
                bid_price = self._determine_prices(energy_value, True)
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': residual_load,
                        'price': bid_price
                    }
                }
                
                # Record our strategy for next round processing
                self.action_scenario_history[next_settle] = {
                    'scenario': 'buy_and_charge',
                    'residual_load': residual_load,
                    'residual_gen': residual_gen,
                    'max_charge': max_charge,
                    'max_discharge': max_discharge,
                    'battery_action': battery_action,
                    'energy_value': energy_value
                }
            
            elif residual_gen > 0:
                # Have excess energy to sell
                ask_price = self._determine_prices(energy_value, False)
                actions['asks'] = {
                    'solar': {  # Assuming excess is from solar
                        str(next_settle): {
                            'quantity': residual_gen,
                            'price': ask_price
                        }
                    }
                }
                
                # Record our strategy for next round processing
                self.action_scenario_history[next_settle] = {
                    'scenario': 'discharge_and_sell',
                    'residual_load': residual_load,
                    'residual_gen': residual_gen,
                    'max_charge': max_charge,
                    'max_discharge': max_discharge,
                    'battery_action': battery_action,
                    'energy_value': energy_value
                }
            
            else:
                # Balanced load and generation after battery action
                if battery_action < 0:
                    # Record that we're discharging for load
                    self.action_scenario_history[next_settle] = {
                        'scenario': 'discharge_for_load',
                        'residual_load': load - generation,  # Original residual
                        'residual_gen': 0,
                        'max_charge': max_charge,
                        'max_discharge': max_discharge,
                        'battery_action': battery_action,
                        'energy_value': energy_value
                    }
                elif battery_action > 0:
                    # Record that we're charging from generation
                    self.action_scenario_history[next_settle] = {
                        'scenario': 'charge_from_generation',
                        'residual_load': 0,
                        'residual_gen': -residual_load,  # Original generation
                        'max_charge': max_charge,
                        'max_discharge': max_discharge,
                        'battery_action': battery_action,
                        'energy_value': energy_value
                    }
            
            # Add battery action if non-zero
            if battery_action != 0:
                # For next settlement period
                actions['bess'] = {str(next_settle): battery_action}
        
        else:
            # No battery - simpler logic
            if residual_load > 0:
                # Need to buy energy
                bid_price = self._determine_prices(0.15, True)  # Default value without battery
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': residual_load,
                        'price': bid_price
                    }
                }
            elif residual_gen > 0:
                # Have excess energy to sell
                ask_price = self._determine_prices(0.15, False)  # Default value without battery
                actions['asks'] = {
                    'solar': {
                        str(next_settle): {
                            'quantity': residual_gen,
                            'price': ask_price
                        }
                    }
                }
        
        return actions
    
    async def step(self):
        """Process the next step in the simulation."""
        next_actions = await self.act()
        return next_actions
    
    async def reset(self, **kwargs):
        """Reset the trader state for a new simulation.
        
        Preserves price history and learned market knowledge between episodes
        while clearing temporary action state.
        """
        # Clear temporary state for the new episode
        self.settlement_history.clear()
        self.action_scenario_history.clear()
        
        # Reset success tracking for the new episode
        self.success_count = 0
        self.attempt_count = 0
        
        # Importantly, we DO NOT clear price_history or price_adjustment_factor
        # This allows the agent to retain market knowledge between episodes
        
        return True 