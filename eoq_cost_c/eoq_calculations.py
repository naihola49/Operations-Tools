"""
Economic Order Quantity (EOQ) Calculations

Core module for EOQ analysis including:
- Basic EOQ calculation
- Total cost analysis
- Sensitivity analysis
- Scenario comparison
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_eoq(
    demand: float,
    setup_cost: float,
    holding_cost_per_unit: float
) -> float:
    """
    Calculate the Economic Order Quantity (EOQ).
    
    Formula: Q* = sqrt(2DS/H)
    where:
        D = Annual demand
        S = Setup/ordering cost per order
        H = Holding cost per unit per year
    
    Args:
        demand: Annual demand (units/year)
        setup_cost: Setup/ordering cost per order ($)
        holding_cost_per_unit: Holding cost per unit per year ($/unit/year)
    
    Returns:
        Optimal order quantity (EOQ)
    """
    if holding_cost_per_unit <= 0:
        raise ValueError("Holding cost must be positive")
    if setup_cost < 0:
        raise ValueError("Setup cost cannot be negative")
    if demand < 0:
        raise ValueError("Demand cannot be negative")
    
    eoq = np.sqrt(2 * demand * setup_cost / holding_cost_per_unit)
    return eoq


def calculate_total_cost(
    order_quantity: float,
    demand: float,
    setup_cost: float,
    holding_cost_per_unit: float,
    unit_cost: float = 0.0,
    discount_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate total cost for a given order quantity.
    
    Total Cost = (D/Q) * S + (Q/2) * H + D * C * (1 - discount)
    where:
        (D/Q) * S = Annual setup cost
        (Q/2) * H = Annual holding cost
        D * C * (1 - discount) = Annual purchase cost
    
    Args:
        order_quantity: Order quantity (units)
        demand: Annual demand (units/year)
        setup_cost: Setup/ordering cost per order ($)
        holding_cost_per_unit: Holding cost per unit per year ($/unit/year)
        unit_cost: Cost per unit ($/unit), default 0 (excluded from calculation)
        discount_rate: Discount rate (0-1), default 0
    
    Returns:
        Dictionary with cost breakdown:
        - total_cost: Total annual cost
        - setup_cost_annual: Annual setup cost
        - holding_cost_annual: Annual holding cost
        - purchase_cost_annual: Annual purchase cost (if unit_cost provided)
        - number_of_orders: Number of orders per year
        - average_inventory: Average inventory level
    """
    if order_quantity <= 0:
        raise ValueError("Order quantity must be positive")
    
    # Number of orders per year
    number_of_orders = demand / order_quantity
    
    # Annual setup cost
    setup_cost_annual = number_of_orders * setup_cost
    
    # Annual holding cost (average inventory = Q/2) -> sawtooth pattern due to batching 
    average_inventory = order_quantity / 2
    holding_cost_annual = average_inventory * holding_cost_per_unit
    
    # Annual purchase cost (if unit cost provided)
    purchase_cost_annual = 0.0
    if unit_cost > 0:
        purchase_cost_annual = demand * unit_cost * (1 - discount_rate)
    
    # Total cost
    total_cost = setup_cost_annual + holding_cost_annual + purchase_cost_annual
    
    return {
        'total_cost': total_cost,
        'setup_cost_annual': setup_cost_annual,
        'holding_cost_annual': holding_cost_annual,
        'purchase_cost_annual': purchase_cost_annual,
        'number_of_orders': number_of_orders,
        'average_inventory': average_inventory,
        'order_quantity': order_quantity
    }


def calculate_eoq_with_costs(
    demand: float,
    setup_cost: float,
    holding_cost_per_unit: float,
    unit_cost: float = 0.0,
    discount_rate: float = 0.0
) -> Dict:
    """
    Calculate EOQ and associated costs.
    
    Args:
        demand: Annual demand (units/year)
        setup_cost: Setup/ordering cost per order ($)
        holding_cost_per_unit: Holding cost per unit per year ($/unit/year)
        unit_cost: Cost per unit ($/unit), default 0
        discount_rate: Discount rate (0-1), default 0
    
    Returns:
        Dictionary with EOQ and cost breakdown
    """
    eoq = calculate_eoq(demand, setup_cost, holding_cost_per_unit)
    costs = calculate_total_cost(
        eoq, demand, setup_cost, holding_cost_per_unit, unit_cost, discount_rate
    )
    
    return {
        'eoq': eoq,
        **costs
    }


def sensitivity_analysis(
    base_demand: float,
    base_setup_cost: float,
    base_holding_cost: float,
    variation_range: float = 0.5
) -> pd.DataFrame:
    """
    Perform sensitivity analysis on EOQ by varying parameters.
    
    Args:
        base_demand: Base annual demand
        base_setup_cost: Base setup cost
        base_holding_cost: Base holding cost per unit
        variation_range: Range of variation (e.g., 0.5 = ±50%)
    
    Returns:
        DataFrame with sensitivity results
    """
    variations = np.linspace(1 - variation_range, 1 + variation_range, 21)
    
    results = []
    
    # Vary demand
    for var in variations:
        demand = base_demand * var
        eoq = calculate_eoq(demand, base_setup_cost, base_holding_cost)
        results.append({
            'parameter': 'Demand',
            'variation': var,
            'value': demand,
            'eoq': eoq,
            'eoq_change_pct': (eoq / calculate_eoq(base_demand, base_setup_cost, base_holding_cost) - 1) * 100
        })
    
    # Vary setup cost
    for var in variations:
        setup_cost = base_setup_cost * var
        eoq = calculate_eoq(base_demand, setup_cost, base_holding_cost)
        results.append({
            'parameter': 'Setup Cost',
            'variation': var,
            'value': setup_cost,
            'eoq': eoq,
            'eoq_change_pct': (eoq / calculate_eoq(base_demand, base_setup_cost, base_holding_cost) - 1) * 100
        })
    
    # Vary holding cost
    for var in variations:
        holding_cost = base_holding_cost * var
        eoq = calculate_eoq(base_demand, base_setup_cost, holding_cost)
        results.append({
            'parameter': 'Holding Cost',
            'variation': var,
            'value': holding_cost,
            'eoq': eoq,
            'eoq_change_pct': (eoq / calculate_eoq(base_demand, base_setup_cost, base_holding_cost) - 1) * 100
        })
    
    return pd.DataFrame(results)


def compare_scenarios(
    scenarios: List[Dict],
    demand: float
) -> pd.DataFrame:
    """
    Compare multiple EOQ scenarios.
    
    Args:
        scenarios: List of scenario dictionaries, each with:
            - name: Scenario name
            - setup_cost: Setup cost
            - holding_cost_per_unit: Holding cost
            - unit_cost: Unit cost (optional)
            - discount_rate: Discount rate (optional)
            - order_quantity: Custom order quantity (optional, uses EOQ if not provided)
        demand: Annual demand
    
    Returns:
        DataFrame comparing all scenarios
    """
    results = []
    
    for scenario in scenarios:
        name = scenario.get('name', 'Unnamed')
        setup_cost = scenario['setup_cost']
        holding_cost = scenario['holding_cost_per_unit']
        unit_cost = scenario.get('unit_cost', 0.0)
        discount_rate = scenario.get('discount_rate', 0.0)
        custom_qty = scenario.get('order_quantity', None)
        
        if custom_qty is not None:
            # Use custom order quantity
            order_qty = custom_qty
            costs = calculate_total_cost(
                order_qty, demand, setup_cost, holding_cost, unit_cost, discount_rate
            )
        else:
            # Calculate EOQ
            result = calculate_eoq_with_costs(
                demand, setup_cost, holding_cost, unit_cost, discount_rate
            )
            order_qty = result['eoq']
            costs = result
        
        results.append({
            'scenario': name,
            'order_quantity': order_qty,
            'total_cost': costs['total_cost'],
            'setup_cost_annual': costs['setup_cost_annual'],
            'holding_cost_annual': costs['holding_cost_annual'],
            'purchase_cost_annual': costs['purchase_cost_annual'],
            'number_of_orders': costs['number_of_orders'],
            'average_inventory': costs['average_inventory']
        })
    
    return pd.DataFrame(results)


def generate_cost_curve_data(
    demand: float,
    setup_cost: float,
    holding_cost_per_unit: float,
    unit_cost: float = 0.0,
    discount_rate: float = 0.0,
    qty_range: Optional[Tuple[float, float]] = None,
    num_points: int = 100
) -> pd.DataFrame:
    """
    Generate data for cost curve visualization.
    
    Args:
        demand: Annual demand
        setup_cost: Setup cost
        holding_cost_per_unit: Holding cost per unit
        unit_cost: Unit cost (optional)
        discount_rate: Discount rate (optional)
        qty_range: (min_qty, max_qty) tuple, auto-calculated if None
        num_points: Number of points to generate
    
    Returns:
        DataFrame with order quantities and associated costs
    """
    eoq = calculate_eoq(demand, setup_cost, holding_cost_per_unit)
    
    if qty_range is None:
        # Generate range around EOQ (0.1x to 5x EOQ)
        min_qty = max(1, eoq * 0.1)
        max_qty = eoq * 5
    else:
        min_qty, max_qty = qty_range
    
    quantities = np.linspace(min_qty, max_qty, num_points)
    
    data = []
    for qty in quantities:
        costs = calculate_total_cost(
            qty, demand, setup_cost, holding_cost_per_unit, unit_cost, discount_rate
        )
        data.append({
            'order_quantity': qty,
            'total_cost': costs['total_cost'],
            'setup_cost': costs['setup_cost_annual'],
            'holding_cost': costs['holding_cost_annual'],
            'purchase_cost': costs['purchase_cost_annual']
        })
    
    return pd.DataFrame(data)
