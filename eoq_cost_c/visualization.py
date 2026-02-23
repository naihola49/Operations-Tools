"""
Visualization functions for EOQ analysis
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Optional


def plot_cost_curves(
    cost_data: pd.DataFrame,
    eoq: float,
    eoq_cost: float,
    title: str = "Total Cost vs Order Quantity"
) -> go.Figure:
    """
    Plot cost curves showing total cost, setup cost, and holding cost.
    
    Args:
        cost_data: DataFrame with order_quantity, total_cost, setup_cost, holding_cost
        eoq: Optimal EOQ value
        eoq_cost: Total cost at EOQ
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Total cost curve
    fig.add_trace(go.Scatter(
        x=cost_data['order_quantity'],
        y=cost_data['total_cost'],
        mode='lines',
        name='Total Cost',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Setup cost curve
    fig.add_trace(go.Scatter(
        x=cost_data['order_quantity'],
        y=cost_data['setup_cost'],
        mode='lines',
        name='Setup Cost',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Holding cost curve
    fig.add_trace(go.Scatter(
        x=cost_data['order_quantity'],
        y=cost_data['holding_cost'],
        mode='lines',
        name='Holding Cost',
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))
    
    # Mark EOQ point
    fig.add_trace(go.Scatter(
        x=[eoq],
        y=[eoq_cost],
        mode='markers',
        name=f'EOQ = {eoq:.0f}',
        marker=dict(
            size=15,
            color='red',
            symbol='star',
            line=dict(width=2, color='darkred')
        ),
        hovertemplate=f'EOQ: {eoq:.0f}<br>Cost: ${eoq_cost:,.2f}<extra></extra>'
    ))
    
    # Add vertical line at EOQ
    fig.add_vline(
        x=eoq,
        line_dash="dot",
        line_color="red",
        annotation_text=f"EOQ = {eoq:.0f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Order Quantity (units)",
        yaxis_title="Annual Cost ($)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_sensitivity_analysis(
    sensitivity_data: pd.DataFrame,
    title: str = "EOQ Sensitivity Analysis"
) -> go.Figure:
    """
    Plot sensitivity analysis showing how EOQ changes with parameter variations.
    
    Args:
        sensitivity_data: DataFrame from sensitivity_analysis function
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    parameters = sensitivity_data['parameter'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, param in enumerate(parameters):
        param_data = sensitivity_data[sensitivity_data['parameter'] == param]
        fig.add_trace(go.Scatter(
            x=param_data['variation'] * 100,  # Convert to percentage
            y=param_data['eoq'],
            mode='lines+markers',
            name=param,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    # Add vertical line at 100% (base value)
    fig.add_vline(
        x=100,
        line_dash="dot",
        line_color="gray",
        annotation_text="Base Value",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Parameter Variation (%)",
        yaxis_title="EOQ (units)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_scenario_comparison(
    scenario_data: pd.DataFrame,
    title: str = "Scenario Comparison"
) -> go.Figure:
    """
    Create bar chart comparing multiple scenarios.
    
    Args:
        scenario_data: DataFrame from compare_scenarios function
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Cost', 'Order Quantity', 'Number of Orders', 'Average Inventory'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    scenarios = scenario_data['scenario'].tolist()
    
    # Total Cost
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=scenario_data['total_cost'],
            name='Total Cost',
            marker_color='#1f77b4',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Order Quantity
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=scenario_data['order_quantity'],
            name='Order Quantity',
            marker_color='#ff7f0e',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Number of Orders
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=scenario_data['number_of_orders'],
            name='Number of Orders',
            marker_color='#2ca02c',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Average Inventory
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=scenario_data['average_inventory'],
            name='Average Inventory',
            marker_color='#d62728',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Scenario", row=1, col=1)
    fig.update_xaxes(title_text="Scenario", row=1, col=2)
    fig.update_xaxes(title_text="Scenario", row=2, col=1)
    fig.update_xaxes(title_text="Scenario", row=2, col=2)
    
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_yaxes(title_text="Quantity (units)", row=1, col=2)
    fig.update_yaxes(title_text="Orders/year", row=2, col=1)
    fig.update_yaxes(title_text="Units", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        height=700,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_cost_breakdown(
    costs: Dict[str, float],
    title: str = "Cost Breakdown"
) -> go.Figure:
    """
    Create pie chart showing cost breakdown.
    
    Args:
        costs: Dictionary with cost components
        title: Plot title
    
    Returns:
        Plotly figure
    """
    labels = []
    values = []
    
    if costs.get('setup_cost_annual', 0) > 0:
        labels.append('Setup Cost')
        values.append(costs['setup_cost_annual'])
    
    if costs.get('holding_cost_annual', 0) > 0:
        labels.append('Holding Cost')
        values.append(costs['holding_cost_annual'])
    
    if costs.get('purchase_cost_annual', 0) > 0:
        labels.append('Purchase Cost')
        values.append(costs['purchase_cost_annual'])
    
    if not labels:
        # Fallback if no costs
        labels = ['No Costs']
        values = [1]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )
    
    return fig
