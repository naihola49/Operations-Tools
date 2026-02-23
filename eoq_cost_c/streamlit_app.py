"""
Streamlit application for EOQ Analysis Tool
"""

import streamlit as st
import pandas as pd
from eoq_calculations import (
    calculate_eoq,
    calculate_eoq_with_costs,
    sensitivity_analysis,
    compare_scenarios,
    generate_cost_curve_data
)
from visualization import (
    plot_cost_curves,
    plot_sensitivity_analysis,
    plot_scenario_comparison,
    plot_cost_breakdown
)


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="EOQ Analysis Tool",
        layout="wide"
    )
    
    st.title("Economic Order Quantity (EOQ) Analysis Tool")
    st.markdown("""
    Economic Order Quantity models.
    Dynamically adjust setup costs, holding costs, and demand to see how they affect optimal batch sizes.
    """)
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Parameters")
        
        # Basic parameters
        st.subheader("Basic Parameters")
        demand = st.number_input(
            "Annual Demand (units/year)",
            min_value=1.0,
            value=10000.0,
            step=100.0,
            help="Total annual demand for the product"
        )
        
        setup_cost = st.number_input(
            "Setup/Ordering Cost ($)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Cost per order or setup"
        )
        
        holding_cost_per_unit = st.number_input(
            "Holding Cost per Unit ($/unit/year)",
            min_value=0.01,
            value=2.0,
            step=0.1,
            help="Annual cost to hold one unit in inventory"
        )
        
        st.divider()
        
        # Optional parameters
        st.subheader("Optional Parameters")
        include_purchase_cost = st.checkbox("Include Purchase Cost", value=False)
        
        unit_cost = 0.0
        discount_rate = 0.0
        
        if include_purchase_cost:
            unit_cost = st.number_input(
                "Unit Cost ($/unit)",
                min_value=0.0,
                value=10.0,
                step=0.5
            )
            
            discount_rate = st.number_input(
                "Discount Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0
            ) / 100.0
        
        st.divider()
        
        # Analysis options
        st.subheader("Analysis Options")
        show_sensitivity = st.checkbox("Show Sensitivity Analysis", value=True)
        show_cost_curve = st.checkbox("Show Cost Curves", value=True)
    
    # Calculate EOQ
    try:
        eoq_result = calculate_eoq_with_costs(
            demand, setup_cost, holding_cost_per_unit, unit_cost, discount_rate
        )
        eoq = eoq_result['eoq']
        costs = eoq_result
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Optimal EOQ",
                f"{eoq:,.0f}",
                help="Economic Order Quantity"
            )
        
        with col2:
            st.metric(
                "Total Annual Cost",
                f"${costs['total_cost']:,.2f}"
            )
        
        with col3:
            st.metric(
                "Orders per Year",
                f"{costs['number_of_orders']:.2f}"
            )
        
        with col4:
            st.metric(
                "Avg Inventory",
                f"{costs['average_inventory']:,.0f}"
            )
        
        # Cost breakdown
        st.subheader("Cost Breakdown")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cost_df = pd.DataFrame({
                'Cost Type': ['Setup Cost', 'Holding Cost', 'Purchase Cost'],
                'Annual Cost ($)': [
                    costs['setup_cost_annual'],
                    costs['holding_cost_annual'],
                    costs['purchase_cost_annual']
                ]
            })
            st.dataframe(cost_df, width='stretch', hide_index=True)
        
        with col2:
            fig_pie = plot_cost_breakdown(costs)
            st.plotly_chart(fig_pie, width='stretch')
        
        # Cost curves
        if show_cost_curve:
            st.subheader("Cost Curves")
            cost_curve_data = generate_cost_curve_data(
                demand, setup_cost, holding_cost_per_unit, unit_cost, discount_rate
            )
            fig_curves = plot_cost_curves(
                cost_curve_data, eoq, costs['total_cost']
            )
            st.plotly_chart(fig_curves, width='stretch')
        
        # Sensitivity analysis
        if show_sensitivity:
            st.subheader("Sensitivity Analysis")
            st.markdown("""
            How does EOQ change when parameters vary? This shows the robustness of the model.
            """)
            
            sens_data = sensitivity_analysis(
                demand, setup_cost, holding_cost_per_unit, variation_range=0.5
            )
            fig_sens = plot_sensitivity_analysis(sens_data)
            st.plotly_chart(fig_sens, width='stretch')
        
        # Scenario testing
        st.subheader("Scenario Testing")
        st.markdown("""
        Compare different scenarios by adjusting parameters below.
        """)
        
        with st.expander("Add Scenario", expanded=False):
            scenario_name = st.text_input("Scenario Name", value="Custom Scenario")
            scenario_setup_cost = st.number_input(
                "Setup Cost ($)",
                min_value=0.0,
                value=float(setup_cost),
                step=10.0,
                key="scenario_setup"
            )
            scenario_holding_cost = st.number_input(
                "Holding Cost ($/unit/year)",
                min_value=0.01,
                value=float(holding_cost_per_unit),
                step=0.1,
                key="scenario_holding"
            )
            use_custom_qty = st.checkbox("Use Custom Order Quantity", key="custom_qty")
            custom_qty = None
            if use_custom_qty:
                custom_qty = st.number_input(
                    "Order Quantity",
                    min_value=1.0,
                    value=float(eoq),
                    step=10.0,
                    key="scenario_qty"
                )
            
            add_scenario = st.button("Add Scenario", key="add_scenario")
        
        # Initialize scenarios in session state
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = [
                {
                    'name': 'Base Case',
                    'setup_cost': setup_cost,
                    'holding_cost_per_unit': holding_cost_per_unit,
                    'unit_cost': unit_cost,
                    'discount_rate': discount_rate
                }
            ]
        
        # Add new scenario
        if add_scenario:
            new_scenario = {
                'name': scenario_name,
                'setup_cost': scenario_setup_cost,
                'holding_cost_per_unit': scenario_holding_cost,
                'unit_cost': unit_cost,
                'discount_rate': discount_rate
            }
            if use_custom_qty:
                new_scenario['order_quantity'] = custom_qty
            
            # Check if scenario name already exists
            if any(s['name'] == scenario_name for s in st.session_state.scenarios):
                st.warning(f"Scenario '{scenario_name}' already exists. Please use a different name.")
            else:
                st.session_state.scenarios.append(new_scenario)
                st.success(f"Added scenario: {scenario_name}")
                st.rerun()
        
        # Display and compare scenarios
        if len(st.session_state.scenarios) > 1:
            scenario_comparison = compare_scenarios(
                st.session_state.scenarios,
                demand
            )
            
            st.subheader("Scenario Comparison")
            fig_comp = plot_scenario_comparison(scenario_comparison)
            st.plotly_chart(fig_comp, width='stretch')
            
            st.subheader("Detailed Comparison Table")
            st.dataframe(scenario_comparison, width='stretch', hide_index=True)
            
            # Clear scenarios button
            if st.button("Clear All Scenarios", type="secondary"):
                st.session_state.scenarios = [
                    {
                        'name': 'Base Case',
                        'setup_cost': setup_cost,
                        'holding_cost_per_unit': holding_cost_per_unit,
                        'unit_cost': unit_cost,
                        'discount_rate': discount_rate
                    }
                ]
                st.rerun()
        else:
            st.info("👆 Add scenarios above to compare different parameter configurations.")
        
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
    
    # Footer
    st.divider()
    st.markdown("""
    ### About EOQ Model
    
    The Economic Order Quantity (EOQ) model minimizes total inventory costs by balancing:
    - **Setup Costs**: Higher order quantities reduce setup frequency
    - **Holding Costs**: Lower order quantities reduce inventory carrying costs
    
    **Key Insights**:
    - EOQ is robust to parameter estimation errors (square root relationship)
    - Business constraints (discounts, safety stock) may justify deviations from EOQ
    - High setup costs justify larger batches, but increase inventory and cycle time
    """)


if __name__ == "__main__":
    main()
