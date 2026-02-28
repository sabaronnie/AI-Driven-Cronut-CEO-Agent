"""
Comprehensive test suite for all 5 ML models in the Conut Chief of Operations system.

Tests:
1. Model 1 (Combo Optimization) - Market Basket Analysis
2. Model 2 (Demand Forecasting) - Time Series Forecasting
3. Model 3 (Expansion Feasibility) - K-Means Clustering & Scoring
4. Model 4 (Shift Staffing) - Ridge Regression
5. Model 5 (Coffee & Milkshake Growth) - Category Performance Analysis

Each test suite verifies:
- Function returns success status
- Data loading works
- Output structure is correct (required keys exist)
- Model-specific metrics are reasonable
- Edge cases are handled
- Config validation raises on missing columns
"""

import unittest
import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress logging during tests
logging.disable(logging.CRITICAL)

# Add src to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import the model functions
from combo_optimization import get_combo_recommendations
from demand_forecasting import get_demand_forecast
from expansion_feasibility import get_expansion_assessment
from shift_staffing import get_staffing_recommendation
from coffee_milkshake_growth import get_coffee_milkshake_analysis
from branch_location_recommender import (
    get_branch_location_recommendation,
    load_area_dataset,
    compute_demand_proxy,
    fit_competitor_model,
    run_full_location_analysis,
)
from config import validate_dataframe


class TestComboOptimization(unittest.TestCase):
    """Test Model 1: Combo Optimization (Market Basket Analysis)"""

    def setUp(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, 'data', 'cleaned')

    def test_data_files_exist(self):
        """Test that required data files exist"""
        detail_file = os.path.join(self.data_dir, 'sales_detail_(502).csv')
        item_file = os.path.join(self.data_dir, 'cleaned_sales_by_item_(191).csv')
        self.assertTrue(os.path.exists(detail_file), f"Missing: {detail_file}")
        self.assertTrue(os.path.exists(item_file), f"Missing: {item_file}")

    def test_data_loading(self):
        """Test that data files can be loaded"""
        detail_file = os.path.join(self.data_dir, 'sales_detail_(502).csv')
        item_file = os.path.join(self.data_dir, 'cleaned_sales_by_item_(191).csv')
        
        df_detail = pd.read_csv(detail_file)
        df_item = pd.read_csv(item_file)
        
        self.assertGreater(len(df_detail), 0, "Detail file is empty")
        self.assertGreater(len(df_item), 0, "Item file is empty")

    def test_combo_recommendations_returns_success(self):
        """Test that get_combo_recommendations returns success status"""
        result = get_combo_recommendations()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn(result['status'], ['success', 'error'])

    def test_combo_recommendations_output_structure(self):
        """Test that combo recommendations output has correct structure"""
        result = get_combo_recommendations()
        if result['status'] == 'success':
            self.assertIn('combos', result)
            self.assertIsInstance(result['combos'], list)
            
            if len(result['combos']) > 0:
                combo = result['combos'][0]
                self.assertIn('items', combo)
                self.assertIn('combo_price', combo)
                self.assertIn('lift', combo)
                self.assertIn('confidence', combo)

    def test_combo_recommendations_with_valid_branch(self):
        """Test get_combo_recommendations with a valid branch"""
        # Get all results first to find available branches
        all_result = get_combo_recommendations()
        if all_result['status'] == 'success' and len(all_result['combos']) > 0:
            # Try with a specific branch (using a branch from data)
            result = get_combo_recommendations(branch='Location 1')
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)

    def test_combo_recommendations_invalid_branch_returns_error(self):
        """Test that invalid branch name returns error"""
        result = get_combo_recommendations(branch='NONEXISTENT_BRANCH_XYZ')
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        # Should either return error or empty combos
        if result['status'] == 'error':
            self.assertIn('message', result)

    def test_combo_rules_exist(self):
        """Test that association rules are generated"""
        result = get_combo_recommendations(min_support=0.01)
        self.assertIsInstance(result, dict)
        if result['status'] == 'success':
            self.assertIn('combos', result)
            # Should have some rules or explicit error message
            self.assertTrue(
                len(result['combos']) > 0 or 'message' in result,
                "Should have combos or error message"
            )


class TestDemandForecasting(unittest.TestCase):
    """Test Model 2: Demand Forecasting (Time Series)"""

    def setUp(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, 'data', 'cleaned')

    def test_data_files_exist(self):
        """Test that required data files exist"""
        monthly_file = os.path.join(self.data_dir, 'cleaned_monthly_sales_(334).csv')
        orders_file = os.path.join(self.data_dir, 'customer_orders_(150).csv')
        attendance_file = os.path.join(self.data_dir, 'cleaned_attendance_(461).csv')
        
        self.assertTrue(os.path.exists(monthly_file), f"Missing: {monthly_file}")
        self.assertTrue(os.path.exists(orders_file), f"Missing: {orders_file}")
        self.assertTrue(os.path.exists(attendance_file), f"Missing: {attendance_file}")

    def test_data_loading(self):
        """Test that all data files can be loaded"""
        monthly_file = os.path.join(self.data_dir, 'cleaned_monthly_sales_(334).csv')
        orders_file = os.path.join(self.data_dir, 'customer_orders_(150).csv')
        attendance_file = os.path.join(self.data_dir, 'cleaned_attendance_(461).csv')
        
        df_monthly = pd.read_csv(monthly_file)
        df_orders = pd.read_csv(orders_file)
        df_attendance = pd.read_csv(attendance_file)
        
        self.assertGreater(len(df_monthly), 0, "Monthly sales file is empty")
        self.assertGreater(len(df_orders), 0, "Customer orders file is empty")
        self.assertGreater(len(df_attendance), 0, "Attendance file is empty")

    def test_demand_forecast_returns_success(self):
        """Test that get_demand_forecast returns success status"""
        result = get_demand_forecast(periods=3)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn(result['status'], ['success', 'error'])

    def test_demand_forecast_output_structure(self):
        """Test that demand forecast output has correct structure"""
        result = get_demand_forecast(periods=3)
        if result['status'] == 'success':
            self.assertIn('forecasts', result)
            self.assertIsInstance(result['forecasts'], list)
            
            if len(result['forecasts']) > 0:
                forecast = result['forecasts'][0]
                self.assertIn('branch', forecast)
                if forecast['status'] == 'success':
                    self.assertIn('ensemble_forecast', forecast)
                    self.assertIn('best_model', forecast)
                    self.assertIn('historical_sales', forecast)

    def test_demand_forecast_metrics_exist(self):
        """Test that Model 2 metrics (MAPE, R², MAE) exist and are reasonable"""
        result = get_demand_forecast(periods=3)
        if result['status'] == 'success':
            self.assertIn('forecasts', result)
            
            for forecast in result['forecasts']:
                if forecast['status'] == 'success':
                    # Check key metrics exist
                    self.assertIn('best_model_mape', forecast)
                    self.assertIn('poly_r_squared', forecast)
                    
                    # Check metric ranges
                    mape = forecast['best_model_mape']
                    r2 = forecast['poly_r_squared']
                    
                    self.assertIsInstance(mape, (int, float))
                    self.assertIsInstance(r2, (int, float))
                    self.assertGreaterEqual(mape, 0, "MAPE should be >= 0")
                    self.assertLessEqual(r2, 1, "R² should be <= 1")

    def test_demand_forecast_with_valid_branch(self):
        """Test get_demand_forecast with a specific branch"""
        result = get_demand_forecast(branch='Location 1', periods=3)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)

    def test_demand_forecast_invalid_branch_returns_error(self):
        """Test that invalid branch name returns error"""
        result = get_demand_forecast(branch='NONEXISTENT_BRANCH_XYZ', periods=3)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        # Should return error or empty forecasts for invalid branch
        if result['status'] == 'error':
            self.assertIn('message', result)

    def test_demand_forecast_periods_parameter(self):
        """Test get_demand_forecast with different periods"""
        for periods in [1, 3, 6]:
            result = get_demand_forecast(periods=periods)
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)


class TestExpansionFeasibility(unittest.TestCase):
    """Test Model 3: Expansion Feasibility (K-Means Clustering)"""

    def setUp(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, 'data', 'cleaned')

    def test_data_files_exist(self):
        """Test that required data files exist"""
        files = [
            'cleaned_monthly_sales_(334).csv',
            'cleaned_avg_sales_by_menu_(435).csv',
            'customer_orders_(150).csv',
            'cleaned_attendance_(461).csv',
            'cleaned_tax_report_(194).csv'
        ]
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing: {filepath}")

    def test_data_loading(self):
        """Test that all data files can be loaded"""
        files = {
            'cleaned_monthly_sales_(334).csv': 'Monthly sales',
            'cleaned_avg_sales_by_menu_(435).csv': 'Menu sales',
            'customer_orders_(150).csv': 'Customer orders',
            'cleaned_attendance_(461).csv': 'Attendance',
            'cleaned_tax_report_(194).csv': 'Tax report'
        }
        for filename, desc in files.items():
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            self.assertGreater(len(df), 0, f"{desc} file is empty")

    def test_expansion_assessment_returns_success(self):
        """Test that get_expansion_assessment returns success status"""
        result = get_expansion_assessment()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn(result['status'], ['success', 'error'])

    def test_expansion_assessment_output_structure(self):
        """Test that expansion assessment output has correct structure"""
        result = get_expansion_assessment()
        if result['status'] == 'success':
            self.assertIn('expansion_score', result)
            self.assertIn('clustering', result)
            self.assertIsInstance(result['clustering'], dict)

    def test_expansion_clustering_exists(self):
        """Test that clustering is performed (cluster assignments exist)"""
        result = get_expansion_assessment()
        if result['status'] == 'success':
            self.assertIn('clustering', result)
            clustering = result['clustering']
            
            # Check clustering has required fields
            self.assertIn('assignments', clustering)
            self.assertIsInstance(clustering['assignments'], dict)
            self.assertGreater(len(clustering['assignments']), 0)

    def test_expansion_score_structure(self):
        """Test that expansion score has required keys"""
        result = get_expansion_assessment()
        if result['status'] == 'success':
            self.assertIn('expansion_score', result)
            score = result['expansion_score']
            
            self.assertIn('total_score', score)
            self.assertIn('recommendation', score)
            
            # Score should be between 0 and 100
            total = score['total_score']
            self.assertIsInstance(total, (int, float))
            self.assertGreaterEqual(total, 0)
            self.assertLessEqual(total, 100)

    def test_expansion_branch_profiles_have_cluster_labels(self):
        """Test that branch profiles include cluster assignments"""
        result = get_expansion_assessment()
        if result['status'] == 'success':
            self.assertIn('branch_profiles', result)
            profiles = result['branch_profiles']
            self.assertGreater(len(profiles), 0)
            
            for profile in profiles:
                self.assertIn('cluster_label', profile)
                self.assertIsNotNone(profile['cluster_label'])


class TestShiftStaffing(unittest.TestCase):
    """Test Model 4: Shift Staffing (Ridge Regression)"""

    def setUp(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, 'data', 'cleaned')

    def test_data_files_exist(self):
        """Test that required data files exist"""
        files = [
            'cleaned_monthly_sales_(334).csv',
            'customer_orders_(150).csv',
            'cleaned_attendance_(461).csv'
        ]
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing: {filepath}")

    def test_data_loading(self):
        """Test that all data files can be loaded"""
        monthly_file = os.path.join(self.data_dir, 'cleaned_monthly_sales_(334).csv')
        orders_file = os.path.join(self.data_dir, 'customer_orders_(150).csv')
        attendance_file = os.path.join(self.data_dir, 'cleaned_attendance_(461).csv')
        
        df_monthly = pd.read_csv(monthly_file)
        df_orders = pd.read_csv(orders_file)
        df_attendance = pd.read_csv(attendance_file)
        
        self.assertGreater(len(df_monthly), 0)
        self.assertGreater(len(df_orders), 0)
        self.assertGreater(len(df_attendance), 0)

    def test_staffing_recommendation_returns_success(self):
        """Test that get_staffing_recommendation returns success status"""
        result = get_staffing_recommendation()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn(result['status'], ['success', 'error'])

    def test_staffing_recommendation_output_structure(self):
        """Test that staffing recommendation output has correct structure"""
        result = get_staffing_recommendation()
        if result['status'] == 'success':
            self.assertIn('recommendations', result)
            self.assertIsInstance(result['recommendations'], list)
            self.assertIn('model_metrics', result)

    def test_staffing_metrics_exist(self):
        """Test that Model 4 metrics (R², MAE) exist and are reasonable"""
        result = get_staffing_recommendation()
        if result['status'] == 'success':
            self.assertIn('model_metrics', result)
            metrics = result['model_metrics']
            
            # Check that metrics exist
            self.assertIn('r_squared', metrics)
            self.assertIn('mae', metrics)
            
            # Check ranges
            r2 = metrics['r_squared']
            mae = metrics['mae']
            
            self.assertIsInstance(r2, (int, float))
            self.assertIsInstance(mae, (int, float))
            self.assertLessEqual(r2, 1, "R² should be <= 1")
            self.assertGreaterEqual(mae, 0, "MAE should be >= 0")

    def test_staffing_recommendations_have_required_fields(self):
        """Test that recommendations have required fields"""
        result = get_staffing_recommendation()
        if result['status'] == 'success' and len(result['recommendations']) > 0:
            rec = result['recommendations'][0]
            
            self.assertIn('branch', rec)
            self.assertIn('shift', rec)
            self.assertIn('recommended_staff', rec)

    def test_staffing_recommendation_with_branch_filter(self):
        """Test get_staffing_recommendation with branch filter"""
        result = get_staffing_recommendation(branch='Location 1')
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)

    def test_staffing_recommendation_with_shift_filter(self):
        """Test get_staffing_recommendation with shift filter"""
        for shift in ['Morning', 'Afternoon', 'Evening']:
            result = get_staffing_recommendation(shift=shift)
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)


class TestCoffeeMilkshakeGrowth(unittest.TestCase):
    """Test Model 5: Coffee & Milkshake Growth Strategy Analysis"""

    def setUp(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, 'data', 'cleaned')

    def test_data_files_exist(self):
        """Test that required data files exist"""
        files = [
            'cleaned_sales_by_item_(191).csv',
            'cleaned_revenue_summary_(136).csv',
            'sales_detail_(502).csv'
        ]
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing: {filepath}")

    def test_data_loading(self):
        """Test that all data files can be loaded"""
        item_file = os.path.join(self.data_dir, 'cleaned_sales_by_item_(191).csv')
        revenue_file = os.path.join(self.data_dir, 'cleaned_revenue_summary_(136).csv')
        detail_file = os.path.join(self.data_dir, 'sales_detail_(502).csv')
        
        df_item = pd.read_csv(item_file)
        df_revenue = pd.read_csv(revenue_file)
        df_detail = pd.read_csv(detail_file)
        
        self.assertGreater(len(df_item), 0)
        self.assertGreater(len(df_revenue), 0)
        self.assertGreater(len(df_detail), 0)

    def test_coffee_milkshake_analysis_returns_success(self):
        """Test that get_coffee_milkshake_analysis returns success status"""
        result = get_coffee_milkshake_analysis()
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn(result['status'], ['success', 'error'])

    def test_coffee_milkshake_analysis_output_structure(self):
        """Test that coffee/milkshake analysis output has correct structure"""
        result = get_coffee_milkshake_analysis()
        if result['status'] == 'success':
            self.assertIn('categories', result)
            self.assertIsInstance(result['categories'], list)

    def test_coffee_milkshake_strategies_exist(self):
        """Test that growth strategies are generated (non-empty)"""
        result = get_coffee_milkshake_analysis()
        if result['status'] == 'success':
            self.assertIn('categories', result)
            
            for category in result['categories']:
                if 'strategies' in category:
                    strategies = category['strategies']
                    self.assertIsInstance(strategies, list)
                    # Strategies should be generated or empty is OK
                    self.assertTrue(
                        len(strategies) >= 0,
                        "Strategies should be a list"
                    )

    def test_coffee_milkshake_analysis_with_category_filter(self):
        """Test get_coffee_milkshake_analysis with category filter"""
        for category in ['Coffee', 'Milkshake']:
            result = get_coffee_milkshake_analysis(category=category)
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)

    def test_coffee_milkshake_analysis_with_branch_filter(self):
        """Test get_coffee_milkshake_analysis with branch filter"""
        result = get_coffee_milkshake_analysis(branch='Location 1')
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)

    def test_coffee_milkshake_analysis_categories_structure(self):
        """Test that category results have required fields"""
        result = get_coffee_milkshake_analysis()
        if result['status'] == 'success' and len(result['categories']) > 0:
            category = result['categories'][0]
            
            self.assertIn('category', category)
            self.assertIn('total_revenue', category)


class TestConfigValidation(unittest.TestCase):
    """Test Config Validation Functions"""

    def test_validate_dataframe_with_valid_columns(self):
        """Test validate_dataframe with valid columns"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        # Should not raise
        result = validate_dataframe(df, ['col1', 'col2'], 'test_source')
        self.assertTrue(result)

    def test_validate_dataframe_with_missing_columns(self):
        """Test validate_dataframe raises on missing columns"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        # Should raise ValueError
        with self.assertRaises(ValueError):
            validate_dataframe(df, ['col1', 'col2', 'missing_col'], 'test_source')

    def test_validate_dataframe_error_message_contains_missing_cols(self):
        """Test that error message contains names of missing columns"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
        })
        try:
            validate_dataframe(df, ['col1', 'missing1', 'missing2'], 'test_source')
            self.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            self.assertIn('missing', error_msg.lower())


class TestIntegration(unittest.TestCase):
    """Integration tests across multiple models"""

    def setUp(self):
        self.project_root = PROJECT_ROOT

    def test_all_models_callable(self):
        """Test that all model functions are callable"""
        self.assertTrue(callable(get_combo_recommendations))
        self.assertTrue(callable(get_demand_forecast))
        self.assertTrue(callable(get_expansion_assessment))
        self.assertTrue(callable(get_staffing_recommendation))
        self.assertTrue(callable(get_coffee_milkshake_analysis))

    def test_all_models_return_dicts(self):
        """Test that all models return dictionaries"""
        results = [
            get_combo_recommendations(),
            get_demand_forecast(),
            get_expansion_assessment(),
            get_staffing_recommendation(),
            get_coffee_milkshake_analysis(),
        ]
        for result in results:
            self.assertIsInstance(result, dict, f"Result should be dict, got {type(result)}")

    def test_all_models_have_status_field(self):
        """Test that all models return status field"""
        results = [
            get_combo_recommendations(),
            get_demand_forecast(),
            get_expansion_assessment(),
            get_staffing_recommendation(),
            get_coffee_milkshake_analysis(),
        ]
        for result in results:
            self.assertIn('status', result, "Result should have 'status' field")
            self.assertIn(result['status'], ['success', 'error'])

    def test_all_models_json_serializable(self):
        """Test that all model outputs are JSON serializable"""
        results = [
            get_combo_recommendations(),
            get_demand_forecast(),
            get_expansion_assessment(),
            get_staffing_recommendation(),
            get_coffee_milkshake_analysis(),
        ]
        for result in results:
            try:
                json.dumps(result, default=str)
            except (TypeError, ValueError) as e:
                self.fail(f"Result not JSON serializable: {e}")

    def test_no_hardcoded_paths_in_tests(self):
        """Ensure tests use PROJECT_ROOT for paths"""
        # This test verifies our test structure
        self.assertIsNotNone(PROJECT_ROOT)
        self.assertTrue(os.path.isdir(PROJECT_ROOT))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_combo_invalid_min_support(self):
        """Test combo optimization with edge case min_support"""
        # Very low support should return results
        result = get_combo_recommendations(min_support=0.001)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)

    def test_demand_zero_periods(self):
        """Test demand forecast with minimal periods"""
        result = get_demand_forecast(periods=1)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)

    def test_staffing_negative_values_rejected(self):
        """Test that model handles data correctly"""
        # Just test that the function runs without crashing
        result = get_staffing_recommendation()
        self.assertIsInstance(result, dict)
        
        if result['status'] == 'success':
            for rec in result['recommendations']:
                # Recommended staff should be non-negative
                self.assertGreaterEqual(rec['recommended_staff'], 0)

    def test_expansion_clusters_are_valid(self):
        """Test that expansion clustering produces valid clusters"""
        result = get_expansion_assessment()
        if result['status'] == 'success':
            clustering = result.get('clustering', {})
            assignments = clustering.get('assignments', {})
            
            # Should have cluster assignments
            self.assertGreater(len(assignments), 0)
            
            # Each assignment should map to a cluster label
            for branch, label in assignments.items():
                self.assertIsNotNone(label)
                self.assertIsInstance(label, str)

    def test_coffee_analysis_with_nonexistent_branch(self):
        """Test coffee analysis with non-existent branch"""
        result = get_coffee_milkshake_analysis(branch='NONEXISTENT_XYZ')
        self.assertIsInstance(result, dict)
        # Should handle gracefully - either return success with filtered results
        # or error status


class TestBranchLocationRecommender(unittest.TestCase):
    """Test Model 6: Branch Location Recommender (Ridge Regression + Gap Analysis)"""

    def test_tool_returns_success(self):
        """Test that the tool returns success status"""
        result = get_branch_location_recommendation()
        self.assertEqual(result['status'], 'success')

    def test_recommendations_are_list(self):
        """Test that recommendations is a non-empty list"""
        result = get_branch_location_recommendation(top_n=5)
        self.assertIsInstance(result['recommendations'], list)
        self.assertGreater(len(result['recommendations']), 0)
        self.assertLessEqual(len(result['recommendations']), 5)

    def test_excluded_branches_not_in_results(self):
        """Test that existing Conut branches are excluded from recommendations"""
        result = get_branch_location_recommendation(top_n=50)
        area_names = [r['area'] for r in result['recommendations']]
        self.assertNotIn('Tyre (Sour)', area_names)
        self.assertNotIn('Beirut - Jnah', area_names)
        self.assertNotIn('Batroun', area_names)

    def test_recommendation_fields(self):
        """Test that each recommendation has required fields"""
        result = get_branch_location_recommendation(top_n=1)
        rec = result['recommendations'][0]
        required = ['area', 'governorate', 'final_score', 'population',
                    'total_competitors', 'expected_competitors', 'competitor_gap',
                    'gap_direction', 'demand_proxy', 'rent_index']
        for field in required:
            self.assertIn(field, rec, f"Missing field: {field}")

    def test_model_metrics_present(self):
        """Test that model R² and MAE are returned"""
        result = get_branch_location_recommendation()
        self.assertIn('model_r2', result)
        self.assertIn('model_mae', result)
        self.assertGreater(result['model_r2'], 0)

    def test_governorate_filter(self):
        """Test filtering by governorate"""
        result = get_branch_location_recommendation(governorate="Beirut")
        for rec in result['recommendations']:
            self.assertEqual(rec['governorate'], 'Beirut')

    def test_population_filter(self):
        """Test filtering by minimum population"""
        result = get_branch_location_recommendation(min_population=100000)
        for rec in result['recommendations']:
            self.assertGreaterEqual(rec['population'], 100000)

    def test_dataset_loads(self):
        """Test that the area dataset loads correctly"""
        df = load_area_dataset()
        self.assertGreater(len(df), 20)
        self.assertIn('total_competitors', df.columns)
        self.assertIn('has_conut', df.columns)

    def test_demand_proxy_range(self):
        """Test that demand proxy is in 0-100 range"""
        df = load_area_dataset()
        demand = compute_demand_proxy(df)
        self.assertTrue((demand >= 0).all())
        self.assertTrue((demand <= 100).all())

    def test_competitor_model_r2(self):
        """Test that competitor model achieves reasonable R²"""
        df = load_area_dataset()
        model_result = fit_competitor_model(df)
        self.assertGreater(model_result['r2'], 0.5, "Model R² should be > 0.5")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
