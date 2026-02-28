#!/usr/bin/env python3
"""
OpenClaw / Agent wrapper script for Conut analytics tools.

Usage:
    python3 scripts/run_tool.py <tool_name> [params_json]

Examples:
    python3 scripts/run_tool.py get_combo_recommendations
    python3 scripts/run_tool.py get_combo_recommendations '{"branch": "Conut Jnah", "top_n": 5}'
"""

import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
os.chdir(project_root)

from combo_optimization import get_combo_recommendations
from coffee_milkshake_growth import get_coffee_milkshake_analysis
from demand_forecasting import get_demand_forecast
from shift_staffing import get_staffing_recommendation
from expansion_feasibility import get_expansion_assessment
from branch_location_recommender import get_branch_location_recommendation


TOOL_MAP = {
    "get_combo_recommendations": get_combo_recommendations,
    "get_coffee_milkshake_analysis": get_coffee_milkshake_analysis,
    "get_demand_forecast": get_demand_forecast,
    "get_staffing_recommendation": get_staffing_recommendation,
    "get_expansion_assessment": get_expansion_assessment,
    "get_branch_location_recommendation": get_branch_location_recommendation,
}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": f"Usage: python3 scripts/run_tool.py <tool_name> [params_json]\nAvailable tools: {list(TOOL_MAP.keys())}",
        }, indent=2))
        sys.exit(1)

    tool_name = sys.argv[1]
    params = {}

    if len(sys.argv) >= 3:
        try:
            params = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(json.dumps({
                "status": "error",
                "message": f"Invalid JSON parameters: {e}",
            }, indent=2))
            sys.exit(1)

    if tool_name not in TOOL_MAP:
        print(json.dumps({
            "status": "error",
            "message": f"Unknown tool: {tool_name}. Available: {list(TOOL_MAP.keys())}",
        }, indent=2))
        sys.exit(1)

    # Call the tool
    fn = TOOL_MAP[tool_name]
    result = fn(**params)

    # Output clean JSON
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
