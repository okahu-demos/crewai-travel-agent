"""
Test suite for CrewAI Travel Agent using Monocle test framework.
This test file uses MonocleValidator to trace test execution 
and send results to Okahu portal.
"""

from asyncio import sleep
import os
import sys
import pytest 
import logging
from dotenv import load_dotenv

# Add parent directory to path to import crewai_travel_agent module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crewai_travel_agent import execute_crewai_travel_request_async
from monocle_test_tools import TestCase, MonocleValidator

OKAHU_API_KEY = os.environ.get('OKAHU_API_KEY')
logging.basicConfig(level=logging.WARN)
load_dotenv()

agent_test_cases: list[TestCase] = [
    # Basic similarity test
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th March 2026. Book a hotel at Marriott in Mumbai for 27th March 2026 for 4 nights."],
        "test_output": "A flight from San Francisco to Mumbai has been booked, along with a stay at Marriott in Mumbai for 4 nights",
        "comparer": "similarity",
    },
    
    # Complete workflow test with all agents and tools
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental in Mumbai for 27th April 2026 for 4 nights."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "output": "A flight from San Francisco to Mumbai has been booked, along with a four-night stay at the Marriott Intercontinental in Mumbai, starting April 27th, 2026.",
                "comparer": "similarity",
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Supervisor Travel Agent"}
                ]
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Flight Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_flight"},
                    {"type": "agent", "name": "Flight Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Hotel Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_hotel"},
                    {"type": "agent", "name": "Hotel Booking Agent"}
                ]
            }
        ]
    },
    
    # Test with detailed span checks
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental in Mumbai for 27th April 2026 for 4 nights."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "comparer": "similarity",
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Supervisor Travel Agent"}
                ]
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Flight Booking Agent"}
                ],
                "comparer": "similarity"
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_flight"},
                    {"type": "agent", "name": "Flight Booking Agent"}
                ],
                "expect_errors": False,
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Hotel Booking Agent"}
                ],
                "comparer": "similarity"
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_hotel"},
                    {"type": "agent", "name": "Hotel Booking Agent"}
                ],
                "comparer": "similarity",
            }
        ]
    },
    
    # Another complete test with different inputs
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental in Mumbai for 27th April 2026 for 4 nights."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "comparer": "similarity",
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Supervisor Travel Agent"}
                ]
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Flight Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_flight"},
                    {"type": "agent", "name": "Flight Booking Agent"}
                ],
                "expect_errors": False,
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "Hotel Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_hotel"},
                    {"type": "agent", "name": "Hotel Booking Agent"}
                ],
                "expect_errors": False,
            }
        ]
    },
    
    # Hotel only test
    {
        "test_input": ["Book a Marriott hotel in New York for 3 nights"],
        "test_spans": [
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_hotel"}
                ],
            }
        ],
        "expect_errors": False,
    },
    
    # Flight only test
    {
        "test_input": ["Book a flight from JFK to LAX"],
        "test_spans": [
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "book_flight"}
                ],
            }
        ],
        "expect_errors": False,
    },
]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_run_agents(my_test_case: TestCase):
    await MonocleValidator().test_workflow_async(execute_crewai_travel_request_async, my_test_case)
    await sleep(2)  # Ensure all telemetry is flushed

if __name__ == "__main__":
    pytest.main([__file__])
