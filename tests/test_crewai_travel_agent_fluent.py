from asyncio import sleep
import os
import sys
import pytest

# Add parent directory to path to import crewai_travel_agent module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monocle_test_tools import TraceAssertion
from crewai_travel_agent import create_crewai_travel_crew

@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter: TraceAssertion):
    """Test tool invocation with CrewAI travel agent."""
    travel_request = "Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental in Mumbai for 27th April 2026 for 4 nights."
    crew = create_crewai_travel_crew(travel_request)
    
    await monocle_trace_asserter.run_agent_async(crew, "crewai", travel_request)
    
    monocle_trace_asserter.called_tool("book_flight", "Flight Booking Agent") \
        .contains_input("Mumbai").contains_input("San Francisco").contains_input("26th April 2026") \
        .contains_output("San Francisco to Mumbai").contains_output("success")
    
    monocle_trace_asserter.called_tool("book_hotel", "Hotel Booking Agent") \
        .contains_input("Mumbai").contains_input("27th April 2026").contains_input("Marriott") \
        .contains_output("Successfully booked a stay at Marriott").contains_output("success")


@pytest.mark.asyncio
async def test_agent_invocation(monocle_trace_asserter: TraceAssertion):
    """Test agent invocation with CrewAI travel agent."""
    travel_request = "Book a flight from San Francisco to Mumbai for 28th April 2026. Book a two queen room at Marriott Intercontinental in Mumbai for 29th April 2026 for 4 nights."
    crew = create_crewai_travel_crew(travel_request)
    
    await monocle_trace_asserter.run_agent_async(crew, "crewai", travel_request)
    
    monocle_trace_asserter.called_agent("Flight Booking Agent") \
        .contains_input("Book a flight from San Francisco to Mumbai for 28th April 2026") \
        .contains_output("booked a flight from San Francisco to Mumbai")
    
    monocle_trace_asserter.called_agent("Hotel Booking Agent") \
        .contains_input("Marriott Intercontinental in Mumbai") \
        .contains_output("booked")
    
    monocle_trace_asserter.called_agent("Supervisor Travel Agent") \
        .contains_input("Book a flight from San Francisco to Mumbai") \
        .contains_output("Mumbai")


@pytest.mark.asyncio
async def test_complete_booking_flow(monocle_trace_asserter: TraceAssertion):
    """Test complete booking flow with both flight and hotel"""
    travel_request = "I need to travel from New York to London on January 15th, 2026. Please book me a flight and a hotel stay for 5 nights."
    crew = create_crewai_travel_crew(travel_request)
    
    await monocle_trace_asserter.run_agent_async(crew, "crewai", travel_request)
    
    # Verify flight booking
    monocle_trace_asserter.called_tool("book_flight", "Flight Booking Agent") \
        .contains_input("New York").contains_input("London") \
        .contains_output("success")
    
    # Verify hotel booking
    monocle_trace_asserter.called_tool("book_hotel", "Hotel Booking Agent") \
        .contains_input("London").contains_input("5") \
        .contains_output("success")
    
    # Verify supervisor coordinated everything
    monocle_trace_asserter.called_agent("Supervisor Travel Agent") \
        .contains_output("flight").contains_output("hotel")


@pytest.mark.asyncio
async def test_hotel_only_booking(monocle_trace_asserter: TraceAssertion):
    """Test hotel-only booking"""
    travel_request = "Book a Marriott hotel in New York for 3 nights"
    crew = create_crewai_travel_crew(travel_request)
    
    await monocle_trace_asserter.run_agent_async(crew, "crewai", travel_request)
    
    monocle_trace_asserter.called_tool("book_hotel", "Hotel Booking Agent") \
        .contains_input("Marriott").contains_input("New York").contains_input("3") \
        .contains_output("success").contains_output("Marriott in New York")
    
    monocle_trace_asserter.called_agent("Hotel Booking Agent") \
        .contains_input("Marriott hotel in New York") \
        .contains_output("booked")


@pytest.mark.asyncio
async def test_flight_only_booking(monocle_trace_asserter: TraceAssertion):
    """Test flight-only booking"""
    travel_request = "Book a flight from JFK to LAX"
    crew = create_crewai_travel_crew(travel_request)
    
    await monocle_trace_asserter.run_agent_async(crew, "crewai", travel_request)
    
    monocle_trace_asserter.called_tool("book_flight", "Flight Booking Agent") \
        .contains_input("JFK").contains_input("LAX") \
        .contains_output("success").contains_output("JFK to LAX")
    
    monocle_trace_asserter.called_agent("Flight Booking Agent") \
        .contains_input("flight from JFK to LAX") \
        .contains_output("booked")


if __name__ == "__main__":
    pytest.main([__file__])
