import logging
import time
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
import uuid
from langchain_openai import ChatOpenAI

from monocle_apptrace.instrumentation.common.scope_wrapper import monocle_trace_scope
from monocle_apptrace.instrumentation import setup_monocle_telemetry
    # Setup monocle telemetry
setup_monocle_telemetry(workflow_name="crewai_travel_agent")

logging.basicConfig(level=logging.WARN)

# Hotel booking tool
class BookHotelTool(BaseTool):
    name: str = "book_hotel"
    description: str = "Book a hotel reservation"

    def _run(self, hotel_name: str, city: str, check_in_date: str, nights: int = 1) -> dict:
        """Books a hotel for a stay.

        Args:
            hotel_name (str): The name of the hotel to book.
            city (str): The city where the hotel is located.
            check_in_date (str): The check-in date for the hotel stay.
            nights (int): The number of nights for the hotel stay.

        Returns:
            dict: status and message.
        """
        time.sleep(0.1)  # Simulate processing time
        return {
            "status": "success",
            "message": f"Successfully booked a stay at {hotel_name} in {city} from {check_in_date} for {nights} nights."
        }


# Flight booking tool
class BookFlightTool(BaseTool):
    name: str = "book_flight"
    description: str = "Book a flight reservation"

    def _run(self, from_airport: str, to_airport: str, date: str = "next week") -> dict:
        """Books a flight from one airport to another.

        Args:
            from_airport (str): The airport from which the flight departs.
            to_airport (str): The airport to which the flight arrives.
            date (str): The date of the flight.

        Returns:
            dict: status and message.
        """
        time.sleep(0.1)  # Simulate processing time
        return {
            "status": "success",
            "message": f"Flight booked from {from_airport} to {to_airport} for {date}."
        }


# Create tools
hotel_tool = BookHotelTool()
flight_tool = BookFlightTool()

def create_agents():
    """Create CrewAI agents. Only call this when OpenAI API key is available."""
    # Create hotel booking agent
    hotel_booking_agent = Agent(
        role="Hotel Booking Agent",
        goal="Book the best hotel accommodations for travelers",
        backstory="You are an expert hotel booking specialist. When specific details like check-in dates or number of nights are not provided, make reasonable assumptions based on context. For example, if a flight date is mentioned, assume hotel check-in on the same date and 1 night stay by default. You MUST use the book_hotel tool to complete any hotel booking.",
        tools=[hotel_tool],
        llm=ChatOpenAI(model="gpt-4o-mini", tool_choice="required"),
        verbose=False,
        allow_delegation=False,
        max_iter=5,
        step_callback=None,
        memory = True
    )

    # Create flight booking agent
    flight_booking_agent = Agent(
        role="Flight Booking Agent", 
        goal="Book the best flight options for travelers",
        backstory="You are an expert flight booking specialist. When dates like 'next week' are mentioned, make reasonable assumptions (e.g., 7 days from today). Always proceed with booking using your best judgment. You MUST use the book_flight tool to complete any flight booking.",
        tools=[flight_tool],
        llm=ChatOpenAI(model="gpt-4o-mini", tool_choice="required"),
        verbose=False,
        allow_delegation=False,
        max_iter=5,
        step_callback=None,
        memory = True
    )

    # Create supervisor agent - with access to all tools to avoid delegation issues
    supervisor_agent = Agent(
        role="Supervisor Travel Agent",
        goal="Coordinate complete travel bookings by directly using specialist tools",
        backstory="You are a travel supervisor who can directly book hotels and flights. Make reasonable assumptions when details are missing and proceed with bookings.",
        tools=[hotel_tool, flight_tool],  # Give supervisor direct access to tools
        llm=ChatOpenAI(model="gpt-4o-mini", tool_choice="auto"),
        verbose=False,
        allow_delegation=False,  # Disable delegation to avoid validation errors,
        max_iter=5,
        step_callback=None,
        memory = True
    )
    
    return hotel_booking_agent, flight_booking_agent, supervisor_agent

# Initialize agents only when imported by tests (not when run directly)
hotel_booking_agent = None
flight_booking_agent = None 
supervisor_agent = None

def generate_session_id():
    return str(uuid.uuid4())

def create_crewai_travel_crew(travel_request: str):
    """Create a CrewAI crew for travel booking based on the request."""
    global hotel_booking_agent, flight_booking_agent, supervisor_agent
    # Initialize agents if not already done
    if hotel_booking_agent is None:
        hotel_booking_agent, flight_booking_agent, supervisor_agent = create_agents()
    # Create tasks based on the request
    tasks = []
    
    # Check if hotel booking is needed
    if any(keyword in travel_request.lower() for keyword in ['hotel', 'stay', 'accommodation', 'room']):
        hotel_task = Task(
            name="Hotel Booking Task",
            description=f"Extract hotel booking details from this request and book accordingly: {travel_request}",
            expected_output="Hotel booking confirmation with details",
            agent=hotel_booking_agent
        )
        tasks.append(hotel_task)
    
    # Check if flight booking is needed
    if any(keyword in travel_request.lower() for keyword in ['flight', 'fly', 'travel', 'airport']):
        flight_task = Task(
            name="Flight Booking Task",
            description=f"Extract flight booking details from this request and book accordingly: {travel_request}",
            expected_output="Flight booking confirmation with details",
            agent=flight_booking_agent
        )
        tasks.append(flight_task)
    
    # Add supervisor task to coordinate
    supervisor_task = Task(
        name="Travel Coordination Task",
        description=f"Coordinate and summarize the complete travel booking for: {travel_request}. Ensure all requested services are booked and provide a comprehensive summary.",
        expected_output="Complete travel booking summary with all confirmations",
        agent=supervisor_agent
    )
    tasks.append(supervisor_task)

    # Create crew
    crew = Crew(
        agents=[supervisor_agent, hotel_booking_agent, flight_booking_agent],
        tasks=tasks,
        verbose=True,
        process="sequential",
        memory=True
    )
    return crew

def execute_crewai_travel_request(travel_request: str):
    """Execute a travel request using CrewAI and return the result."""
    crew = create_crewai_travel_crew(travel_request)
    result = crew.kickoff(inputs={
        "travel_request": travel_request
    })
    return str(result)


if __name__ == "__main__":
    session_id = generate_session_id()
    print(f"Session: {session_id}")
    with monocle_trace_scope("agentic.session", session_id):
        while True:
            try:
                user_request = input("\nI am a travel booking agent. How can I assist you with your travel plans? ").strip()
                
                # Check for quit commands
                if user_request.lower() in ["quit", "exit"]:
                    print("\nEnding session. Goodbye!")
                    break
                
                # Skip empty inputs
                if not user_request:
                    continue
                
                # Execute the travel request
                result = execute_crewai_travel_request(user_request)
                
                print(f"\nAssistant: {result}\n")
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n\nSession interrupted. Goodbye!")
                break

            except Exception as e:
                # Catch any unexpected errors without crashing
                print(f"\nAn error occurred: {e}\n")

