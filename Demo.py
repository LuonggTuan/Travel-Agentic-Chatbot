import os
import sqlite3
import time
import random
from datetime import date, datetime
from typing import Optional, Union, Literal
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import BaseModel, Field
import pytz

from config import settings

# ===========================
# CONFIGURATION & SETUP
# ===========================

CHROMA_PATH = "D:\\Dev\\WSLDev\\Final\\app\\vectorstore\\faq_chroma"
db = "D:\\Dev\\WSLDev\\Final\\app\\DB\\travel2.sqlite"

embedding = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retrieve = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

# ===========================
# SIMPLE MEMORY SETUP
# ===========================

# Simplified memory - only in RAM
memory = MemorySaver()
print("✅ In-memory conversation storage enabled")

# ===========================
# STATE MANAGEMENT
# ===========================

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]

# ===========================
# TOOLS
# ===========================

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted."""
    docs = retrieve.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None) 
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)
    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)
    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)
    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."

@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."

# ===========================
# BOOKING TOOLS
# ===========================

@tool
def search_car_rentals(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
) -> list[dict]:
    """Search for car rental companies based on location, name, and price tier."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT id, name, location, price_tier, booked FROM car_rentals WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if price_tier:
        query += " AND price_tier LIKE ?"
        params.append(f"%{price_tier}%")

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return [dict(zip([column[0] for column in cursor.description], row)) for row in results]

@tool
def book_car_rental(
    car_rental_id: int, 
    pickup_date: str, 
    return_date: str,
    *, 
    config: RunnableConfig
) -> str:
    """Book a car rental for specific dates."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("SELECT name, location FROM car_rentals WHERE id = ?", (car_rental_id,))
    car_rental = cursor.fetchone()
    if not car_rental:
        conn.close()
        return f"Car rental with ID {car_rental_id} not found."

    try:
        cursor.execute("""
            INSERT INTO car_rental_bookings (passenger_id, car_rental_id, booking_date, pickup_date, return_date)
            VALUES (?, ?, ?, ?, ?)
        """, (passenger_id, car_rental_id, datetime.now().strftime('%Y-%m-%d'), pickup_date, return_date))
        
        cursor.execute("UPDATE car_rentals SET booked = booked + 1 WHERE id = ?", (car_rental_id,))
        
        conn.commit()
        conn.close()
        return f"Car rental from {car_rental[0]} in {car_rental[1]} successfully booked from {pickup_date} to {return_date}."
    except Exception as e:
        conn.close()
        return f"Error booking car rental: {str(e)}"

@tool
def cancel_car_rental_booking(car_rental_id: int, *, config: RunnableConfig) -> str:
    """Cancel a car rental booking."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM car_rental_bookings 
        WHERE passenger_id = ? AND car_rental_id = ?
    """, (passenger_id, car_rental_id))
    
    if cursor.rowcount > 0:
        cursor.execute("UPDATE car_rentals SET booked = booked - 1 WHERE id = ?", (car_rental_id,))
        conn.commit()
        conn.close()
        return f"Car rental booking {car_rental_id} successfully cancelled."
    else:
        conn.close()
        return f"No car rental booking found for ID {car_rental_id}."

@tool
def get_user_car_rental_bookings(*, config: RunnableConfig) -> list[dict]:
    """Get all car rental bookings for the current user."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT crb.*, cr.name, cr.location, cr.price_tier
    FROM car_rental_bookings crb
    JOIN car_rentals cr ON crb.car_rental_id = cr.id
    WHERE crb.passenger_id = ?
    ORDER BY crb.booking_date DESC
    """
    cursor.execute(query, (passenger_id,))
    results = cursor.fetchall()
    conn.close()

    return [dict(zip([column[0] for column in cursor.description], row)) for row in results]

# Hotel tools (similar structure)
@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
) -> list[dict]:
    """Search for hotels based on location, name, and price tier."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT id, name, location, price_tier, booked FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if price_tier:
        query += " AND price_tier LIKE ?"
        params.append(f"%{price_tier}%")

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return [dict(zip([column[0] for column in cursor.description], row)) for row in results]

@tool
def book_hotel(
    hotel_id: int, 
    checkin_date: str, 
    checkout_date: str,
    *, 
    config: RunnableConfig
) -> str:
    """Book a hotel for specific dates."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("SELECT name, location FROM hotels WHERE id = ?", (hotel_id,))
    hotel = cursor.fetchone()
    if not hotel:
        conn.close()
        return f"Hotel with ID {hotel_id} not found."

    try:
        cursor.execute("""
            INSERT INTO hotel_bookings (passenger_id, hotel_id, booking_date, checkin_date, checkout_date)
            VALUES (?, ?, ?, ?, ?)
        """, (passenger_id, hotel_id, datetime.now().strftime('%Y-%m-%d'), checkin_date, checkout_date))
        
        cursor.execute("UPDATE hotels SET booked = booked + 1 WHERE id = ?", (hotel_id,))
        
        conn.commit()
        conn.close()
        return f"Hotel {hotel[0]} in {hotel[1]} successfully booked from {checkin_date} to {checkout_date}."
    except Exception as e:
        conn.close()
        return f"Error booking hotel: {str(e)}"

@tool
def cancel_hotel_booking(hotel_id: int, *, config: RunnableConfig) -> str:
    """Cancel a hotel booking."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM hotel_bookings 
        WHERE passenger_id = ? AND hotel_id = ?
    """, (passenger_id, hotel_id))
    
    if cursor.rowcount > 0:
        cursor.execute("UPDATE hotels SET booked = booked - 1 WHERE id = ?", (hotel_id,))
        conn.commit()
        conn.close()
        return f"Hotel booking {hotel_id} successfully cancelled."
    else:
        conn.close()
        return f"No hotel booking found for ID {hotel_id}."

@tool
def get_user_hotel_bookings(*, config: RunnableConfig) -> list[dict]:
    """Get all hotel bookings for the current user."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT hb.*, h.name, h.location, h.price_tier
    FROM hotel_bookings hb
    JOIN hotels h ON hb.hotel_id = h.id
    WHERE hb.passenger_id = ?
    ORDER BY hb.booking_date DESC
    """
    cursor.execute(query, (passenger_id,))
    results = cursor.fetchall()
    conn.close()

    return [dict(zip([column[0] for column in cursor.description], row)) for row in results]

# Trip recommendation tools (similar structure)
@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """Search for trip recommendations based on location, name, and keywords."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT id, name, location, keywords, details, booked FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return [dict(zip([column[0] for column in cursor.description], row)) for row in results]

@tool
def book_trip_recommendation(
    trip_id: int, 
    trip_date: str,
    *, 
    config: RunnableConfig
) -> str:
    """Book a trip recommendation for a specific date."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("SELECT name, location FROM trip_recommendations WHERE id = ?", (trip_id,))
    trip = cursor.fetchone()
    if not trip:
        conn.close()
        return f"Trip recommendation with ID {trip_id} not found."

    try:
        cursor.execute("""
            INSERT INTO trip_bookings (passenger_id, trip_id, booking_date)
            VALUES (?, ?, ?)
        """, (passenger_id, trip_id, datetime.now().strftime('%Y-%m-%d')))
        
        cursor.execute("UPDATE trip_recommendations SET booked = booked + 1 WHERE id = ?", (trip_id,))
        
        conn.commit()
        conn.close()
        return f"Trip '{trip[0]}' in {trip[1]} successfully booked for {trip_date}."
    except Exception as e:
        conn.close()
        return f"Error booking trip: {str(e)}"

@tool
def cancel_trip_booking(trip_id: int, *, config: RunnableConfig) -> str:
    """Cancel a trip booking."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM trip_bookings 
        WHERE passenger_id = ? AND trip_id = ?
    """, (passenger_id, trip_id))
    
    if cursor.rowcount > 0:
        cursor.execute("UPDATE trip_recommendations SET booked = booked - 1 WHERE id = ?", (trip_id,))
        conn.commit()
        conn.close()
        return f"Trip booking {trip_id} successfully cancelled."
    else:
        conn.close()
        return f"No trip booking found for ID {trip_id}."

@tool
def get_user_trip_bookings(*, config: RunnableConfig) -> list[dict]:
    """Get all trip bookings for the current user."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT tb.*, tr.name, tr.location, tr.keywords, tr.details
    FROM trip_bookings tb
    JOIN trip_recommendations tr ON tb.trip_id = tr.id
    WHERE tb.passenger_id = ?
    ORDER BY tb.booking_date DESC
    """
    cursor.execute(query, (passenger_id,))
    results = cursor.fetchall()
    conn.close()

    return [dict(zip([column[0] for column in cursor.description], row)) for row in results]

@tool
def get_all_user_bookings(*, config: RunnableConfig) -> dict:
    """Get comprehensive view of all user's bookings across flights, cars, hotels, and trips."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    flight_info = fetch_user_flight_information.invoke({}, config)
    car_bookings = get_user_car_rental_bookings.invoke({}, config)
    hotel_bookings = get_user_hotel_bookings.invoke({}, config) 
    trip_bookings = get_user_trip_bookings.invoke({}, config)

    return {
        "flights": flight_info,
        "car_rentals": car_bookings,
        "hotels": hotel_bookings,
        "trips": trip_bookings,
        "summary": {
            "total_flights": len(flight_info),
            "total_car_rentals": len(car_bookings),
            "total_hotels": len(hotel_bookings),
            "total_trips": len(trip_bookings)
        }
    }

# ===========================
# UTILITY FUNCTIONS
# ===========================

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            print(msg_repr)  # In ra toàn bộ không cắt
            _printed.add(message.id)

# ===========================
# ASSISTANT CLASSES
# ===========================

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant."""
    cancel: bool = True
    reason: str

# ===========================
# DELEGATION MODELS
# ===========================

class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""
    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )

class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""
    location: str = Field(description="The location where the user wants to rent a car.")
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(description="Any additional information or requests from the user.")

class ToHotelBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""
    location: str = Field(description="The location where the user wants to book a hotel.")
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(description="Any additional information or requests from the user.")

class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""
    location: str = Field(description="The location where the user wants to book a recommended trip.")
    request: str = Field(description="Any additional information or requests from the user.")

# ===========================
# PROMPTS WITH FIXED BOOKING LOGIC
# ===========================

flight_booking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling flight updates and cancellations. "
        "The primary assistant delegates work to you whenever the user needs help updating their bookings. "
        
        "IMPORTANT: You can ONLY update or cancel existing bookings. You CANNOT book new flights. "
        "If customers ask about booking new flights, tell them to visit: https://lat-airlines.com/book-flights "
        
        "When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "Confirm the updated flight details with the customer and inform them of any additional fees. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
        "\nCurrent time: {time}."
        "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
        ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time.',
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

book_car_rental_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling car rental bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
        
        "IMPORTANT - BOOKING AVAILABILITY:"
        "- The 'booked' field shows how many customers have currently reserved this service"
        "- If booked < 30: Show as 'Available to book'"
        "- If booked >= 30: Show as 'Currently unavailable - Fleet at maximum capacity. Due to high demand, all vehicles from this provider are currently reserved. Please try another car rental company or check back later for availability.'"
        "- NEVER show the actual booked number to customers - only show availability status"
        "- Present availability in a professional, customer-friendly manner"
        
        "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
        "When searching, expand your query bounds if the first search returns no results."
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        
        "\nCurrent time: {time}."
        
        'If the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.',
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

book_hotel_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling hotel bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
        
        "IMPORTANT - BOOKING AVAILABILITY:"
        "- The 'booked' field shows how many guests have currently reserved rooms at this hotel"
        "- If booked < 30: Show as 'Available to book'"
        "- If booked >= 30: Show as 'Currently unavailable - Hotel at full occupancy. Due to high demand, this hotel has reached maximum capacity for the requested dates. We recommend selecting an alternative hotel or adjusting your travel dates.'"
        "- NEVER show the actual booked number to customers - only show availability status"
        "- Present availability in a professional, customer-friendly manner"
        
        "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
        "When searching, expand your query bounds if the first search returns no results."
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        
        "\nCurrent time: {time}."
        
        'If the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.',
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

book_excursion_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling trip recommendations and excursion bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
        
        "IMPORTANT - BOOKING AVAILABILITY:"
        "- The 'booked' field shows how many travelers have currently reserved spots for this excursion"
        "- If booked < 30: Show as 'Available to book'"
        "- If booked >= 30: Show as 'Currently unavailable - Tour at maximum capacity. Due to high demand and guide availability constraints, this excursion has reached its group size limit. Please consider alternative tours or different dates for the best experience.'"
        "- NEVER show the actual booked number to customers - only show availability status"
        "- Present availability in a professional, customer-friendly manner"
        
        "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
        "When searching, expand your query bounds if the first search returns no results."
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        
        "\nCurrent time: {time}."
        
        'If the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.',
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

primary_assistant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful customer support assistant for LAT Airlines. "
        "Your primary role is to search for flight information and company policies to answer customer queries. "
        
        "IMPORTANT: When you provide flight search results or when customers ask about booking new flights, "
        "you MUST always include this information:\n"
        "- This chatbot can only UPDATE or CANCEL existing flight bookings\n"
        "- For NEW flight bookings, direct them to: https://lat-airlines.com/book-flights\n"
        "- Explain that the website has live booking, pricing, and customer support\n"
        
        "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
        "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. "
        "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
        "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
        "\nCurrent time: {time}."
        "\n\nYou also have access to get_all_user_bookings tool to show comprehensive booking information across all services.",
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

# ===========================
# TOOL GROUPS AND RUNNABLES
# ===========================

# Tool groups
update_flight_safe_tools = [search_flights]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools

book_car_rental_safe_tools = [search_car_rentals, get_user_car_rental_bookings]
book_car_rental_sensitive_tools = [book_car_rental, cancel_car_rental_booking]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools

book_hotel_safe_tools = [search_hotels, get_user_hotel_bookings]
book_hotel_sensitive_tools = [book_hotel, cancel_hotel_booking]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools

book_excursion_safe_tools = [search_trip_recommendations, get_user_trip_bookings]
book_excursion_sensitive_tools = [book_trip_recommendation, cancel_trip_booking]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools

primary_assistant_tools = [
    search_flights,
    lookup_policy,
    get_all_user_bookings,
]

# Create runnable chains
update_flight_runnable = flight_booking_prompt | llm.bind_tools(
    update_flight_tools + [CompleteOrEscalate]
)

book_car_rental_runnable = book_car_rental_prompt | llm.bind_tools(
    book_car_rental_tools + [CompleteOrEscalate]
)

book_hotel_runnable = book_hotel_prompt | llm.bind_tools(
    book_hotel_tools + [CompleteOrEscalate]
)

book_excursion_runnable = book_excursion_prompt | llm.bind_tools(
    book_excursion_tools + [CompleteOrEscalate]
)

assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools + [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion,
    ]
)

# ===========================
# SIMPLIFIED ROUTING
# ===========================

def route_to_workflow(state: State) -> Literal["primary_assistant", "update_flight", "book_car_rental", "book_hotel", "book_excursion"]:
    """Simplified routing - always start with primary assistant for fresh conversations."""
    # For simplicity, always route to primary assistant
    # Let the primary assistant handle delegation through tool calls
    return "primary_assistant"

# ===========================
# GRAPH CONSTRUCTION
# ===========================

def create_entry_node(assistant_name: str, new_dialog_state: str):
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node

builder = StateGraph(State)

def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")

# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

def route_primary_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

builder.add_conditional_edges("primary_assistant", route_primary_assistant, [
    "enter_update_flight", "enter_book_car_rental", "enter_book_hotel", 
    "enter_book_excursion", "primary_assistant_tools", END
])
builder.add_edge("primary_assistant_tools", "primary_assistant")

# Shared exit node
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant."""
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {"dialog_state": "pop", "messages": messages}

builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

# Flight booking assistant
builder.add_node("enter_update_flight", create_entry_node("Flight Updates & Booking Assistant", "update_flight"))
builder.add_node("update_flight", Assistant(update_flight_runnable))
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node("update_flight_sensitive_tools", create_tool_node_with_fallback(update_flight_sensitive_tools))
builder.add_node("update_flight_safe_tools", create_tool_node_with_fallback(update_flight_safe_tools))

def route_update_flight(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_flight_safe_tools"
    return "update_flight_sensitive_tools"

builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_edge("update_flight_safe_tools", "update_flight")
builder.add_conditional_edges("update_flight", route_update_flight, ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END])

# Car rental assistant
builder.add_node("enter_book_car_rental", create_entry_node("Car Rental Assistant", "book_car_rental"))
builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_node("book_car_rental_safe_tools", create_tool_node_with_fallback(book_car_rental_safe_tools))
builder.add_node("book_car_rental_sensitive_tools", create_tool_node_with_fallback(book_car_rental_sensitive_tools))

def route_book_car_rental(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "book_car_rental_safe_tools"
    return "book_car_rental_sensitive_tools"

builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
builder.add_conditional_edges("book_car_rental", route_book_car_rental, ["book_car_rental_safe_tools", "book_car_rental_sensitive_tools", "leave_skill", END])

# Hotel booking assistant
builder.add_node("enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel"))
builder.add_node("book_hotel", Assistant(book_hotel_runnable))
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node("book_hotel_safe_tools", create_tool_node_with_fallback(book_hotel_safe_tools))
builder.add_node("book_hotel_sensitive_tools", create_tool_node_with_fallback(book_hotel_sensitive_tools))

def route_book_hotel(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_hotel_safe_tools"
    return "book_hotel_sensitive_tools"

builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
builder.add_edge("book_hotel_safe_tools", "book_hotel")
builder.add_conditional_edges("book_hotel", route_book_hotel, ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END])

# Excursion assistant
builder.add_node("enter_book_excursion", create_entry_node("Trip Recommendation Assistant", "book_excursion"))
builder.add_node("book_excursion", Assistant(book_excursion_runnable))
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node("book_excursion_safe_tools", create_tool_node_with_fallback(book_excursion_safe_tools))
builder.add_node("book_excursion_sensitive_tools", create_tool_node_with_fallback(book_excursion_sensitive_tools))

def route_book_excursion(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_excursion_safe_tools"
    return "book_excursion_sensitive_tools"

builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
builder.add_edge("book_excursion_safe_tools", "book_excursion")
builder.add_conditional_edges("book_excursion", route_book_excursion, ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END])

# Connect fetch_user_info to primary_assistant
builder.add_edge("fetch_user_info", "primary_assistant")

# ===========================
# COMPILE GRAPH
# ===========================

part_4_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)

print("✅ Graph compiled successfully with in-memory checkpointer")

# ===========================
# SIMPLIFIED DATABASE HELPERS
# ===========================

def validate_passenger_id(passenger_id: str) -> bool:
    """Simple passenger ID validation"""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE passenger_id = ?", (passenger_id,))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0
    except Exception:
        return False

def get_passenger_info(passenger_id: str) -> str:
    """Get passenger info"""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE passenger_id = ?", (passenger_id,))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return f"Welcome! You have {count} ticket(s) in our system."
    except Exception:
        return "Welcome! (Could not retrieve detailed info)"

def start_conversation():
    """Simple conversation starter"""
    print("=" * 50)
    print("   SWISS AIRLINES CUSTOMER SERVICE SYSTEM")
    print("=" * 50)
    print()
    
    while True:
        passenger_id = input("Please enter your Passenger ID: ").strip()
        
        if not passenger_id:
            print("❌ Passenger ID cannot be empty. Please try again.")
            continue
            
        if validate_passenger_id(passenger_id):
            print("✅ Passenger ID verified!")
            passenger_info = get_passenger_info(passenger_id)
            print(passenger_info)
            print()
            return passenger_id
        else:
            print("❌ Invalid Passenger ID. Please check and try again.")
            
            # Show sample IDs
            try:
                conn = sqlite3.connect(db)
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT passenger_id FROM tickets LIMIT 3")
                sample_ids = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if sample_ids:
                    print("💡 Sample valid Passenger IDs for testing:")
                    for idx, (pid,) in enumerate(sample_ids, 1):
                        print(f"   {idx}. {pid}")
            except Exception:
                pass
            print()

# ===========================
# MAIN APPLICATION
# ===========================

def main():
    """Simplified main application"""
    
    print("🚀 SWISS AIRLINES SYSTEM STARTING...")
    print("-" * 40)
    print("⚠️  In-memory conversation storage - conversations reset on restart")
    print()
    
    # Get passenger ID
    try:
        passenger_id = start_conversation()
    except Exception as e:
        print(f"❌ Conversation setup failed: {e}")
        return
    
    # Create unique thread ID for each session
    thread_id = f"user_{passenger_id.replace(' ', '_')}_{int(time.time())}"
    
    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": thread_id,
        }
    }
    
    _printed = set()
    
    print("=== SWISS AIRLINES ASSISTANT ===")
    print("I'm here to help you with your flight information, bookings, and travel needs.")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("=" * 50)
    
    while True:
        try:
            question = input(f"\n[{passenger_id}] Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\n✈️ Thank you for using Swiss Airlines customer service!")
                print("Have a great flight! 🛫")
                break
                
            try:
                print("Processing your request...")
                
                events = part_4_graph.stream(
                    {"messages": ("user", question)}, config, stream_mode="values"
                )
                
                for event in events:
                    _print_event(event, _printed)
                
                # Handle approval workflow
                snapshot = part_4_graph.get_state(config)
                    
                while snapshot and snapshot.next:
                    try:
                        user_input = input(
                            "\n🔐 Do you approve of the above actions? Type 'y' to continue; "
                            "otherwise, explain your requested changes.\n\n"
                        )
                    except:
                        user_input = "y"
                        
                    if user_input.strip().lower() == "y":
                        result = part_4_graph.invoke(None, config)
                        
                        # Check for booking confirmation
                        try:
                            updated_snapshot = part_4_graph.get_state(config)
                            latest_messages = updated_snapshot.values.get("messages", [])
                            
                            booking_confirmed = False
                            service_type = "service"
                            
                            for msg in latest_messages[-3:]:
                                if hasattr(msg, 'content') and msg.content:
                                    content = str(msg.content).lower()
                                    if any(keyword in content for keyword in ["successfully booked", "booking confirmed", "successfully updated"]):
                                        booking_confirmed = True
                                        if "car rental" in content or "rental" in content:
                                            service_type = "car rental"
                                        elif "hotel" in content:
                                            service_type = "hotel"
                                        elif "excursion" in content or "trip" in content:
                                            service_type = "excursion"
                                        elif "flight" in content:
                                            service_type = "flight"
                                        break
                            
                            if booking_confirmed:
                                print("\n" + "="*60)
                                print("🎉 BOOKING CONFIRMATION")
                                print("="*60)
                                print(f"✅ Your {service_type} has been successfully booked!")
                                print("📞 Our customer service team will contact you shortly")
                                print("   to confirm the details and finalize your booking.")
                                print("📧 You will also receive a confirmation email soon.")
                                print("="*60)
                                print()
                        except Exception:
                            print()
                    else:
                        try:
                            if snapshot.next and len(snapshot.next) > 0:
                                last_ai_message = None
                                for msg in reversed(snapshot.values.get("messages", [])):
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        last_ai_message = msg
                                        break
                                
                                if last_ai_message and last_ai_message.tool_calls:
                                    tool_call_id = last_ai_message.tool_calls[0]["id"]
                                else:
                                    break
                            else:
                                break
                        except (IndexError, KeyError, AttributeError):
                            break
                            
                        result = part_4_graph.invoke(
                            {
                                "messages": [
                                    ToolMessage(
                                        tool_call_id=tool_call_id,
                                        content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                    )
                                ]
                            },
                            config,
                        )
                    
                    snapshot = part_4_graph.get_state(config)
                    
            except Exception as graph_error:
                print(f"❌ Processing error: {graph_error}")
                print("💡 Please try rephrasing your question or contact support.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n✈️ Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except Exception as main_error:
        print(f"❌ Application failed to start: {main_error}")
        print("💡 Please check your setup and try again.")