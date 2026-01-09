"""
AIRGO Backend Server
This handles the AI intelligence (Claude API) and flight data from Amadeus API
Now supports MULTIPLE Nigerian airlines!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
import httpx
from datetime import datetime, timedelta
from typing import Optional, List
import json

app = FastAPI(title="AIRGO API")

# Allow our frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CONFIGURATION
# ============================================
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY", "")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET", "")

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Amadeus API endpoints
AMADEUS_TOKEN_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_SEARCH_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"


# ============================================
# DATA MODELS
# ============================================
class ChatMessage(BaseModel):
    message: str
    history: Optional[List[dict]] = None


class BookingState(BaseModel):
    """Tracks the current booking state"""
    flight_selected: Optional[dict] = None
    passenger_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    payment_method: Optional[str] = None
    booking_reference: Optional[str] = None
    step: str = "none"  # none, awaiting_details, confirming, completed


# Store booking states per session (in production, use a database)
booking_sessions = {}


class FlightInfo(BaseModel):
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price: str
    currency: str
    duration: str
    available_seats: Optional[int] = None


# ============================================
# AMADEUS API INTEGRATION
# ============================================
class AmadeusClient:
    """Handles Amadeus API authentication and requests"""
    
    def __init__(self):
        self.api_key = AMADEUS_API_KEY
        self.api_secret = AMADEUS_API_SECRET
        self.access_token = None
        self.token_expires_at = None
    
    async def get_access_token(self):
        """Get or refresh Amadeus access token"""
        # Check if we have a valid token
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token
        
        # Get new token
        async with httpx.AsyncClient() as client:
            response = await client.post(
                AMADEUS_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.api_secret
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                # Token expires in seconds, set expiry time
                expires_in = data.get("expires_in", 1799)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
                return self.access_token
            else:
                raise Exception(f"Failed to get Amadeus token: {response.text}")
    
    async def search_flights(self, origin: str, destination: str, date: str = None, adults: int = 1):
        """Search for flights using Amadeus API"""
        try:
            token = await self.get_access_token()
            
            # Format date (default to tomorrow if not provided)
            if not date:
                search_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                search_date = self.parse_date(date)
            
            # Build search parameters
            params = {
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": search_date,
                "adults": adults,
                "max": 10,  # Get up to 10 results
                "currencyCode": "NGN"  # Nigerian Naira
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    AMADEUS_FLIGHT_SEARCH_URL,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return self.parse_flight_offers(response.json())
                else:
                    print(f"Amadeus API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"Error searching flights: {e}")
            return None
    
    def parse_date(self, date_str: str) -> str:
        """Convert natural language date to YYYY-MM-DD format"""
        date_str_lower = date_str.lower()
        
        if "today" in date_str_lower:
            return datetime.now().strftime("%Y-%m-%d")
        elif "tomorrow" in date_str_lower:
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "next week" in date_str_lower:
            return (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        else:
            # Default to tomorrow
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    def parse_flight_offers(self, data: dict) -> List[FlightInfo]:
        """Parse Amadeus API response into FlightInfo objects"""
        flights = []
        
        if not data.get("data"):
            return flights
        
        for offer in data["data"]:
            try:
                # Get the first itinerary (one-way flights have one itinerary)
                itinerary = offer["itineraries"][0]
                segment = itinerary["segments"][0]  # First segment
                
                # Extract flight details
                airline_code = segment["carrierCode"]
                flight_number = f"{airline_code}{segment['number']}"
                
                # Get airline name from dictionaries
                airline_name = data.get("dictionaries", {}).get("carriers", {}).get(airline_code, airline_code)
                
                # Parse times
                departure = segment["departure"]
                arrival = segment["arrival"]
                
                # Format times nicely
                dep_time = datetime.fromisoformat(departure["at"].replace("Z", "+00:00"))
                arr_time = datetime.fromisoformat(arrival["at"].replace("Z", "+00:00"))
                
                # Get price
                price_info = offer["price"]
                price = price_info["total"]
                currency = price_info["currency"]
                
                # Calculate duration
                duration = itinerary["duration"].replace("PT", "").replace("H", "h ").replace("M", "m")
                
                flight = FlightInfo(
                    airline=airline_name,
                    flight_number=flight_number,
                    origin=f"{departure['iataCode']}",
                    destination=f"{arrival['iataCode']}",
                    departure_time=dep_time.strftime("%I:%M %p"),
                    arrival_time=arr_time.strftime("%I:%M %p"),
                    price=f"₦{float(price):,.2f}" if currency == "NGN" else f"{price} {currency}",
                    currency=currency,
                    duration=duration,
                    available_seats=offer.get("numberOfBookableSeats")
                )
                
                flights.append(flight)
                
            except Exception as e:
                print(f"Error parsing flight offer: {e}")
                continue
        
        return flights

# Initialize Amadeus client
amadeus = AmadeusClient()


# ============================================
# CLAUDE AI INTEGRATION
# ============================================
async def process_with_claude(user_message: str, flight_data: List[FlightInfo] = None, conversation_history: List[dict] = None):
    """
    Uses Claude to understand the user's request and provide intelligent responses
    Now with conversation memory!
    """
    
    # Build context for Claude
    system_prompt = """You are AIRGO, a friendly and helpful flight booking assistant for Nigerian travelers.

CRITICAL RULES FOR SHOWING FLIGHTS:
1. When you have flight data, show it IMMEDIATELY at the start of your response
2. Present flights in a clean, scannable format (numbered list)
3. AFTER showing all flights, then add brief recommendations or tips
4. Never make the user scroll through paragraphs before seeing flight options

BOOKING FLOW:
When user wants to book a flight:
1. If they select a specific flight, acknowledge it and ask for details
2. Required details: Full name, Phone number, Email, Payment method
3. After receiving details, confirm the booking with all information
4. **CRITICAL**: When showing booking confirmation, use the EXACT booking reference provided in the context
   - Never make up your own booking reference format
   - Use exactly what's provided: "Booking Reference: AIRGO-XXXXXX"
   - Do not create different formats like "AR-W3401-OE789" or similar

Response Format When You Have Flights:
Option 1: [Airline] ([Flight Number])
  [Origin] → [Destination]
  Departs: [Time] | Arrives: [Time]
  Duration: [Duration]
  Price: [Price] | [Seats] seats left

[Brief helpful comment after all flights - max 2-3 sentences]

Booking Confirmation Format (USE EXACT REFERENCE PROVIDED):
🎉 Booking Confirmed!

Booking Reference: [USE EXACT REFERENCE FROM CONTEXT]
E-ticket sent to [email]

Flight Details:
[Airline] [Flight Number]
[Route and times]

Next Steps:
• Check-in opens 24 hours before departure
• Arrive at airport 2 hours early
• Bring valid ID

Keep responses:
- Data first, commentary second
- Clean and scannable
- Action-oriented
- Use Nigerian context (Naira prices, local airports)
- Remember previous conversation context
- Handle booking requests professionally
- Use EXACT booking references provided, never create your own"""

    # Build the conversation messages for Claude
    messages = []
    
    # Add conversation history if available
    if conversation_history:
        for msg in conversation_history:
            if msg.get('role') and msg.get('content'):
                # Skip adding the current message since we'll add it separately
                if msg['content'] != user_message:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
    
    # Add flight data to the current context if available
    current_context = user_message
    if flight_data and len(flight_data) > 0:
        flight_info = "\n\n[FLIGHT DATA FOUND - Present these to the user naturally]\n\n"
        for i, flight in enumerate(flight_data[:5], 1):  # Show max 5 flights
            flight_info += f"Option {i}: {flight.airline} ({flight.flight_number})\n"
            flight_info += f"  {flight.origin} → {flight.destination}\n"
            flight_info += f"  Departs: {flight.departure_time} | Arrives: {flight.arrival_time}\n"
            flight_info += f"  Duration: {flight.duration}\n"
            flight_info += f"  Price: {flight.price}"
            if flight.available_seats:
                flight_info += f" | {flight.available_seats} seats left"
            flight_info += "\n\n"
        
        current_context += flight_info
    elif flight_data is not None and len(flight_data) == 0:
        current_context += "\n\n[NO FLIGHTS FOUND - Apologize and ask user to adjust search parameters]"
    
    # Add current message
    messages.append({
        "role": "user",
        "content": current_context
    })

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=messages
        )
        
        return message.content[0].text
        
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return "I'm having trouble processing your request right now. Please try again!"


# ============================================
# BOOKING HELPERS
# ============================================
def detect_booking_intent(message: str, conversation_history: List[dict] = None) -> tuple:
    """
    Detects if user wants to book a flight
    Returns: (wants_to_book, flight_info)
    """
    message_lower = message.lower()
    
    booking_keywords = ['book', 'reserve', 'buy', 'purchase', 'i want to book', 'book this', 'book that']
    wants_to_book = any(keyword in message_lower for keyword in booking_keywords)
    
    # Extract flight reference if mentioned
    flight_info = None
    if wants_to_book:
        # Look for airline and flight number patterns
        import re
        flight_pattern = r'([A-Z][a-z]+\s+(?:Air|Airlines?))\s+(?:flight\s+)?([A-Z0-9]+)'
        match = re.search(flight_pattern, message, re.IGNORECASE)
        if match:
            flight_info = {
                'airline': match.group(1),
                'flight_number': match.group(2)
            }
    
    return wants_to_book, flight_info


def extract_booking_details(message: str) -> dict:
    """
    Extracts passenger details from user message
    Returns dict with name, phone, email, payment_method
    """
    import re
    
    details = {}
    
    # Extract name (look for "name:" or similar patterns)
    name_pattern = r'(?:name|full name|passenger):\s*([A-Za-z\s]+?)(?:\n|,|$)'
    name_match = re.search(name_pattern, message, re.IGNORECASE)
    if name_match:
        details['passenger_name'] = name_match.group(1).strip()
    
    # Extract phone (Nigerian format)
    phone_pattern = r'\+?234\s?\d{3}\s?\d{3}\s?\d{4}|\d{11}'
    phone_match = re.search(phone_pattern, message)
    if phone_match:
        details['phone'] = phone_match.group(0).strip()
    
    # Extract email
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    email_match = re.search(email_pattern, message)
    if email_match:
        details['email'] = email_match.group(0).strip()
    
    # Extract payment method
    payment_keywords = {
        'card': ['card', 'debit', 'credit', 'mastercard', 'visa'],
        'bank transfer': ['transfer', 'bank transfer'],
        'ussd': ['ussd', 'dial']
    }
    
    message_lower = message.lower()
    for method, keywords in payment_keywords.items():
        if any(kw in message_lower for kw in keywords):
            details['payment_method'] = method
            break
    
    return details


def generate_booking_reference() -> str:
    """Generate a unique booking reference"""
    import random
    import string
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"AIRGO-{code}"


# ============================================
# INTENT DETECTION
# ============================================
def detect_flight_search_intent(message: str, conversation_history: List[dict] = None):
    """
    Detects if user is searching for flights and extracts details
    Returns: (is_search, origin, destination, date)
    """
    message_lower = message.lower()
    
    # Nigerian cities/airports
    nigerian_cities = {
        'lagos': 'LOS',
        'abuja': 'ABV',
        'port harcourt': 'PHC',
        'ph': 'PHC',
        'kano': 'KAN',
        'enugu': 'ENU',
        'calabar': 'CBQ',
        'owerri': 'QOW',
        'jos': 'JOS',
        'ibadan': 'IBA',
        'benin': 'BNI',
        'benin city': 'BNI',
        'kaduna': 'KAD',
        'maiduguri': 'MIU',
        'sokoto': 'SKO',
        'ilorin': 'ILR',
        'akure': 'AKR',
        'warri': 'QRW',
        'yola': 'YOL',
    }
    
    # Check if this is a flight search - expanded keywords
    search_keywords = ['flight', 'fly', 'book', 'ticket', 'travel', 'trip', 'show me', 'find', 'search', 'get me']
    is_search = any(keyword in message_lower for keyword in search_keywords)
    
    # Also check if user mentions two cities (likely a flight search)
    cities_mentioned = [city for city in nigerian_cities.keys() if city in message_lower]
    if len(cities_mentioned) >= 2:
        is_search = True
    
    # Extract origin and destination with better logic
    origin = None
    destination = None
    
    # Method 1: Look for "from X to Y" pattern
    if 'from' in message_lower and 'to' in message_lower:
        from_idx = message_lower.index('from')
        to_idx = message_lower.index('to')
        
        for city, code in nigerian_cities.items():
            city_idx = message_lower.find(city)
            if city_idx > from_idx and city_idx < to_idx:
                origin = code
            elif city_idx > to_idx:
                destination = code
    
    # Method 2: Look for airport codes (ABV, LOS, etc.)
    for code in nigerian_cities.values():
        if code.lower() in message_lower:
            code_idx = message_lower.find(code.lower())
            if 'from' in message_lower and code_idx > message_lower.index('from'):
                if 'to' not in message_lower or code_idx < message_lower.index('to'):
                    origin = code
            if 'to' in message_lower and code_idx > message_lower.index('to'):
                destination = code
    
    # Method 3: If we found cities but no clear origin/destination, use position
    if not origin or not destination:
        for city, code in nigerian_cities.items():
            if city in message_lower:
                if not origin:
                    origin = code
                elif not destination and code != origin:
                    destination = code
    
    # Check conversation history for context (e.g., user mentioned route earlier)
    if conversation_history and (not origin or not destination):
        for msg in reversed(conversation_history):
            if msg.get('role') == 'user':
                hist_msg = msg.get('content', '').lower()
                for city, code in nigerian_cities.items():
                    if city in hist_msg:
                        if not origin:
                            origin = code
                        elif not destination and code != origin:
                            destination = code
                if origin and destination:
                    break
    
    # Extract date with more patterns
    date = None
    if 'tomorrow' in message_lower:
        date = 'tomorrow'
    elif 'today' in message_lower:
        date = 'today'
    elif 'next week' in message_lower:
        date = 'next week'
    # Look for date patterns like "from the 11th", "the 11th", "11th", "nov 11", "november 11"
    elif 'from the' in message_lower or 'on the' in message_lower or 'the ' in message_lower:
        import re
        # Pattern for "11th", "21st", "3rd", etc
        date_pattern = r'\b(\d{1,2})(?:st|nd|rd|th)\b'
        match = re.search(date_pattern, message_lower)
        if match:
            day = match.group(1)
            # Assume current month if not specified
            from datetime import datetime
            current_year = datetime.now().year
            current_month = datetime.now().month
            # If "next week" or future reference, assume next month if day has passed
            if int(day) < datetime.now().day:
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
            date = f"{current_year}-{current_month:02d}-{int(day):02d}"
    else:
        # Look for month name patterns
        import re
        month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{1,2})'
        match = re.search(month_pattern, message_lower)
        if match:
            month_name = match.group(1)
            day = match.group(2)
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = month_map.get(month_name)
            if month:
                from datetime import datetime
                year = datetime.now().year
                # If month has passed this year, assume next year
                if month < datetime.now().month:
                    year += 1
                date = f"{year}-{month:02d}-{int(day):02d}"
    
    # Debug output
    if is_search:
        print(f"[INTENT] Detected flight search: {origin} → {destination} on {date or 'unspecified date'}")
    
    return is_search, origin, destination, date


# ============================================
# MAIN API ENDPOINT
# ============================================
@app.post("/api/chat")
async def chat(message: ChatMessage):
    """
    Main endpoint that receives user messages and returns AI responses
    Now with conversation memory AND booking capability!
    """
    try:
        user_msg = message.message
        history = message.history or []
        
        # Generate a simple session ID from history length (in production, use proper session management)
        session_id = f"session_{len(history) % 1000}"
        
        # Get or create booking state for this session
        if session_id not in booking_sessions:
            booking_sessions[session_id] = BookingState()
        
        booking_state = booking_sessions[session_id]
        
        # Check if user wants to book
        wants_to_book, flight_info = detect_booking_intent(user_msg, history)
        
        # Handle booking flow
        if wants_to_book and flight_info:
            # User selected a flight to book
            booking_state.flight_selected = flight_info
            booking_state.step = "awaiting_details"
            print(f"📝 Booking initiated: {flight_info}")
        
        # Check if user is providing booking details
        booking_details = extract_booking_details(user_msg)
        if booking_details and booking_state.step == "awaiting_details":
            # Update booking state with provided details
            for key, value in booking_details.items():
                setattr(booking_state, key, value)
            
            # Check if we have all required details
            if all([booking_state.passenger_name, booking_state.phone, 
                   booking_state.email, booking_state.payment_method]):
                # Generate booking reference
                booking_state.booking_reference = generate_booking_reference()
                booking_state.step = "confirming"
                print(f"✅ Booking ready for confirmation: {booking_state.booking_reference}")
                
                # Create confirmation message with clear instructions
                confirmation_context = f"\n\n[BOOKING CONFIRMED - USE EXACT DETAILS BELOW]\n"
                confirmation_context += f"Booking Reference: {booking_state.booking_reference}\n"
                confirmation_context += f"Flight: {booking_state.flight_selected.get('airline')} {booking_state.flight_selected.get('flight_number')}\n"
                confirmation_context += f"Passenger: {booking_state.passenger_name}\n"
                confirmation_context += f"Phone: {booking_state.phone}\n"
                confirmation_context += f"Email: {booking_state.email}\n"
                confirmation_context += f"Payment: {booking_state.payment_method}\n"
                confirmation_context += "\nIMPORTANT: Show the user a booking confirmation with EXACTLY this booking reference: {booking_state.booking_reference}"
                confirmation_context += "\nFormat the response as:"
                confirmation_context += "\n🎉 Booking Confirmed!"
                confirmation_context += f"\n\nBooking Reference: {booking_state.booking_reference}"
                confirmation_context += f"\nE-ticket sent to {booking_state.email}"
                confirmation_context += "\n\nFlight Details:"
                confirmation_context += f"\n{booking_state.flight_selected.get('airline')} {booking_state.flight_selected.get('flight_number')}"
                confirmation_context += "\n[Include departure/arrival times if known]"
                confirmation_context += "\n\nNext Steps:"
                confirmation_context += "\n• Check-in opens 24 hours before departure"
                confirmation_context += "\n• Arrive at airport 2 hours early"
                confirmation_context += "\n• Bring valid ID"
                
                user_msg += confirmation_context
        
        # Detect if user is searching for flights (existing functionality)
        is_search, origin, dest, date = detect_flight_search_intent(user_msg, history)
        
        flight_data = None
        if is_search and origin and dest:
            # Check if we have Amadeus credentials
            if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
                print("Warning: Amadeus API credentials not set, using demo mode")
                flight_data = get_demo_flights(origin, dest)
            else:
                # Get real flight data from Amadeus
                print(f"🔍 Searching Amadeus for flights: {origin} → {dest} on {date or 'tomorrow'}")
                flight_data = await amadeus.search_flights(origin, dest, date)
                
                if not flight_data:
                    print("⚠️  No results from Amadeus, trying demo data")
                    flight_data = get_demo_flights(origin, dest)
                else:
                    print(f"✅ Found {len(flight_data)} flights from Amadeus")
        elif is_search and not (origin and dest):
            print(f"⚠️  Flight search detected but missing route info: origin={origin}, dest={dest}")
        
        # Process with Claude AI, passing conversation history and booking context
        ai_response = await process_with_claude(user_msg, flight_data, history)
        
        return {"response": ai_response}
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def get_demo_flights(origin: str = None, dest: str = None) -> List[FlightInfo]:
    """Demo flights for testing when Amadeus is not configured"""
    return [
        FlightInfo(
            airline="Air Peace",
            flight_number="P47123",
            origin=origin or "LOS",
            destination=dest or "ABV",
            departure_time="08:30 AM",
            arrival_time="09:45 AM",
            price="₦85,000",
            currency="NGN",
            duration="1h 15m",
            available_seats=12
        ),
        FlightInfo(
            airline="Arik Air",
            flight_number="W3401",
            origin=origin or "LOS",
            destination=dest or "ABV",
            departure_time="02:15 PM",
            arrival_time="03:30 PM",
            price="₦92,000",
            currency="NGN",
            duration="1h 15m",
            available_seats=8
        ),
        FlightInfo(
            airline="Dana Air",
            flight_number="9J204",
            origin=origin or "LOS",
            destination=dest or "ABV",
            departure_time="05:45 PM",
            arrival_time="07:00 PM",
            price="₦78,000",
            currency="NGN",
            duration="1h 15m",
            available_seats=15
        ),
    ]


@app.get("/")
async def read_root():
    return {
        "service": "AIRGO API",
        "status": "running",
        "version": "1.0.0",
        "description": "AI-powered flight booking microservice"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
