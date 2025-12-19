import sqlite3
from app.config import settings
from datetime import date, datetime
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import pytz

@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Lay tat ca ve may bay cua nguoi dung cung voi thong tin chuyen bay va vi tri ghe ngoi tuong ung."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None) 
    if not user_id:
        raise ValueError("Khong co ID hanh khach duoc cau hinh.")

    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT 
            t.ticket_no, t.book_ref,
            f.flight_id, f.flight_no,
            f.status,
            f.scheduled_departure, f.scheduled_arrival,
            f.actual_departure, f.actual_arrival,
            f.departure_airport, f.arrival_airport,
            bp.seat_no,
            tf.fare_conditions,
            dep_airport.airport_name AS departure_airport_name,
            dep_airport.city AS departure_city,
            arr_airport.airport_name AS arrival_airport_name,
            arr_airport.city AS arrival_city
        FROM tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        LEFT JOIN boarding_passes bp 
            ON bp.ticket_no = t.ticket_no 
            AND bp.flight_id = f.flight_id
        LEFT JOIN airports_data dep_airport 
            ON f.departure_airport = dep_airport.airport_code
        LEFT JOIN airports_data arr_airport 
            ON f.arrival_airport = arr_airport.airport_code
        WHERE t.user_id = ?
        ORDER BY f.scheduled_departure ASC
    """
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    # Format ket qua voi ten san bay
    for result in results:
        # Tao ten hien thi cho san bay khoi hanh
        dep_name = result.get('departure_airport_name')
        if dep_name:
            result['departure_display'] = f"{dep_name} ({result['departure_airport']})"
        else:
            result['departure_display'] = result['departure_airport']
        
        # Tao ten hien thi cho san bay den
        arr_name = result.get('arrival_airport_name') 
        if arr_name:
            result['arrival_display'] = f"{arr_name} ({result['arrival_airport']})"
        else:
            result['arrival_display'] = result['arrival_airport']

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
    """Tim kiem chuyen bay dua tren san bay khoi hanh, san bay den va khoang thoi gian khoi hanh."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Query voi JOIN de lay ten san bay
    query = """
    SELECT 
        f.*,
        dep_airport.airport_name as departure_airport_name,
        dep_airport.city as departure_city,
        arr_airport.airport_name as arrival_airport_name,
        arr_airport.city as arrival_city
    FROM flights f
    LEFT JOIN airports_data dep_airport ON f.departure_airport = dep_airport.airport_code
    LEFT JOIN airports_data arr_airport ON f.arrival_airport = arr_airport.airport_code
    WHERE 1 = 1
    """
    params = []

    if departure_airport:
        # Tim kiem ca ma san bay va ten
        query += """ AND (f.departure_airport = ? 
                        OR UPPER(dep_airport.airport_name) LIKE UPPER(?) 
                        OR UPPER(dep_airport.city) LIKE UPPER(?))"""
        search_term = f"%{departure_airport}%"
        params.extend([departure_airport.upper(), search_term, search_term])
        
    if arrival_airport:
        query += """ AND (f.arrival_airport = ? 
                        OR UPPER(arr_airport.airport_name) LIKE UPPER(?) 
                        OR UPPER(arr_airport.city) LIKE UPPER(?))"""
        search_term = f"%{arrival_airport}%"
        params.extend([arrival_airport.upper(), search_term, search_term])
        
    if start_time:
        query += " AND f.scheduled_departure >= ?"
        params.append(start_time)
    if end_time:
        query += " AND f.scheduled_departure <= ?"
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
    """Cap nhat ve cua nguoi dung sang chuyen bay moi hop le."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    if not user_id:
        raise ValueError("")

    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "ID chuyen bay moi khong hop le."
    
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    
    # Xử lý múi giờ - kiểm tra định dạng thời gian
    timezone = pytz.timezone("Asia/Ho_Chi_Minh")  # Đổi timezone phù hợp với Việt Nam
    current_time = datetime.now(tz=timezone)
    
    try:
        # Thử parse với định dạng ISO có timezone
        departure_time = datetime.fromisoformat(new_flight_dict["scheduled_departure"].replace('Z', '+00:00'))
        if departure_time.tzinfo is None:
            departure_time = timezone.localize(departure_time)
    except:
        # Fallback cho định dạng khác
        departure_time = datetime.strptime(
            new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
        )
    
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"Khong duoc phep doi sang chuyen bay cach thoi diem hien tai it hon 3 gio. Chuyen bay da chon khoi hanh luc {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "Khong tim thay ve hien co cho so ve da cung cap."

    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND user_id = ?",
        (ticket_no, user_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Hanh khach hien tai dang dang nhap voi ID {user_id} khong phai la chu so huu ve {ticket_no}"

    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ve da duoc cap nhat thanh cong sang chuyen bay moi."

@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Huy ve cua nguoi dung va xoa khoi co so du lieu."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    if not user_id:
        raise ValueError("Khong co ID hanh khach duoc cau hinh.")
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "Khong tim thay ve hien co cho so ve da cung cap."

    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND user_id = ?",
        (ticket_no, user_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Hanh khach hien tai dang dang nhap voi ID {user_id} khong phai la chu so huu ve {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ve da duoc huy thanh cong."
