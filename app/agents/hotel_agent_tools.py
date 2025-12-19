import sqlite3
from app.config import settings
from datetime import date, datetime
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import pytz

@tool
def search_hotels(
    airport_code: str | None = None,
    city: str | None = None,
    min_star: int | None = None,
    max_star: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """Tim kiem khach san theo san bay, thanh pho va hang sao."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT
        h.hotel_id,
        h.hotel_name,
        h.airport_code,
        a.airport_name,
        a.city,
        h.address,
        h.star_rating
    FROM hotels h
    LEFT JOIN airports_data a
        ON h.airport_code = a.airport_code
    WHERE 1 = 1
    """
    params = []

    if airport_code:
        query += " AND h.airport_code = ?"
        params.append(airport_code.upper())

    if city:
        query += " AND UPPER(a.city) LIKE UPPER(?)"
        params.append(f"%{city}%")

    if min_star:
        query += " AND h.star_rating >= ?"
        params.append(min_star)

    if max_star:
        query += " AND h.star_rating <= ?"
        params.append(max_star)

    query += " ORDER BY h.star_rating DESC LIMIT ?"
    params.append(min(limit, 50))

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    results = [dict(zip(cols, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def get_hotel_details(hotel_id: int) -> dict | None:
    """Lay thong tin chi tiet khach san."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT
        h.hotel_id,
        h.hotel_name,
        h.address,
        h.star_rating,
        h.airport_code,
        a.airport_name,
        a.city
    FROM hotels h
    LEFT JOIN airports_data a
        ON h.airport_code = a.airport_code
    WHERE h.hotel_id = ?
    """
    cursor.execute(query, (hotel_id,))
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:
        return None

    cols = [c[0] for c in cursor.description]
    return dict(zip(cols, row))

@tool
def list_hotel_room_types(hotel_id: int) -> list[dict]:
    """Danh sach loai phong cua khach san."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT
        room_type_id,
        room_name,
        base_price,
        max_guests,
        total_rooms
    FROM hotel_room_types
    WHERE hotel_id = ?
    ORDER BY base_price ASC
    """
    cursor.execute(query, (hotel_id,))
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    results = [dict(zip(cols, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def create_hotel_booking(
    room_type_id: int,
    checkin_date: date,
    checkout_date: date,
    config: RunnableConfig,
) -> dict:
    """Dat phong khach san cho nguoi dung."""
    if checkout_date <= checkin_date:
        raise ValueError("Ngay checkout phai sau ngay checkin.")

    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    if not user_id:
        raise ValueError("Khong co ID nguoi dung duoc cau hinh.")

    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Lay gia phong
    cursor.execute(
        "SELECT base_price FROM hotel_room_types WHERE room_type_id = ?",
        (room_type_id,)
    )
    row = cursor.fetchone()
    if not row:
        raise ValueError("Loai phong khong ton tai.")

    base_price = row[0]
    nights = (checkout_date - checkin_date).days
    total_price = base_price * nights

    cursor.execute(
        """
        INSERT INTO hotel_bookings (
            user_id,
            room_type_id,
            booking_date,
            checkin_date,
            checkout_date,
            total_price
        )
        VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
        """,
        (user_id, room_type_id, checkin_date, checkout_date, total_price)
    )

    booking_id = cursor.lastrowid
    conn.commit()

    cursor.close()
    conn.close()

    return {
        "booking_id": booking_id,
        "room_type_id": room_type_id,
        "checkin_date": str(checkin_date),
        "checkout_date": str(checkout_date),
        "total_price": total_price,
    }

@tool
def get_user_hotel_bookings(config: RunnableConfig) -> list[dict]:
    """Lay lich su dat phong khach san cua nguoi dung."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    if not user_id:
        raise ValueError("Khong co ID nguoi dung duoc cau hinh.")

    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT
        hb.booking_id,
        h.hotel_name,
        rt.room_name,
        hb.checkin_date,
        hb.checkout_date,
        hb.total_price
    FROM hotel_bookings hb
    JOIN hotel_room_types rt
        ON hb.room_type_id = rt.room_type_id
    JOIN hotels h
        ON rt.hotel_id = h.hotel_id
    WHERE hb.user_id = ?
    ORDER BY hb.checkin_date DESC
    """
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    results = [dict(zip(cols, row)) for row in rows]

    cursor.close()
    conn.close()
    return results

@tool
def cancel_hotel_booking(booking_id: int) -> dict:
    """Huy dat phong khach san."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM hotel_bookings WHERE booking_id = ?",
        (booking_id,)
    )

    if cursor.rowcount == 0:
        raise ValueError("Khong tim thay booking.")

    conn.commit()
    cursor.close()
    conn.close()

    return {"status": "cancelled", "booking_id": booking_id}
