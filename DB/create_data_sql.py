import os
import sqlite3
import pandas as pd
import shutil
from datetime import datetime

# Đường dẫn
DATABASE_PATH = "app/db/DB_SQL/travel2.sqlite"
BACKUP_PATH = "app/db/DB_SQL/travel2.backup.sqlite"
TABLES_DIR = "DB/Tables_Data"
DB_DIR = "app/db/DB_SQL"

def ensure_db_directory():
    """Tạo thư mục DB nếu chưa có"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)
        print(f"📁 Created directory: {DB_DIR}")

def clean_old_databases():
    """Xóa database cũ"""
    files_to_remove = [DATABASE_PATH, BACKUP_PATH]
    removed_files = []
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(os.path.basename(file_path))
                print(f"🗑️  Removed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
    
    if removed_files:
        print(f"✅ Cleaned {len(removed_files)} old database files")
    else:
        print("📝 No old database files to remove")

def get_csv_files():
    """Lấy danh sách file CSV"""
    if not os.path.exists(TABLES_DIR):
        print(f"❌ CSV directory not found: {TABLES_DIR}")
        return []
    
    csv_files = [f for f in os.listdir(TABLES_DIR) if f.endswith('.csv')]
    csv_files.sort()
    
    print(f"\n📁 Found {len(csv_files)} CSV files to import:")
    total_size = 0
    
    for i, file in enumerate(csv_files, 1):
        file_path = os.path.join(TABLES_DIR, file)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        
        size_str = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f}MB"
        print(f"   {i:2d}. {file:<25} ({size_str})")
    
    total_size_str = f"{total_size/1024:.1f}KB" if total_size < 1024*1024 else f"{total_size/(1024*1024):.1f}MB"
    print(f"📊 Total size: {total_size_str}")
    
    return csv_files

def create_table_from_csv(db_conn, table_name, csv_file_path):
    """Tạo bảng từ CSV file"""
    try:
        # Đọc CSV
        df = pd.read_csv(csv_file_path)
        
        # Tạo bảng và import dữ liệu
        df.to_sql(table_name, db_conn, if_exists='replace', index=False)
        
        print(f"✅ {table_name:<25}: {len(df):>6,} rows imported")
        return True, len(df)
        
    except Exception as e:
        print(f"❌ {table_name:<25}: Error - {e}")
        return False, 0

def get_database_info(db_path):
    """Hiển thị thông tin database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Lấy danh sách bảng
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\n📊 DATABASE SUMMARY:")
        print("-" * 50)
        total_rows = 0
        
        for (table_name,) in sorted(tables):
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            total_rows += row_count
            print(f"📋 {table_name:<25}: {row_count:>8,} rows")
        
        print("-" * 50)
        print(f"📊 Total: {len(tables)} tables, {total_rows:,} rows")
        
        # File size
        file_size = os.path.getsize(db_path)
        size_str = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f}MB"
        print(f"💾 Database size: {size_str}")
        
        conn.close()
        return len(tables), total_rows
        
    except Exception as e:
        print(f"❌ Error getting database info: {e}")
        return 0, 0

def create_database_backup(source_path, backup_path):
    """Tạo file backup"""
    try:
        shutil.copy(source_path, backup_path)
        print(f"💾 Created backup: {os.path.basename(backup_path)}")
        return True
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False

def main():
    """Hàm chính tạo database mới"""
    print("🆕 CREATE NEW DATABASE FROM CSV FILES")
    print("=" * 60)
    
    # Tạo thư mục DB
    ensure_db_directory()
    
    # Hiển thị thông tin hiện tại
    print(f"📂 Target database: {DATABASE_PATH}")
    print(f"📂 Backup location: {BACKUP_PATH}")
    print(f"📂 CSV source: {TABLES_DIR}")
    
    # Lấy danh sách CSV
    csv_files = get_csv_files()
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    # Xác nhận
    print(f"\n⚠️  This will:")
    print(f"   🗑️  DELETE existing databases")
    print(f"   🆕 CREATE new database from {len(csv_files)} CSV files")
    print(f"   💾 CREATE backup copy")
    
    confirm = input(f"\n❓ Continue? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("❌ Operation cancelled.")
        return
    
    # Bước 1: Xóa database cũ
    print(f"\n🧹 STEP 1: Cleaning old databases...")
    clean_old_databases()
    
    # Bước 2: Tạo database mới
    print(f"\n🆕 STEP 2: Creating new database...")
    print("-" * 50)
    
    try:
        # Tạo database mới
        conn = sqlite3.connect(DATABASE_PATH)
        
        success_count = 0
        total_rows = 0
        failed_tables = []
        
        # Import từng CSV file
        for csv_file in csv_files:
            table_name = csv_file.replace('.csv', '')
            csv_file_path = os.path.join(TABLES_DIR, csv_file)
            
            success, row_count = create_table_from_csv(conn, table_name, csv_file_path)
            if success:
                success_count += 1
                total_rows += row_count
            else:
                failed_tables.append(table_name)
        
        # Commit và đóng
        conn.commit()
        conn.close()
        
        # Kết quả import
        print("-" * 50)
        print(f"✅ Successfully imported: {success_count}/{len(csv_files)} tables")
        print(f"📊 Total rows: {total_rows:,}")
        
        if failed_tables:
            print(f"❌ Failed tables: {', '.join(failed_tables)}")
        
        # Bước 3: Tạo backup
        print(f"\n💾 STEP 3: Creating backup...")
        if create_database_backup(DATABASE_PATH, BACKUP_PATH):
            print("✅ Backup created successfully")
        
        # Hiển thị kết quả cuối
        print(f"\n🎉 NEW DATABASE CREATED!")
        print("=" * 50)
        
        # Thông tin database chính
        print(f"📊 Main database: {os.path.basename(DATABASE_PATH)}")
        get_database_info(DATABASE_PATH)
        
        # Xác nhận backup
        if os.path.exists(BACKUP_PATH):
            backup_size = os.path.getsize(BACKUP_PATH)
            size_str = f"{backup_size/1024:.1f}KB" if backup_size < 1024*1024 else f"{backup_size/(1024*1024):.1f}MB"
            print(f"\n💾 Backup file: {os.path.basename(BACKUP_PATH)} ({size_str})")
        
        print(f"\n✅ Ready to use!")
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")

def verify_database():
    """Kiểm tra database đã tạo"""
    print("🔍 DATABASE VERIFICATION")
    print("=" * 40)
    
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Main database not found: {DATABASE_PATH}")
        return
    
    if not os.path.exists(BACKUP_PATH):
        print(f"❌ Backup database not found: {BACKUP_PATH}")
        return
    
    print(f"✅ Main database: {os.path.basename(DATABASE_PATH)}")
    tables, rows = get_database_info(DATABASE_PATH)
    
    print(f"\n✅ Backup database: {os.path.basename(BACKUP_PATH)}")
    backup_tables, backup_rows = get_database_info(BACKUP_PATH)
    
    if tables == backup_tables and rows == backup_rows:
        print(f"\n✅ Main and backup databases are identical!")
    else:
        print(f"\n⚠️  Main and backup databases differ!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_database()
    else:
        main()
        
    print(f"\n💡 Usage:")
    print(f"   python {sys.argv[0]}        # Create new database")
    print(f"   python {sys.argv[0]} verify # Verify created database")