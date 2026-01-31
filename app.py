import os
import random
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BrainBites API",
    description="API for accessing book summaries and information from BooksDB",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request parameters", "errors": exc.errors()}
    )

# Load env
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)
if not os.path.exists(env_path):
    load_dotenv(os.path.join(os.path.dirname(script_dir), '.env'))
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "brain_bites_books_db"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Cache storage
cache: Dict[str, Dict[str, Any]] = {
    "daily_digest": {"data": None, "timestamp": None},
    "categories": {"data": None, "timestamp": None},
    "authors": {"data": None, "timestamp": None}
}
CACHE_DURATION = timedelta(hours=24)
BOOK_CATEGORIES = [
    "Business & Money", "Self-Help & Personal Development", "Psychology",
    "Health & Wellness", "Leadership & Management", "Communication & Social Skills",
    "Science & Technology", "Career Development", "Motivation & Inspiration",
    "Mindfulness & Spirituality", "Marketing & Sales", "Economics",
    "Creativity & Innovation", "Other"   
]

# --- Helper Functions ---

def _is_cache_valid(cache_key: str) -> bool:
    if cache_key not in cache:
        return False
    entry = cache[cache_key]
    if entry["data"] is None or entry["timestamp"] is None:
        return False
    return (datetime.now() - entry["timestamp"]) < CACHE_DURATION

def _get_from_cache(cache_key: str) -> Optional[Any]:
    return cache[cache_key]["data"] if _is_cache_valid(cache_key) else None

def _set_cache(cache_key: str, data: Any) -> None:
    cache[cache_key] = {"data": data, "timestamp": datetime.now()}

def _parse_list_field(value) -> List[str]:
    if not value: return []
    if isinstance(value, list): return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip(): return []
    return [item.strip() for item in value.split(",") if item.strip()]

def _convert_book_from_supabase(row: dict) -> dict:
    return {
        "book_id": int(row.get("book_id", 0)),
        "title": row.get("title", ""),
        "subtitle": row.get("subtitle", ""),
        "authors": _parse_list_field(row.get("authors", "")),
        "publication_date": row.get("publication_date", ""),
        "language": row.get("language", ""),
        "categories": _parse_list_field(row.get("primary_category", "")),
        "description": row.get("description", ""),
        "cover_image_url": row.get("cover_image_url", ""),
        "average_rating": float(row.get("average_rating", 0) or 0) if row.get("average_rating") else 0.0,
        "ratings_count": int(row.get("ratings_count", 0) or 0) if row.get("ratings_count") else 0,
        "brief_summary": row.get("brief_summary", ""),
        "detailed_summary": row.get("detailed_summary", ""),
        "key_takeaways": _parse_list_field(row.get("key_takeaways", "")),
        "notable_quotes": _parse_list_field(row.get("notable_quotes", "")),
        "read_time": int(row.get("read_time", 0) or 0) if row.get("read_time") else 0,
        "created_at": row.get("created_at", ""),
        "updated_at": row.get("updated_at", "")
    }

# --- Models ---

class Book(BaseModel):
    book_id: int
    title: str
    subtitle: Optional[str] = ""
    authors: List[str]
    publication_date: Optional[str] = ""
    language: Optional[str] = ""
    categories: List[str]
    description: Optional[str] = ""
    cover_image_url: Optional[str] = ""
    average_rating: Optional[float] = 0.0
    ratings_count: Optional[int] = 0
    brief_summary: Optional[str] = ""
    detailed_summary: Optional[str] = ""
    key_takeaways: List[str]
    notable_quotes: List[str]
    read_time: Optional[int] = 0
    created_at: Optional[str] = ""
    updated_at: Optional[str] = ""

# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "BrainBites API",
        "version": "1.0.0",
        "endpoints": {
            "all_books": "/books",
            "book_by_id": "/books/{book_id}",
            "search": "/books/search?q={query}",
            "by_category": "/books/category/{category}",
            "by_author": "/books/author/{author}",
            "popular": "/books/popular",
            "trending": "/books/trending",
            "top_rated": "/books/top-rated",
            "daily_digest": "/daily-digest",
            "about": "/about"
        }
    }

@app.get("/books", response_model=List[Book])
async def get_all_books(
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum average rating")
):
    try:
        query = supabase.table(TABLE_NAME).select("*")
        
        # Apply filters directly in DB query
        if min_rating is not None:
            query = query.gte("average_rating", min_rating)
        
        if category:
            # Assumes categories are stored as comma-sep strings or text
            query = query.ilike("primary_category", f"%{category}%")
            
        # Apply pagination at DB level
        final_limit = limit if limit else 100 # Default limit to prevent overflow if not specified
        query = query.range(offset, offset + final_limit - 1)
        
        response = query.execute()
        return [_convert_book_from_supabase(row) for row in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching books: {str(e)}")

@app.get("/books/search", response_model=List[Book])
async def search_books(
    q: str = Query(..., description="Search query"),
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results")
):
    query_trimmed = q.strip() if q else ""
    if not query_trimmed:
        return []
    
    try:
        # Optimized: Use Supabase 'or' filter for text search instead of fetching all
        # This searches title, authors, description, or category
        search_filter = (
            f"title.ilike.%{query_trimmed}%,"
            f"authors.ilike.%{query_trimmed}%,"
            f"description.ilike.%{query_trimmed}%,"
            f"primary_category.ilike.%{query_trimmed}%"
        )
        
        query = supabase.table(TABLE_NAME).select("*").or_(search_filter)
        
        if limit:
            query = query.limit(limit)
        else:
            query = query.limit(50) # Reasonable default for search
            
        response = query.execute()
        
        return [_convert_book_from_supabase(row) for row in response.data]
    except Exception as e:
        logger.error(f"[SEARCH ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Error searching books: {str(e)}")

@app.get("/books/{book_id}", response_model=Book)
async def get_book_by_id(book_id: int):
    try:
        response = supabase.table(TABLE_NAME).select("*").eq("book_id", book_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        return _convert_book_from_supabase(response.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching book: {str(e)}")

@app.get("/books/category/{category}", response_model=List[Book])
async def get_books_by_category(
    category: str,
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results")
):
    try:
        query = supabase.table(TABLE_NAME).select("*").ilike("primary_category", f"%{category}%")
        
        if limit:
            query = query.limit(limit)
            
        response = query.execute()
        return [_convert_book_from_supabase(row) for row in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching books by category: {str(e)}")

@app.get("/books/author/{author}", response_model=List[Book])
async def get_books_by_author(
    author: str,
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results")
):
    try:
        query = supabase.table(TABLE_NAME).select("*").ilike("authors", f"%{author}%")
        
        if limit:
            query = query.limit(limit)
            
        response = query.execute()
        return [_convert_book_from_supabase(row) for row in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching books by author: {str(e)}")

@app.get("/categories", response_model=List[str])
async def get_all_categories():
    try:
        if cached := _get_from_cache("categories"):
            return cached
        
        # Static categories provided, so we stick to them. 
        # If you wanted DB categories: 
        # response = supabase.table(TABLE_NAME).select("primary_category").execute()
        # But per requirements, we keep logic same.
        result = BOOK_CATEGORIES.copy()
        _set_cache("categories", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching categories: {str(e)}")

@app.get("/authors", response_model=List[str])
async def get_all_authors():
    try:
        if cached := _get_from_cache("authors"):
            return cached
        
        # Optimization: Fetch ONLY the authors column, not the whole book
        response = supabase.table(TABLE_NAME).select("authors").execute()
        
        authors = set()
        for row in response.data:
            # Re-use the existing parse logic
            row_authors = _parse_list_field(row.get("authors", ""))
            authors.update(row_authors)
        
        result = sorted(list(authors))
        _set_cache("authors", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching authors: {str(e)}")

@app.get("/books/popular", response_model=List[Book])
async def get_popular_books(
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results")
):
    try:
        # Optimized: DB Sort
        query = supabase.table(TABLE_NAME).select("*")\
            .order("ratings_count", desc=True)\
            .order("average_rating", desc=True)
            
        if limit:
            query = query.limit(limit)
        else:
            query = query.limit(100) # Default limit for safety
            
        response = query.execute()
        return [_convert_book_from_supabase(row) for row in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular books: {str(e)}")

@app.get("/books/trending", response_model=List[Book])
async def get_trending_books(
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results")
):
    try:
        # Optimization Strategy:
        # We can't do the complex Python math (rating * 0.6...) easily in standard Supabase call.
        # BUT we can fetch a smaller subset of candidates (e.g. rating >= 4.0) 
        # and THEN sort in Python, instead of fetching the whole DB.
        
        fetch_limit = 100 # Fetch pool of candidates
        if limit and limit > 100: fetch_limit = limit * 2

        response = supabase.table(TABLE_NAME).select("*")\
            .gte("average_rating", 4.0)\
            .gt("ratings_count", 0)\
            .order("updated_at", desc=True)\
            .limit(fetch_limit)\
            .execute()
            
        books = [_convert_book_from_supabase(row) for row in response.data]
        
        # Python-side complex sort on the reduced dataset
        sorted_books = sorted(
            books,
            key=lambda x: (x["average_rating"] * 0.6 + min(x["ratings_count"] / 1000, 1) * 0.4),
            reverse=True
        )
        
        if limit:
            sorted_books = sorted_books[:limit]
            
        return sorted_books
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trending books: {str(e)}")

@app.get("/books/top-rated", response_model=List[Book])
async def get_top_rated_books(
    limit: Optional[int] = Query(None, ge=1, description="Limit the number of results"),
    min_ratings: Optional[int] = Query(10, ge=0, description="Minimum number of ratings required")
):
    try:
        # Optimized: DB Filter & Sort
        query = supabase.table(TABLE_NAME).select("*")\
            .gte("ratings_count", min_ratings)\
            .order("average_rating", desc=True)\
            .order("ratings_count", desc=True)
            
        if limit:
            query = query.limit(limit)
        else:
            query = query.limit(100)
            
        response = query.execute()
        return [_convert_book_from_supabase(row) for row in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top-rated books: {str(e)}")

@app.get("/daily-digest", response_model=Book)
async def get_daily_digest():
    try:
        if cached := _get_from_cache("daily_digest"):
            return cached
        
        # Optimized Random Selection:
        # 1. Get count of eligible books (rating > 4.0) instead of fetching all
        count_res = supabase.table(TABLE_NAME).select("*", count="exact", head=True).gt("average_rating", 4.0).execute()
        total_eligible = count_res.count
        
        if total_eligible == 0:
            raise HTTPException(status_code=404, detail="No suitable books found")

        # 2. Pick random index
        random_offset = random.randint(0, total_eligible - 1)
        
        # 3. Fetch only that one book
        response = supabase.table(TABLE_NAME).select("*")\
            .gt("average_rating", 4.0)\
            .range(random_offset, random_offset)\
            .execute()
            
        if not response.data:
            # Fallback (shouldn't happen given logic above)
            raise HTTPException(status_code=404, detail="Error selecting random book")
            
        book = _convert_book_from_supabase(response.data[0])
        _set_cache("daily_digest", book)
        return book
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching daily digest: {str(e)}")

@app.get("/about")
async def get_about():
    return {
        "app_name": "BrainBites API",
        "version": "1.0.0",
        "description": "API for accessing book summaries and information from BooksDB",
        "features": [
            "Book summaries and detailed information",
            "Search functionality",
            "Category and author filtering",
            "Popular, trending, and top-rated books",
            "Daily book recommendations",
            "Comprehensive book database"
        ],
        "database": "Supabase",
        "framework": "FastAPI",
        "endpoints": {
            "all_books": "/books",
            "book_by_id": "/books/{book_id}",
            "search": "/books/search?q={query}",
            "by_category": "/books/category/{category}",
            "by_author": "/books/author/{author}",
            "popular": "/books/popular",
            "trending": "/books/trending",
            "top_rated": "/books/top-rated",
            "daily_digest": "/daily-digest",
            "categories": "/categories",
            "authors": "/authors",
            "about": "/about"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )