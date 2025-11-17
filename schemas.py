"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Entry(BaseModel):
    """
    Journal entries collection schema
    Collection name: "entry" (lowercase of class name)
    """
    title: str = Field(..., description="Short title for the entry")
    content: str = Field(..., description="Free-form text content of the entry")
    mood: Optional[int] = Field(None, ge=1, le=5, description="Mood score from 1 (low) to 5 (high)")
    tags: Optional[List[str]] = Field(default=None, description="List of tags or topics")
    date: Optional[datetime] = Field(default=None, description="When this entry happened. Defaults to now if not provided")

class Feedback(BaseModel):
    """
    Optional user feedback on summaries or reflections
    Collection name: "feedback"
    """
    period_type: str = Field(..., description="weekly | monthly | yearly")
    period_start: datetime = Field(..., description="Start of the summarized period (UTC)")
    helpful: bool = Field(..., description="Whether the summary was helpful")
    comment: Optional[str] = Field(None, description="Additional notes")
