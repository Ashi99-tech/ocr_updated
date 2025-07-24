from sqlalchemy import Column, Integer, String
from database import Base

class OCRText(Base):
    __tablename__ = "ocr_texts"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
