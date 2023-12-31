
from sqlalchemy import create_engine, Column, String, Integer, UniqueConstraint
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()
class Storyes(Base):
    __tablename__ = 'Storyes'
    __table_args__ = (
        UniqueConstraint('storyid', name='unique_id'),
    )

    storyid1 = Column(Integer, primary_key=True, autoincrement=True)
    storyText = Column(String, nullable=False)
    storyVector = Column(String, nullable=False)

if __name__ == '__main__':

 postgresql_url = 'postgresql://postgres:12345@localhost:5432/psycho'
 engine = create_engine(postgresql_url)
 Session = sessionmaker(bind=engine)
 session = Session()
 # Создаем таблицу
 Base.metadata.create_all(engine)
 session.commit()
